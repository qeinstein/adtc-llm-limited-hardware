"""P0 scheduler sweep: poll/thread flags on the LOCKED config.

All JOIN4b CLI runs forced --poll 0 (no polling; llama.cpp default is
--poll 50). Recent Qwen3.5 CPU work reports 13-21% worker time in
barrier/tail bubbles with the largest single win from eliminating a
scheduler usleep-backoff oversleep. Before touching weights, sweep the
scheduler flags on a SMALL warm resident matrix.

Model: NO re-transcode. Uses the JOIN4b v1 Q2K artifact (dataset mount),
verified by size + sha256 + a functional gate that reproduces the JOIN4b
pid-21 n=80 output hash exactly.

Stages (same binary/model, warm, pid 21 "Hello.", seed 1021, K4/16,
N_GEN=192, 3 interleaved reps; outputs must stay bit-identical):
  GATE    resident t=4 poll=0 n=80 -> artifact hash gate + page-cache warmup
  A       resident t=4, poll in {0, 25, 50, 100}
  B       best poll, threads in {1, 2, 3, 4}
  C       ONE affinity A/B at winner (default vs physical-core taskset),
          only if HT is present and taskset can isolate winner threads
  D       winner config on the <3GB bounded arm (755 slots, 80 pinL2 pins)
Decision: >=5% gain LOCK; 2-5% keep if robust; <2% stop flag tuning.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-edge0sched-v1")
OUT = WORK / "native-sparse-edge0sched-v1-results"
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
CLI = BUILD / "bin" / "llama-cli"

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
# JOIN4b v1 Q2K artifact, republished as a dataset (NO re-transcode).
DATASET_SLUG = "edge0-q2k-experts"
Q2K_NAME = "Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"
Q2K_SIZE = 12262341600  # JOIN4b v1 artifact bytes (layout-verified)
Q2K_SHA256 = "0f3698ae92f91db2eb10a3650bdb6693ff8cfaf7060cbc5f256845c700c7603b"
# JOIN4b pid-21 (n=80, t=4, poll 0, seed 1021) response hash (r1==r2).
EXPECTED_PID21_N80_SHA = "099f8728ef2db3b494f8a69ca9ea612331ca47342abc0e99a33e831041b7ce7e"

K1, K2 = 4, 16
PID = 21
PROMPT = "Hello."
N_GEN = 192
REPS = 3
SETTLE_SEC = 10
BASELINE = {"threads": 4, "poll": 0}  # JOIN4b setting
POLLS_A = [0, 25, 50, 100]
THREADS_B = [1, 2, 3, 4]
B3 = {"name": "bounded_3gb", "bounded": True, "slots": 755, "pins": "3.0"}
RES = {"name": "resident", "bounded": False}

# Embedded pin sets (one key per line), injected at kernel build from
# cache_config_k4.json (LOCKED; transfer-validated).
PINS_3 = "@@PINS_3@@"
PINS_4 = "@@PINS_4@@"
PINS_5 = "@@PINS_5@@"
PINS_6 = "@@PINS_6@@"

# JOIN4 executor C (staged + pins + section timers + weight buckets),
# injected at build. Identical source to JOIN4b v2.
JOIN4_C = "@@JOIN4_C@@"

K2_NORM_OLD = '''        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);'''

K2_NORM_NEW = '''        ggml_tensor * weights_sum = nullptr;
        if (edge0_k2 == edge0_k1) {
            weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        } else {
            // reference mass over top-k2 (paper 2609.04575 Eq.2 denominator)
            ggml_tensor * sel_k2 = ggml_argsort_top_k(ctx0, selection_probs, (int) edge0_k2); // [k2, T]
            ggml_tensor * rk2 = ggml_get_rows(ctx0, probs, sel_k2); // [1, k2, T]
            rk2 = ggml_reshape_2d(ctx0, rk2, edge0_k2, n_tokens); // [k2, T]
            weights_sum = ggml_sum_rows(ctx0, rk2); // [1, T]
        }
        cb(weights_sum, "ffn_moe_weights_sum", il);'''

K1K2_CODE = '''
    // ---- edge0 k1/k2 (env; default native) ----
    int64_t edge0_k1 = n_expert_used;
    int64_t edge0_k2 = n_expert_used;
    if (const char * e1 = getenv("GGML_MOE_K1")) { int v = atoi(e1); if (v > 0) edge0_k1 = v; }
    if (const char * e2 = getenv("GGML_MOE_K2")) { int v = atoi(e2); if (v > 0) edge0_k2 = v; }
    if (edge0_k2 < edge0_k1) edge0_k2 = edge0_k1;
    if (edge0_k2 > n_expert) edge0_k2 = n_expert;
    n_expert_used = edge0_k1;
'''


def run_checked(cmd, cwd=None, log=None):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       check=False)
    print(p.stdout[-3000:], flush=True)
    if log:
        Path(log).write_text(p.stdout, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"command exited {p.returncode}: {cmd}")
    return p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(8 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def replace_once(path, old, new):
    text = Path(path).read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"patch anchor mismatch in {path}")
    Path(path).write_text(text.replace(old, new), encoding="utf-8")


def drop_file_cache(path):
    os.sync()
    if not hasattr(os, "posix_fadvise"):
        return {"available": False}
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return {"available": True, "called": True}
    except OSError as exc:
        return {"available": True, "called": False, "error": repr(exc)}
    finally:
        os.close(fd)


def proc_sample(pid):
    row = {"mono_ns": time.monotonic_ns(), "rss_kib": 0, "rss_anon_kib": 0,
           "rss_file_kib": 0, "read_bytes": 0, "rchar": 0, "minflt": 0,
           "majflt": 0, "utime": 0, "stime": 0, "valid": False}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"): row["rss_kib"] = int(line.split()[1])
            elif line.startswith("RssAnon:"): row["rss_anon_kib"] = int(line.split()[1])
            elif line.startswith("RssFile:"): row["rss_file_kib"] = int(line.split()[1])
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("read_bytes", "rchar"): row[key] = int(value.strip())
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        row["minflt"] = int(stat[7])
        row["majflt"] = int(stat[9])
        row["utime"] = int(stat[11])
        row["stime"] = int(stat[12])
        row["valid"] = True
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        pass
    return row


def cpu_pct_from_samples(samples):
    """Mean process CPU% over sampler lifetime (100 = 1 core fully busy)."""
    valid = [x for x in samples if x.get("valid")]
    if len(valid) < 2:
        return None
    a, b = valid[0], valid[-1]
    dticks = (b["utime"] + b["stime"]) - (a["utime"] + a["stime"])
    dsec = (b["mono_ns"] - a["mono_ns"]) / 1e9
    if dsec <= 0:
        return None
    hz = os.sysconf("SC_CLK_TCK")
    return 100.0 * (dticks / hz) / dsec


PERF_RE = re.compile(r"eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) runs\s*\(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\s*\)")
PREFILL_RE = re.compile(r"prompt eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) (?:tokens|runs)")
SUMMARY_RE = re.compile(r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]")
CACHE_RE = re.compile(r"PHASE6_BOUNDED_CACHE ([^\n]*)")
PROF_RE = re.compile(r"PHASE6_PROFILE ([^\n]*)")
KV_RE = re.compile(r"([a-z_]+)=([0-9]+)")


def parse_perf(text):
    detailed = PERF_RE.findall(text)
    if detailed:
        elapsed, runs, ms, tps = detailed[-1]
        out = {"ms_per_token": float(ms), "tokens_per_second": float(tps),
               "eval_ms": float(elapsed), "eval_runs": int(runs),
               "perf_source": "detailed"}
    else:
        summary = SUMMARY_RE.findall(text)
        if not summary:
            raise RuntimeError("no parseable llama performance line")
        prompt, generation = summary[-1]
        g = float(generation)
        out = {"prompt_tokens_per_second": float(prompt),
               "tokens_per_second": g,
               # Prefill-only runs print Generation: 0.00 (no decode happened,
               # not an error): never divide by zero here.
               "ms_per_token": (1000.0 / g) if g > 0 else float("inf"),
               "perf_source": "summary"}
    pre = PREFILL_RE.findall(text)
    if pre:
        ms_p, n_p = pre[-1]
        out.update({"prefill_ms": float(ms_p), "prefill_tokens": int(n_p)})
    return out


def fill_perf_from_prof(perf, prof):
    """Summary-only CLI output lacks eval wall/runs; derive exactly from
    the profiler's decode-graph count (eval_ms = ms/tok x dec_graphs)."""
    if "eval_ms" not in perf:
        if prof["dec_graphs"] > 0:
            perf["eval_ms"] = perf["ms_per_token"] * prof["dec_graphs"]
            perf["eval_runs"] = prof["dec_graphs"]
        else:
            perf["eval_ms"] = 0.0
            perf["eval_runs"] = 0
    return perf


def prefill_c_ms(prof):
    """Prefill wall from C section counters (exact, both arms)."""
    return sum(prof[f"pre_{k}_ns"] for k in
               ("attn", "gdn", "moe_rest", "expert_node", "shared",
                "lmhead", "misc")) / 1e6


def parse_kv_list(text):
    return {k: int(v) for k, v in KV_RE.findall(text)}


def response_payload(stdout, prompt):
    marker = stdout.find("[Start thinking]")
    if marker < 0:
        marker = stdout.rfind(prompt) + len(prompt)
    tail = stdout[marker:]
    perf = SUMMARY_RE.search(tail)
    if perf: tail = tail[:perf.start()]
    if tail.rstrip().endswith("Exiting..."): tail = tail.rstrip()[:-len("Exiting...")]
    return tail.strip()


def setup_runtime():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not LLAMA.exists():
        run_checked(["git", "clone", "--filter=blob:none",
                     "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    head = run_checked(["git", "rev-parse", "HEAD"], cwd=LLAMA).stdout.strip()
    print(f"built base: {head} (want {LLAMA_COMMIT})", flush=True)
    assert head == LLAMA_COMMIT, f"NOT on pin: {head}"
    # K1K2 graph patch (s-kernel/tracek4-proven anchors on this pin)
    g = LLAMA / "src" / "llama-graph.cpp"
    replace_once(g, "#include <cstring>\n#include <numeric>",
                 "#include <cstdlib>\n#include <cstring>\n#include <numeric>")
    replace_once(g,
                 "    ggml_tensor * logits = nullptr;\n\n    if (probs_in == nullptr) {",
                 K1K2_CODE + "\n    ggml_tensor * logits = nullptr;\n\n    if (probs_in == nullptr) {")
    replace_once(g,
                 "    const uint32_t n_expert_used_il = hparams.n_expert_used(il);",
                 "    const uint32_t n_expert_used_il = (uint32_t) n_expert_used; "
                 "// edge0: follows k1 (uniform arch; native-identical unset)")
    replace_once(g, K2_NORM_OLD, K2_NORM_NEW)
    # Lazy expert tensors (Qwen3.6 shares the qwen35moe arch file on this pin)
    qwen = LLAMA / "src" / "models" / "qwen35moe.cpp"
    replace_once(qwen,
                 '        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);\n'
                 '        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);',
                 '        const int expert_flags = flags | TENSOR_READ_LAZY;\n'
                 '        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, expert_flags);\n'
                 '        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, expert_flags);')
    # Loader: register lazy tensors with the executor (env-gated)
    loader = LLAMA / "src" / "llama-model-loader.cpp"
    replace_once(loader, "#include <cstring>\n", "#include <cstring>\n#include <cstdlib>\n")
    replace_once(loader, "#include <regex>\n",
                 '#include <regex>\n\nextern "C" void ggml_cpu_phase6_register_lazy_tensor(const struct ggml_tensor *, int, size_t, size_t, size_t);\n')
    replace_once(loader,
                 "        size_t n_size = ggml_nbytes(cur);\n\n        const bool from_mapping = use_mmap || lazy.has(cur);",
                 "        size_t n_size = ggml_nbytes(cur);\n\n"
                 '        if (lazy.has(cur) && std::getenv("GGML_PHASE6_BOUNDED_CACHE") != nullptr) {\n'
                 "            const auto & file = files.at(weight->idx);\n"
                 "            ggml_cpu_phase6_register_lazy_tensor(cur, file->file_id(), weight->offs, n_size, cur->nb[2]);\n"
                 "        }\n\n"
                 "        const bool from_mapping = use_mmap || lazy.has(cur);")
    # ggml-cpu: JOIN4 executor + route trace + MUL_MAT_ID redirect + timers
    cpu = LLAMA / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
    replace_once(cpu, "static struct ggml_state g_state = {0};\n",
                 JOIN4_C + "\nstatic struct ggml_state g_state = {0};\n")
    replace_once(cpu,
                 "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
                 "        return;\n"
                 "    }\n"
                 "\n"
                 "    // extra_buffer op?\n",
                 "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
                 "        return;\n"
                 "    }\n"
                 "\n"
                 "    ggml_phase6_route_trace(params, tensor);\n"
                 "\n"
                 "    // extra_buffer op?\n")
    replace_once(cpu,
                 "    const int ith = params->ith;\n"
                 "    const int nth = params->nth;\n\n"
                 "    const enum ggml_type type = src0->type;\n\n"
                 "    const bool src1_cont = ggml_is_contiguous(src1);\n",
                 "    const int ith = params->ith;\n"
                 "    const int nth = params->nth;\n\n"
                 "    const enum ggml_type type = src0->type;\n\n"
                 "    const bool src1_cont = ggml_is_contiguous(src1);\n"
                 "    const bool phase6 = phase6_enabled();\n")
    replace_once(cpu,
                 "    // reset current_chunk\n"
                 "    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {\n"
                 "        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);\n"
                 "        *current_chunk_ctr = nth;\n"
                 "    }\n\n"
                 "    ggml_barrier(params->threadpool);\n\n"
                 "    for (int cur_a = 0; cur_a < n_as; ++cur_a) {",
                 "    // reset current_chunk\n"
                 "    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {\n"
                 "        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);\n"
                 "        *current_chunk_ctr = nth;\n"
                 "    }\n\n"
                 "    ggml_barrier(params->threadpool);\n\n"
                 "    if (phase6 && ith == 0) phase6_prepare(src0, ids);\n"
                 "    ggml_barrier(params->threadpool);\n\n"
                 "    for (int cur_a = 0; cur_a < n_as; ++cur_a) {")
    replace_once(cpu,
                 "        const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n",
                 "        const char * src0_cur = phase6 ? phase6_tensor_ptr(src0, cur_a)\n"
                 "            : (const char *) src0->data + cur_a * nb02;\n")
    replace_once(cpu,
                 "        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time\n"
                 "        // Try fused ops, fall back to normal compute\n",
                 "        if (state->ith == 0 && join4_prof_on()) join4_node_start(node, node_n);\n"
                 "        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time\n"
                 "        // Try fused ops, fall back to normal compute\n")
    replace_once(cpu,
                 "        if (node_n + 1 < cgraph->n_nodes) {\n"
                 "            ggml_barrier(state->threadpool);\n"
                 "        }\n"
                 "    }\n"
                 "\n"
                 "#ifdef GGML_USE_OPENMP",
                 "        if (node_n + 1 < cgraph->n_nodes) {\n"
                 "            ggml_barrier(state->threadpool);\n"
                 "        }\n"
                 "    }\n"
                 "\n"
                 "    if (state->ith == 0 && join4_prof_on()) join4_graph_end();\n"
                 "\n"
                 "#ifdef GGML_USE_OPENMP")
    # iqp: selected-expert source redirect
    iqp = LLAMA / "ggml" / "src" / "ggml-cpu" / "iqp.cpp"
    replace_once(iqp, '#include "iqp.h"\n',
                 '#include "iqp.h"\n\nextern "C" const char * ggml_cpu_phase6_iqp_source(const struct ggml_tensor *, int64_t);\n')
    replace_once(iqp,
                 "    const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n",
                 "    const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n"
                 "    const char * phase6_src0_cur = ggml_cpu_phase6_iqp_source(src0, cur_a);\n"
                 "    if (phase6_src0_cur != NULL) src0_cur = phase6_src0_cur;\n")
    # precise CLI timings
    cli = LLAMA / "tools" / "cli" / "cli-context.cpp"
    replace_once(cli,
                 "[ Prompt: %.1f t/s | Generation: %.1f t/s ]",
                 "[ Prompt: %.6f t/s | Generation: %.6f t/s ]")
    patch = run_checked(["git", "diff"], cwd=LLAMA).stdout
    (OUT / "sched-runtime.patch").write_text(patch, encoding="utf-8")
    return {"llama_commit": LLAMA_COMMIT, "built_head": head,
            "k1": K1, "k2": K2,
            "runtime_patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}


def build():
    run_checked(["cmake", "-S", str(LLAMA), "-B", str(BUILD),
                 "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON",
                 "-DLLAMA_CURL=ON"],
                log=OUT / "cmake-configure.log")
    # No quantize target: the model comes from the dataset mount.
    run_checked(["cmake", "--build", str(BUILD), "--config", "Release",
                 "-j4", "--target", "llama-cli"],
                log=OUT / "cmake-build.log")


def mount_model():
    """Locate + verify the JOIN4b v1 Q2K artifact (no re-transcode)."""
    cand = Path("/kaggle/input") / DATASET_SLUG / Q2K_NAME
    if not cand.exists():
        have = sorted(str(p) for p in Path("/kaggle/input").rglob("*.gguf"))
        raise RuntimeError(f"dataset model missing: {cand}; have={have}")
    size = cand.stat().st_size
    if size != Q2K_SIZE:
        raise RuntimeError(f"model size {size} != artifact {Q2K_SIZE}")
    d = sha256(cand)
    if d != Q2K_SHA256:
        raise RuntimeError("model sha256 != JOIN4b v1 artifact")
    print(f"model OK: {cand} ({size/1e9:.2f}GB, sha {d[:16]}...)", flush=True)
    return {"dataset": DATASET_SLUG, "file": Q2K_NAME, "size_bytes": size,
            "sha256": d, "path": str(cand)}


def cpu_topology():
    """Physical cores + HT detection for the single affinity A/B."""
    info = {"ncpu": os.cpu_count(), "physical_cpus": [], "has_ht": False,
            "taskset": shutil.which("taskset")}
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return info
    seen, order = {}, []
    cur = {}
    for line in text.splitlines() + [""]:
        if line.strip():
            if ":" in line:
                k, v = line.split(":", 1)
                cur[k.strip()] = v.strip()
            continue
        if "processor" in cur:
            key = (cur.get("physical id", "0"), cur.get("core id", cur["processor"]))
            if key not in seen:
                seen[key] = int(cur["processor"])
                order.append(int(cur["processor"]))
        cur = {}
    info["physical_cpus"] = sorted(order)
    info["has_ht"] = len(order) < (info["ncpu"] or 0)
    return info


def run_case(arm, rep, model, pins_path, tag="", cold=False, n_gen=None,
             expect_decode=True, poll=0, threads=4, taskset=None):
    """One CLI run. Scheduler flags (poll/threads/taskset) vary; everything
    else matches JOIN4b exactly (same K4/16, temp, top-p, seed, ctx)."""
    name = arm["name"]
    aff = f"_aff{taskset.replace(',', '-')}" if taskset else ""
    prefix = f"{name}_t{threads}_poll{poll}{aff}_r{rep}{tag}"
    trace = OUT / f"{prefix}.routes.jsonl"
    stdout_path = OUT / f"{prefix}.stdout.txt"
    stderr_path = OUT / f"{prefix}.stderr.txt"
    samples_path = OUT / f"{prefix}.process.jsonl"
    time_path = OUT / f"{prefix}.time.txt"
    for path in (trace, stdout_path, stderr_path, samples_path, time_path):
        path.unlink(missing_ok=True)
    cli_cmd = [str(CLI), "-m", str(model), "-ngl", "0", "-t", str(threads),
               "-c", "512", "-n", str(N_GEN if n_gen is None else n_gen),
               "--temp", "0.7", "--top-p",
               "0.9", "--seed", str(1000 + PID), "--single-turn",
               "--no-display-prompt", "--no-warmup", "--perf", "-lm",
               "mmap", "-lzm", "on" if arm["bounded"] else "off",
               "--poll", str(poll), "-p", PROMPT]
    cmd = cli_cmd
    if shutil.which("/usr/bin/time"):
        cmd = ["/usr/bin/time", "-v", "-o", str(time_path)] + cmd
    if taskset:
        cmd = ["taskset", "-c", taskset] + cmd
    env = dict(os.environ, GGML_PHASE6_PROFILE="1",
               GGML_PHASE6_ROUTE_TRACE=str(trace),
               GGML_MOE_K1=str(K1), GGML_MOE_K2=str(K2))
    if arm["bounded"]:
        env.update(GGML_PHASE6_BOUNDED_CACHE="1",
                   GGML_PHASE6_SLOTS=str(arm["slots"]),
                   GGML_PHASE6_ASYNC="1",
                   GGML_PHASE6_PINS=str(pins_path))
    else:
        for k in ("GGML_PHASE6_BOUNDED_CACHE", "GGML_PHASE6_SLOTS",
                  "GGML_PHASE6_CACHE_BYTES", "GGML_PHASE6_ASYNC",
                  "GGML_PHASE6_PINS", "GGML_PHASE6_ZERO_COPY"):
            env.pop(k, None)
    if cold:
        cache_drop = drop_file_cache(model)
        os.sync()
        time.sleep(SETTLE_SEC)
    else:
        cache_drop = {"available": True, "called": False, "warm": True}
    start = time.monotonic_ns()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    samples = []
    stop = threading.Event()

    def sample_loop():
        while not stop.is_set():
            samples.append(proc_sample(proc.pid))
            stop.wait(0.02)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    out, err = proc.communicate()
    stop.set()
    sampler.join(timeout=2)
    samples.append(proc_sample(proc.pid))
    elapsed = (time.monotonic_ns() - start) / 1e9
    stdout_path.write_text(out, encoding="utf-8")
    stderr_path.write_text(err, encoding="utf-8")
    samples_path.write_text("".join(json.dumps(x) + "\n" for x in samples),
                            encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{prefix} exited {proc.returncode}: {err[-3000:]}")
    valid = [x for x in samples if x.get("valid")]
    if expect_decode:
        perf = parse_perf(out + "\n" + err)
    else:
        perf = {"perf_source": "diagnostic-prefill-only",
                "tokens_per_second": 0.0, "ms_per_token": float("inf"),
                "eval_ms": 0.0, "eval_runs": 0}
    cache_m = CACHE_RE.findall(err)
    prof_m = PROF_RE.findall(err)
    if not prof_m:
        raise RuntimeError(f"{prefix}: no PHASE6_PROFILE line (timers dead?)")
    prof = parse_kv_list(prof_m[-1])
    cache = parse_kv_list(cache_m[-1]) if cache_m else {}
    if expect_decode:
        fill_perf_from_prof(perf, prof)
    perf["prefill_c_ms"] = prefill_c_ms(prof)
    wb_unmatched = sorted(set(re.findall(r"PHASE6_WB_UNMATCHED (\S+)", err)))
    if arm["bounded"]:
        if not cache_m:
            raise RuntimeError(f"{prefix}: bounded arm has no cache line")
        if cache.get("slots", -1) != arm["slots"]:
            raise RuntimeError(f"{prefix}: slots {cache.get('slots')} != {arm['slots']}")
        exp_pins = {"3.0": 80, "4.0": 80, "5.0": 1346, "6.0": 915}[arm["pins"]]
        if cache.get("pins", -1) != exp_pins:
            raise RuntimeError(f"{prefix}: pins {cache.get('pins')} != {exp_pins}")
        if cache.get("pin_violations", -1) != 0:
            raise RuntimeError(f"{prefix}: pin violations!")
    else:
        if cache_m:
            raise RuntimeError(f"{prefix}: resident arm emitted cache line?!")
    if expect_decode and prof.get("dec_graphs", 0) <= 0:
        raise RuntimeError(f"{prefix}: no decode graphs counted")
    if prof.get("markers", 0) <= 0:
        raise RuntimeError(f"{prefix}: no section markers seen")
    payload = response_payload(out, PROMPT)
    time_txt = time_path.read_text(encoding="utf-8") if time_path.exists() else ""
    m_rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", time_txt)
    m_min = re.search(r"Minor \(reclaiming a frame\) page faults: (\d+)", time_txt)
    m_maj = re.search(r"Major \(requiring I/O\) page faults: (\d+)", time_txt)
    m_cpu = re.search(r"Percent of CPU this job got: (\d+)%", time_txt)
    return {"arm": name, "rep": rep, "threads": threads, "poll": poll,
            "taskset": taskset, "cold": cold, "elapsed_sec": elapsed,
            "cache_drop": cache_drop,
            "peak_rss_mib": max((x["rss_kib"] for x in valid), default=0) / 1024,
            "peak_rss_anon_mib": max((x["rss_anon_kib"] for x in valid), default=0) / 1024,
            "peak_rss_file_mib": max((x["rss_file_kib"] for x in valid), default=0) / 1024,
            "read_bytes_max": max((x["read_bytes"] for x in valid), default=0),
            "rchar_max": max((x["rchar"] for x in valid), default=0),
            "minflt_max": max((x["minflt"] for x in valid), default=0),
            "majflt_max": max((x["majflt"] for x in valid), default=0),
            "cpu_pct_sampler": cpu_pct_from_samples(samples),
            "time_maxrss_kib": int(m_rss.group(1)) if m_rss else None,
            "time_minflt": int(m_min.group(1)) if m_min else None,
            "time_majflt": int(m_maj.group(1)) if m_maj else None,
            "time_cpu_pct": int(m_cpu.group(1)) if m_cpu else None,
            "decode_perf": perf, "cache": cache, "prof": prof,
            "wb_unmatched": wb_unmatched,
            "response_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "trace_sha256": sha256(trace) if trace.exists() else None}


def cond_summary(runs):
    """Timing/profile means over a config's reps (all warm here)."""
    import statistics as st
    dec_toks = sum(r["prof"]["dec_graphs"] for r in runs)
    dec_wall_ms = sum(r["decode_perf"]["eval_ms"] for r in runs)
    tps_total = dec_toks / (dec_wall_ms / 1000)
    tps_runs = [r["decode_perf"]["tokens_per_second"] for r in runs]
    sec = {}
    for k in ("attn", "gdn", "moe_rest", "expert_node", "fetchprep",
              "shared", "lmhead", "misc"):
        sec[k] = sum(r["prof"][f"dec_{k}_ns"] for r in runs) / dec_toks / 1e6
    sec["expert_compute"] = sec["expert_node"] - sec["fetchprep"]
    sec["router_k2"] = sec["moe_rest"]
    sec["graph_sum"] = sum(sec[k] for k in
                           ("attn", "gdn", "moe_rest", "expert_node",
                            "shared", "lmhead", "misc"))
    wb = {}
    for k in ("exps", "attn", "gdn", "shexp", "router", "out", "other"):
        wb[k] = sum(r["prof"].get(f"wb_dec_{k}", 0) for r in runs) / dec_toks / 1e6
    wb["matmul_sum"] = sum(wb.values())
    cpu_s = [r["cpu_pct_sampler"] for r in runs if r["cpu_pct_sampler"]]
    cpu_t = [r["time_cpu_pct"] for r in runs if r["time_cpu_pct"]]
    return {"runs": len(runs), "dec_toks": dec_toks,
            "tps_total": tps_total, "ms_per_tok": 1000 / tps_total,
            "tps_mean_runs": st.mean(tps_runs),
            "tps_stdev_runs": st.stdev(tps_runs) if len(tps_runs) > 1 else 0.0,
            "tps_runs": tps_runs,
            "sections_ms_tok": sec, "wb_ms_tok": wb,
            "cpu_pct_sampler_mean": st.mean(cpu_s) if cpu_s else None,
            "time_cpu_pct_mean": st.mean(cpu_t) if cpu_t else None,
            "peak_rss_mib_max": max(r["peak_rss_mib"] for r in runs),
            "ttft_ms_mean": st.mean(r["prof"]["ttft_ns"] / 1e6 for r in runs),
            "prefill_c_ms_mean": st.mean(r["decode_perf"]["prefill_c_ms"] for r in runs)}


def run_stage(label, arm, model, pins_path, configs, n_gen, cold_first=False):
    """Interleaved reps: rep-major order cancels drift across configs."""
    runs = []
    for rep in range(1, REPS + 1):
        for cfg in configs:
            cold = cold_first and rep == 1 and cfg == configs[0]
            print(f"===== {label} t={cfg['threads']} poll={cfg['poll']}"
                  f"{' aff=' + cfg['taskset'] if cfg.get('taskset') else ''}"
                  f" rep={rep} {'cold' if cold else 'warm'} =====", flush=True)
            r = run_case(arm, rep, model, pins_path, cold=cold, n_gen=n_gen,
                         poll=cfg["poll"], threads=cfg["threads"],
                         taskset=cfg.get("taskset"))
            r["cfg"] = {k: cfg.get(k) for k in ("threads", "poll", "taskset")}
            runs.append(r)
            c = r["cache"]
            extra = (f" hit={c['dec_hits']/c['dec_requests']:.3f}"
                     if c else "")
            print(f"  {r['decode_perf']['tokens_per_second']:.2f} t/s "
                  f"rss={r['peak_rss_mib']:.0f}MiB{extra}", flush=True)
    return runs


def summarize_stage(runs):
    out = {}
    cfgs = sorted(set((r["threads"], r["poll"], r["taskset"]) for r in runs))
    for t, p, a in cfgs:
        key = f"t{t}_poll{p}" + (f"_aff{a}" if a else "")
        sub = [r for r in runs if (r["threads"], r["poll"], r["taskset"]) == (t, p, a)]
        out[key] = cond_summary(sub)
        out[key]["cfg"] = {"threads": t, "poll": p, "taskset": a}
    return out


def pick_best(summaries):
    return max(summaries.items(), key=lambda kv: kv[1]["tps_total"])[0]


def verdict(gain_pct, robust):
    if gain_pct >= 5.0:
        return "LOCK"
    if gain_pct >= 2.0:
        return "KEEP" if robust else "WEAK-KEEP?"
    return "STOP"


def main():
    started = time.time()
    runtime = setup_runtime()
    wfree = shutil.disk_usage(WORK).free / 1e9
    sfree = shutil.disk_usage(SCRATCH).free / 1e9
    print(f"disk free: WORK {wfree:.1f}GB SCRATCH {sfree:.1f}GB", flush=True)
    if wfree < 2 or sfree < 15:
        raise RuntimeError(f"disk too tight: WORK {wfree} SCRATCH {sfree}")
    build()
    topo = cpu_topology()
    hardware = {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                "topology": {k: v for k, v in topo.items() if k != "taskset"},
                "taskset_avail": bool(topo["taskset"]),
                "cpuinfo": Path("/proc/cpuinfo").read_text()[:6000],
                "meminfo": Path("/proc/meminfo").read_text()}
    (OUT / "hardware.json").write_text(json.dumps(hardware, indent=2))
    print(f"topology: ncpu={topo['ncpu']} physical={topo['physical_cpus']} "
          f"HT={topo['has_ht']} taskset={bool(topo['taskset'])}", flush=True)
    model = mount_model()
    q2k_path = Path(model["path"])
    # Locked b3 pins to file (counts asserted here + in C at runtime)
    p = OUT / "pins_3.0.txt"
    p.write_text(PINS_3 if PINS_3.endswith("\n") else PINS_3 + "\n")
    keys = [x for x in p.read_text().split() if x.strip()]
    assert len(keys) == 80, len(keys)
    assert all(0 <= int(k) < 10240 for k in keys)
    pins3 = p
    # GATE: reproduce the JOIN4b pid-21 n=80 output hash exactly (artifact
    # proof) + page-cache warmup for the resident sweep.
    print("===== GATE (resident t=4 poll=0 n=80 artifact+warmup) =====", flush=True)
    gate = run_case(RES, 0, q2k_path, None, tag="_gate", cold=False,
                    n_gen=80, poll=0, threads=4)
    print(f"  gate: {gate['decode_perf']['tokens_per_second']:.2f} t/s "
          f"rss={gate['peak_rss_mib']:.0f}MiB", flush=True)
    if gate["response_sha256"] != EXPECTED_PID21_N80_SHA:
        raise RuntimeError("GATE FAIL: dataset artifact output != JOIN4b; "
                           "wrong model bytes?")
    print("  gate: artifact hash MATCHES JOIN4b pid-21", flush=True)
    # Stage A: poll sweep at t=4
    print("===== STAGE A (resident t=4, poll sweep) =====", flush=True)
    runs_a = run_stage("A", RES, q2k_path, None,
                       [{"threads": 4, "poll": x} for x in POLLS_A], N_GEN)
    sum_a = summarize_stage(runs_a)
    base_key = f"t{BASELINE['threads']}_poll{BASELINE['poll']}"
    best_poll = pick_best(sum_a)
    for k, s in sum_a.items():
        g = 100 * (s["tps_total"] / sum_a[base_key]["tps_total"] - 1)
        print(f"  A {k}: {s['tps_total']:.2f} t/s ({s['ms_per_tok']:.1f} ms/tok) "
              f"stdev={s['tps_stdev_runs']:.2f} cpu={s['cpu_pct_sampler_mean']:.0f}% "
              f"gain={g:+.1f}%", flush=True)
    best_poll_val = sum_a[best_poll]["cfg"]["poll"]
    print(f"  A winner: poll={best_poll_val}", flush=True)
    # Stage B: thread sweep at best poll
    print(f"===== STAGE B (resident poll={best_poll_val}, thread sweep) =====",
          flush=True)
    runs_b = run_stage("B", RES, q2k_path, None,
                       [{"threads": t, "poll": best_poll_val} for t in THREADS_B],
                       N_GEN)
    sum_b = summarize_stage(runs_b)
    best_key = pick_best(sum_b)
    for k, s in sum_b.items():
        g = 100 * (s["tps_total"] / sum_a[base_key]["tps_total"] - 1)
        print(f"  B {k}: {s['tps_total']:.2f} t/s ({s['ms_per_tok']:.1f} ms/tok) "
              f"stdev={s['tps_stdev_runs']:.2f} cpu={s['cpu_pct_sampler_mean']:.0f}% "
              f"gain={g:+.1f}%", flush=True)
    win = sum_b[best_key]["cfg"]
    wsum = sum_b[best_key]
    bsum = sum_a[base_key]
    gain = 100 * (wsum["tps_total"] / bsum["tps_total"] - 1)
    robust = ((wsum["tps_total"] - wsum["tps_stdev_runs"])
              > (bsum["tps_total"] + bsum["tps_stdev_runs"]))
    dec = verdict(gain, robust)
    print(f"  B winner: t={win['threads']} poll={win['poll']} "
          f"{wsum['tps_total']:.2f} t/s gain={gain:+.1f}% robust={robust} "
          f"-> {dec}", flush=True)
    # Stage C: ONE affinity A/B at the winner (physical cores only if HT).
    runs_c, sum_c, aff_note = [], {}, "skipped"
    phys = topo["physical_cpus"]
    if topo["has_ht"] and topo["taskset"] and len(phys) >= win["threads"]:
        aff = ",".join(str(c) for c in phys[:win["threads"]])
        print(f"===== STAGE C (affinity A/B at winner, taskset {aff}) =====",
              flush=True)
        runs_c = run_stage("C", RES, q2k_path, None,
                           [{"threads": win["threads"], "poll": win["poll"],
                             "taskset": None},
                            {"threads": win["threads"], "poll": win["poll"],
                             "taskset": aff}], N_GEN)
        sum_c = summarize_stage(runs_c)
        for k, s in sum_c.items():
            print(f"  C {k}: {s['tps_total']:.2f} t/s "
                  f"stdev={s['tps_stdev_runs']:.2f}", flush=True)
        aff_note = f"taskset {aff}"
        best_c = pick_best(sum_c)
        cw = sum_c[best_c]
        cg = 100 * (cw["tps_total"] / wsum["tps_total"] - 1)
        print(f"  C winner: {best_c} gain={cg:+.1f}% vs no-affinity", flush=True)
        if best_c != f"t{win['threads']}_poll{win['poll']}":
            win = dict(cw["cfg"])
            wsum = cw
            print(f"  C UPDATES winner: {win}", flush=True)
    else:
        print(f"  C skipped: HT={topo['has_ht']} taskset={bool(topo['taskset'])} "
              f"nphys={len(phys)} need={win['threads']}", flush=True)
        aff_note = (f"skipped (HT={topo['has_ht']} "
                    f"taskset={bool(topo['taskset'])} nphys={len(phys)})")
    # Stage D: winner on the <3GB bounded arm (first rep cold per protocol).
    print(f"===== STAGE D (bounded_3gb winner t={win['threads']} "
          f"poll={win['poll']}) =====", flush=True)
    runs_d = run_stage("D", B3, q2k_path, pins3,
                       [{"threads": win["threads"], "poll": win["poll"],
                         "taskset": win.get("taskset")}], N_GEN,
                       cold_first=True)
    sum_d = summarize_stage(runs_d)
    for k, s in sum_d.items():
        print(f"  D {k}: {s['tps_total']:.2f} t/s ({s['ms_per_tok']:.1f} ms/tok) "
              f"rss={s['peak_rss_mib_max']:.0f}MiB", flush=True)
    # Bit-exactness gates: scheduler flags must not change outputs.
    res_runs = runs_a + runs_b + runs_c
    shas = set(r["response_sha256"] for r in res_runs)
    if len(shas) != 1:
        raise RuntimeError(f"resident sweep has {len(shas)} distinct outputs!")
    dsha = set(r["response_sha256"] for r in runs_d)
    if dsha != shas:
        raise RuntimeError("bounded_3gb output != resident output!")
    print("  gates: bit-exact across poll/threads/affinity/arms OK", flush=True)
    unmatched = sorted(set(u for r in res_runs + runs_d for u in r["wb_unmatched"]))
    print(f"  wb unmatched weights: {unmatched if unmatched else 'NONE'}",
          flush=True)
    result = {"schema": "native-sparse-edge0sched/v1", "status": "ok",
              "wb_unmatched": unmatched,
              "runtime": runtime, "model": model,
              "hardware": {k: v for k, v in hardware.items()
                           if k not in ("cpuinfo", "meminfo")},
              "decode": {"k1": K1, "k2": K2, "pid": PID,
                         "tokens_requested": N_GEN, "reps": REPS},
              "baseline": BASELINE, "winner": win,
              "gain_pct_vs_baseline": gain, "robust": robust,
              "decision": dec, "affinity_note": aff_note,
              "stage_a": sum_a, "stage_b": sum_b, "stage_c": sum_c,
              "stage_d": sum_d, "wall_sec": time.time() - started}
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print("== FINAL ==", flush=True)
    print(f"baseline t={BASELINE['threads']} poll={BASELINE['poll']}: "
          f"{bsum['tps_total']:.2f} t/s", flush=True)
    print(f"winner {win}: {wsum['tps_total']:.2f} t/s "
          f"gain={gain:+.1f}% -> {dec}", flush=True)
    for k, s in sum_d.items():
        print(f"bounded_3gb {k}: {s['tps_total']:.2f} t/s "
              f"rss={s['peak_rss_mib_max']:.0f}MiB", flush=True)
    print(json.dumps({"status": "ok", "winner": win, "decision": dec,
                      "gain_pct": gain}), flush=True)


if __name__ == "__main__":
    main()

