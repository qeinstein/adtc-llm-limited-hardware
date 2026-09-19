"""JOIN4b section repair: weight-bucket profile on the LOCKED config.

v2 finding: cb-name sections leaked ~110ms/tok into MISC (shexp matmuls
run before their "ffn_shexp" opener; TAIL markers never match). This
kernel adds weight-bucket timers (EVERY matmul by loader weight name:
immutable) + a nodelist ground-truth dump, and re-profiles resident +
locked b6 on 12 spread prompts. Arm comparison (tps/RSS/traffic) stands
from v2; ONLY the component split is repaired here.

Same binary/flags/seeds/N_GEN as JOIN4; arms resident x2 + b6 x2 on
PIDS (12 spread prompts). Nodelist ground truth on one short run.
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
SCRATCH = Path("/tmp/native-sparse-edge0join4b-v1")
OUT = WORK / "native-sparse-edge0join4b-v1-results"
PIDS = [0, 1, 4, 6, 7, 10, 12, 13, 15, 17, 18, 21]
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
CLI = BUILD / "bin" / "llama-cli"
QUANTIZE = BUILD / "bin" / "llama-quantize"

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
BASE = "3.6"
MODELS = {
    "3.6": {
        "repo": "a483e9e6cbd595906af30beda3187c2663a1118c",
        "file": "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
        "size": 10_756_586_464,
        "sha": "2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef",
        "hf": "unsloth/Qwen3.6-35B-A3B-GGUF",
    },
}
Q2K_NAME = "Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"
K1, K2 = 4, 16
THREADS = 4
N_GEN = 80
SETTLE_SEC = 10

PROMPTS = [
    ("medical", "A 3-year-old has watery diarrhoea and is thirsty. Explain how to assess dehydration and what home treatment to start."),
    ("medical", "A pregnant woman has headache, swollen face, and high blood pressure. What danger signs and referral steps apply?"),
    ("medical", "List the steps to prepare oral rehydration solution at home and when to seek care."),
    ("medical", "A newborn feels cold and feeds poorly. Describe warming, feeding, and referral actions."),
    ("medical", "Explain why deworming tablets are given at school and what side effects to watch for."),
    ("medical", "A child has fast breathing and chest indrawing. Outline assessment and urgent actions."),
    ("general", "Explain how rain forms and why some seasons are drier than others."),
    ("general", "Describe how a market day works in a small town, from morning to evening."),
    ("general", "Summarize the plot of a story about a fisherman who finds something unexpected."),
    ("general", "Compare travelling by bus and by motorcycle taxi for a long trip."),
    ("general", "Explain what a budget is and how a family can make one."),
    ("general", "Describe the steps to plant maize from land preparation to harvest."),
    ("swahili", "Eleza kwa kifupi kwa nini maji ya kunywa yenye chumvi na sukari husaidia mtoto mwenye kuharisha."),
    ("swahili", "Ni dalili zipi za hatari zinahitaji mtoto apelekwe hospitali haraka?"),
    ("swahili", "Andika mpango mfupi wa usafi wa mikono kwa shule ya kijijini."),
    ("swahili", "Tofautisha ukweli, maoni, na dhana kwa lugha rahisi ya Kiswahili."),
    ("swahili", "Mweleze mgonjwa kwa heshima kwa nini anatakiwa kufuata dozi aliyopewa."),
    ("instruction", "Answer in exactly three bullet points: how to prepare for a job interview."),
    ("instruction", "Give a two-sentence answer and one caveat about using online medical advice."),
    ("instruction", "Create a small table comparing prevention, diagnosis, and treatment."),
    ("instruction", "Ask one useful clarification question before answering an underspecified request."),
    ("short", "Hello."),
    ("short", "What should I do next?"),
]

ARMS = [
    {"name": "resident", "bounded": False, "reps": 2},
    {"name": "bounded_6gb", "bounded": True, "slots": 3662, "pins": "6.0", "reps": 2},
]

# Embedded pin sets (one key per line), injected at kernel build from
# cache_config_k4.json (LOCKED; transfer-validated).
PINS_3 = "@@PINS_3@@"
PINS_4 = "@@PINS_4@@"
PINS_5 = "@@PINS_5@@"
PINS_6 = "@@PINS_6@@"

# JOIN4 executor C (staged + pins + section timers), injected at build.
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
           "majflt": 0, "valid": False}
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
        row["valid"] = True
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
        pass
    return row


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
        out = {"prompt_tokens_per_second": float(prompt),
               "tokens_per_second": float(generation),
               "ms_per_token": 1000.0 / float(generation),
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
        perf["eval_ms"] = perf["ms_per_token"] * prof["dec_graphs"]
        perf["eval_runs"] = prof["dec_graphs"]
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
    (OUT / "join4-runtime.patch").write_text(patch, encoding="utf-8")
    return {"llama_commit": LLAMA_COMMIT, "built_head": head,
            "k1": K1, "k2": K2,
            "runtime_patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}


def build():
    run_checked(["cmake", "-S", str(LLAMA), "-B", str(BUILD),
                 "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON",
                 "-DLLAMA_CURL=ON"],
                log=OUT / "cmake-configure.log")
    run_checked(["cmake", "--build", str(BUILD), "--config", "Release",
                 "-j4", "--target", "llama-cli", "llama-quantize"],
                log=OUT / "cmake-build.log")


def fetch_model():
    m = MODELS[BASE]
    url = f"https://huggingface.co/{m['hf']}/resolve/{m['repo']}/{m['file']}"
    dest = SCRATCH / m["file"]
    if not dest.exists() or dest.stat().st_size != m["size"]:
        run_checked(["curl", "-L", "--fail", "--http1.1", "--retry", "5",
                     "--retry-all-errors", "--retry-delay", "5", "-C", "-",
                     "-o", str(dest), url],
                    log=OUT / "model-download.log")
    d = sha256(dest)
    if dest.stat().st_size != m["size"] or d != m["sha"]:
        raise RuntimeError("model identity mismatch")
    return {"base": BASE, "file": m["file"], "size_bytes": m["size"],
            "sha256": d, "repo_commit": m["repo"]}


GGML_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 9: "q8_1", 10: "q2_k", 11: "q3_k", 12: "q4_k",
    13: "q5_k", 14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs",
    18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s",
    23: "iq4_xs", 24: "i8", 25: "i16", 26: "i32", 27: "i64", 28: "f64",
    29: "iq1_m", 30: "bf16", 34: "tq1_0", 35: "tq2_0", 36: "mxfp4"}


def gguf_tensor_table(path):
    import struct as _st
    head = open(path, "rb").read(64 << 20)
    assert head[:4] == b"GGUF"
    n_tensors, n_kv = _st.unpack_from("<QQ", head, 8)
    off = 24

    def read_str(o):
        (n,) = _st.unpack_from("<Q", head, o)
        return head[o + 8:o + 8 + n].decode(), o + 8 + n

    _SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
               10: 8, 11: 8, 12: 8}

    def skip_value(o, typ):
        if typ in _SCALAR:
            return o + _SCALAR[typ]
        if typ == 8:
            _, o2 = read_str(o)
            return o2
        if typ == 9:
            (at,) = _st.unpack_from("<I", head, o)
            (n,) = _st.unpack_from("<Q", head, o + 4)
            o += 12
            if at == 8:
                for _ in range(n):
                    _, o = read_str(o)
                return o
            return o + {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                        10: 8, 11: 8, 12: 8}[at] * n
        raise RuntimeError(f"kv type {typ}")

    for _ in range(n_kv):
        _, off = read_str(off)
        (typ,) = _st.unpack_from("<I", head, off)
        off = skip_value(off + 4, typ)
    out = []
    for _ in range(n_tensors):
        name, off = read_str(off)
        (nd,) = _st.unpack_from("<I", head, off)
        off += 4 + 8 * nd
        (typ,) = _st.unpack_from("<I", head, off)
        off += 4 + 8
        out.append((name, GGML_TYPE_NAMES.get(typ, f"T{typ}")))
    return out


def transcode_q2k(model_iq2, out_path):
    table = gguf_tensor_table(model_iq2)
    lines, n = [], 0
    for name, typ in table:
        tgt = "q2_k" if "_exps" in name else typ
        n += tgt == "q2_k"
        lines.append(f"^{re.escape(name)}$={tgt}")
    (OUT / "overrides_q2k.txt").write_text("\n".join(lines) + "\n")
    print(f"transcode q2k: {n}/{len(lines)} -> q2_k", flush=True)
    t0 = time.time()
    run_checked([str(QUANTIZE), "--allow-requantize", "--tensor-type-file",
                 str(OUT / "overrides_q2k.txt"), str(model_iq2),
                 str(out_path), "Q8_0", str(THREADS)],
                log=OUT / "transcode_q2k.log")
    dt = time.time() - t0
    os.sync()
    time.sleep(60)
    drop_file_cache(out_path)
    os.sync()
    time.sleep(30)
    chk = gguf_tensor_table(out_path)
    nexp = sum(1 for x, t in chk if "_exps" in x and t == "q2_k")
    if nexp != 120:
        raise RuntimeError(f"q2k expert count {nexp} != 120")
    return {"size_bytes": out_path.stat().st_size,
            "elapsed_sec": dt, "q2k_experts": nexp}


# Sim predictions (tracka_k4_pareto.json) for on-device validation.
EXPECTED = {
    "bounded_3gb": {"hit": 0.5573, "miss_tok": 70.8},
    "bounded_4gb": {"hit": 0.7528, "miss_tok": 39.5},
    "bounded_5gb": {"hit": 0.8590, "miss_tok": 22.6},
    "bounded_6gb": {"hit": 0.9328, "miss_tok": 10.8},
}


def run_case(arm, pid, rep, model, pins_path, tag="", cold=True,
             nodelist=None, n_gen=None):
    """One CLI run. cold=True drops the file cache first (cold-start cost);
    warm runs (cold=False) measure steady state. The executor cache starts
    empty every run (fresh process) either way."""
    name = arm["name"]
    prefix = f"{name}_p{pid:02d}_r{rep}{tag}"
    trace = OUT / f"{prefix}.routes.jsonl"
    stdout_path = OUT / f"{prefix}.stdout.txt"
    stderr_path = OUT / f"{prefix}.stderr.txt"
    samples_path = OUT / f"{prefix}.process.jsonl"
    time_path = OUT / f"{prefix}.time.txt"
    for path in (trace, stdout_path, stderr_path, samples_path, time_path):
        path.unlink(missing_ok=True)
    category, prompt = PROMPTS[pid]
    cli_cmd = [str(CLI), "-m", str(model), "-ngl", "0", "-t", str(THREADS),
               "-c", "512", "-n", str(N_GEN if n_gen is None else n_gen),
               "--temp", "0.7", "--top-p",
               "0.9", "--seed", str(1000 + pid), "--single-turn",
               "--no-display-prompt", "--no-warmup", "--perf", "-lm",
               "mmap", "-lzm", "on" if arm["bounded"] else "off",
               "--poll", "0", "-p", prompt]
    if shutil.which("/usr/bin/time"):
        cmd = ["/usr/bin/time", "-v", "-o", str(time_path)] + cli_cmd
    else:
        cmd = cli_cmd
    env = dict(os.environ, GGML_PHASE6_PROFILE="1",
               GGML_PHASE6_ROUTE_TRACE=str(trace),
               GGML_MOE_K1=str(K1), GGML_MOE_K2=str(K2))
    if nodelist:
        env["GGML_PHASE6_NODELIST"] = str(nodelist)
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
    perf = parse_perf(out + "\n" + err)
    cache_m = CACHE_RE.findall(err)
    prof_m = PROF_RE.findall(err)
    if not prof_m:
        raise RuntimeError(f"{prefix}: no PHASE6_PROFILE line (timers dead?)")
    prof = parse_kv_list(prof_m[-1])
    cache = parse_kv_list(cache_m[-1]) if cache_m else {}
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
    if prof.get("dec_graphs", 0) <= 0:
        raise RuntimeError(f"{prefix}: no decode graphs counted")
    if prof.get("markers", 0) <= 0:
        raise RuntimeError(f"{prefix}: no section markers seen")
    payload = response_payload(out, prompt)
    time_txt = time_path.read_text(encoding="utf-8") if time_path.exists() else ""
    m_rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", time_txt)
    m_min = re.search(r"Minor \(reclaiming a frame\) page faults: (\d+)", time_txt)
    m_maj = re.search(r"Major \(requiring I/O\) page faults: (\d+)", time_txt)
    return {"arm": name, "pid": pid, "rep": rep, "category": category,
            "cold": cold, "elapsed_sec": elapsed, "cache_drop": cache_drop,
            "peak_rss_mib": max((x["rss_kib"] for x in valid), default=0) / 1024,
            "peak_rss_anon_mib": max((x["rss_anon_kib"] for x in valid), default=0) / 1024,
            "peak_rss_file_mib": max((x["rss_file_kib"] for x in valid), default=0) / 1024,
            "read_bytes_max": max((x["read_bytes"] for x in valid), default=0),
            "rchar_max": max((x["rchar"] for x in valid), default=0),
            "minflt_max": max((x["minflt"] for x in valid), default=0),
            "majflt_max": max((x["majflt"] for x in valid), default=0),
            "time_maxrss_kib": int(m_rss.group(1)) if m_rss else None,
            "time_minflt": int(m_min.group(1)) if m_min else None,
            "time_majflt": int(m_maj.group(1)) if m_maj else None,
            "decode_perf": perf, "cache": cache, "prof": prof,
            "wb_unmatched": wb_unmatched,
            "response_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "trace_sha256": sha256(trace) if trace.exists() else None}


def cond_summary(runs):
    """Timing/profile means over a run subset (cold or warm)."""
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
    return {"runs": len(runs), "dec_toks": dec_toks,
            "tps_total": tps_total, "ms_per_tok": 1000 / tps_total,
            "tps_mean_runs": st.mean(tps_runs),
            "tps_stdev_runs": st.stdev(tps_runs) if len(tps_runs) > 1 else 0.0,
            "sections_ms_tok": sec, "wb_ms_tok": wb,
            "ttft_ms_mean": st.mean(r["prof"]["ttft_ns"] / 1e6 for r in runs),
            "prefill_c_ms_mean": st.mean(r["decode_perf"]["prefill_c_ms"] for r in runs)}


def arm_summary(name, runs):
    warm = [r for r in runs if not r["cold"]]
    cold = [r for r in runs if r["cold"]]
    out = {"warm": cond_summary(warm) if warm else None,
           "cold": cond_summary(cold) if cold else None,
           # Headline = warm steady state (flattened for convenience)
           "runs": len(runs),
           "dec_toks": sum(r["prof"]["dec_graphs"] for r in runs),
           "peak_rss_mib_max": max(r["peak_rss_mib"] for r in runs),
           "peak_rss_anon_mib_max": max(r["peak_rss_anon_mib"] for r in runs),
           "peak_rss_file_mib_max": max(r["peak_rss_file_mib"] for r in runs),
           "time_maxrss_kib_max": max((r["time_maxrss_kib"] or 0 for r in runs), default=None),
           "read_bytes_total": sum(r["read_bytes_max"] for r in runs),
           "minflt_max": max(r["minflt_max"] for r in runs),
           "majflt_max": max(r["majflt_max"] for r in runs)}
    w = out["warm"] or out["cold"]
    out.update({"tps_total": w["tps_total"], "ms_per_tok": w["ms_per_tok"],
                "sections_ms_tok": w["sections_ms_tok"],
                "wb_ms_tok": w["wb_ms_tok"],
                "ttft_ms_mean": w["ttft_ms_mean"],
                "prefill_c_ms_mean": w["prefill_c_ms_mean"]})
    if runs[0]["cache"]:
        # Hit rate is file-temp independent (executor cache always starts
        # empty); aggregate over ALL runs for max N.
        req = sum(r["cache"]["dec_requests"] for r in runs)
        hit = sum(r["cache"]["dec_hits"] for r in runs)
        mis = sum(r["cache"]["dec_misses"] for r in runs)
        dec_toks = out["dec_toks"]
        out["cache"] = {
            "hit_rate": hit / req, "miss_tok": mis / dec_toks,
            "bytes_tok": sum(r["cache"]["dec_read_bytes"] for r in runs) / dec_toks,
            "read_ns_tok": sum(r["cache"]["read_ns"] for r in runs) / dec_toks / 1e6,
            "async_wait_ms_tok": sum(r["cache"]["async_wait_ns"] for r in runs) / dec_toks / 1e6,
            "ready_wait_ms_tok": sum(r["cache"]["ready_wait_ns"] for r in runs) / dec_toks / 1e6,
            "evictions": sum(r["cache"]["evictions"] for r in runs),
            "preload_bytes": runs[0]["cache"]["preload_bytes"],
            "slot_bytes": runs[0]["cache"]["slot_bytes"]}
    return out


def main():
    started = time.time()
    runtime = setup_runtime()
    wfree = shutil.disk_usage(WORK).free / 1e9
    sfree = shutil.disk_usage(SCRATCH).free / 1e9
    print(f"disk free: WORK {wfree:.1f}GB SCRATCH {sfree:.1f}GB", flush=True)
    if wfree < 15 or sfree < 15:
        raise RuntimeError(f"disk too tight: WORK {wfree} SCRATCH {sfree}")
    build()
    model = fetch_model()
    hardware = {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                "threads_used": THREADS,
                "cpuinfo": Path("/proc/cpuinfo").read_text()[:6000],
                "meminfo": Path("/proc/meminfo").read_text()}
    (OUT / "hardware.json").write_text(json.dumps(hardware, indent=2))
    model_iq2 = SCRATCH / MODELS[BASE]["file"]
    q2k_path = WORK / Q2K_NAME
    transcode = transcode_q2k(model_iq2, q2k_path)
    model["q2k_file"] = Q2K_NAME
    model["transcode"] = transcode
    # Locked pins to files (counts asserted here + in C at runtime)
    pins_map = {}
    for tag, blob, n in (("3.0", PINS_3, 80), ("4.0", PINS_4, 80),
                         ("5.0", PINS_5, 1346), ("6.0", PINS_6, 915)):
        p = OUT / f"pins_{tag}.txt"
        p.write_text(blob if blob.endswith("\n") else blob + "\n")
        keys = [x for x in p.read_text().split() if x.strip()]
        assert len(keys) == n, (tag, len(keys), n)
        assert all(0 <= int(k) < 10240 for k in keys), tag
        pins_map[tag] = p
    # SMOKE: resident + locked arm on one prompt; gates the full loop
    print("===== SMOKE (resident vs bounded_6gb, pid 21) =====", flush=True)
    res = next(a for a in ARMS if a["name"] == "resident")
    b6 = next(a for a in ARMS if a["name"] == "bounded_6gb")
    s0 = run_case(res, 21, 0, q2k_path, None, tag="_smoke")
    s1 = run_case(b6, 21, 0, q2k_path, pins_map["6.0"], tag="_smoke")
    print(f"  smoke resident: {s0['decode_perf']['tokens_per_second']:.2f} t/s "
          f"rss={s0['peak_rss_mib']:.0f}MiB", flush=True)
    print(f"  smoke b6: {s1['decode_perf']['tokens_per_second']:.2f} t/s "
          f"rss={s1['peak_rss_mib']:.0f}MiB hit={s1['cache']['dec_hits']/s1['cache']['dec_requests']:.4f} "
          f"markers={s1['prof']['markers']}", flush=True)
    if s0["response_sha256"] != s1["response_sha256"]:
        raise RuntimeError("SMOKE FAIL: bounded output != resident output")
    if s0["trace_sha256"] != s1["trace_sha256"]:
        raise RuntimeError("SMOKE FAIL: bounded routes != resident routes")
    print("  smoke: bit-exact outputs + routes OK", flush=True)
    # NODELIST ground truth (one short resident run; first graph only)
    print("===== NODELIST (resident, pid 21, n=1) =====", flush=True)
    nl = run_case(res, 21, 0, q2k_path, None, tag="_nodelist", cold=False,
                  nodelist=OUT / "nodelist.txt", n_gen=1)
    print(f"  nodelist nodes: {len((OUT / 'nodelist.txt').read_text().splitlines())}",
          flush=True)
    # FULL LOOP (first run per arm cold, rest warm; the file stays
    # warm within an arm after the first drop)
    runs = []
    for arm in ARMS:
        for pid in PIDS:
            for rep in range(1, arm["reps"] + 1):
                cold = (pid == PIDS[0] and rep == 1)
                print(f"===== {arm['name']} pid={pid:02d} rep={rep} "
                      f"{'cold' if cold else 'warm'} =====", flush=True)
                r = run_case(arm, pid, rep, q2k_path,
                             pins_map.get(arm.get("pins", "")), cold=cold)
                runs.append(r)
                c = r["cache"]
                extra = (f" hit={c['dec_hits']/c['dec_requests']:.3f}"
                         if c else "")
                print(f"  {r['decode_perf']['tokens_per_second']:.2f} t/s "
                      f"rss={r['peak_rss_mib']:.0f}MiB{extra}", flush=True)
    # Bit-exactness gate: every pid has ONE output across arms+reps
    sha_gate = {}
    for pid in PIDS:
        shas = set(r["response_sha256"] for r in runs if r["pid"] == pid)
        sha_gate[pid] = len(shas)
        if len(shas) != 1:
            raise RuntimeError(f"pid {pid}: {len(shas)} distinct outputs!")
    # Weight-bucket coverage gate: unmatched matmul weights must be ~none
    unmatched = sorted(set(u for r in runs for u in r["wb_unmatched"]))
    print(f"  wb unmatched weights: {unmatched if unmatched else 'NONE'}",
          flush=True)
    summaries = {}
    for arm in ARMS:
        summaries[arm["name"]] = arm_summary(
            arm["name"], [r for r in runs if r["arm"] == arm["name"]])
    # Sim validation: measured vs predicted hit/miss per bounded arm
    sim_check = {}
    for name, exp in EXPECTED.items():
        if name not in summaries:
            continue
        m = summaries[name]["cache"]
        sim_check[name] = {
            "pred_hit": exp["hit"], "meas_hit": m["hit_rate"],
            "pred_miss_tok": exp["miss_tok"], "meas_miss_tok": m["miss_tok"]}
    q2k_path.unlink()  # keep the pull small (telemetry + logs only)
    result = {"schema": "native-sparse-edge0join4b/v1", "status": "ok",
              "wb_unmatched": unmatched,
              "runtime": runtime, "model": model,
              "hardware": {k: v for k, v in hardware.items()
                           if k not in ("cpuinfo", "meminfo")},
              "decode": {"k1": K1, "k2": K2, "threads": THREADS,
                         "tokens_requested_per_prompt": N_GEN,
                         "pids": PIDS},
              "arms": [a["name"] for a in ARMS],
              "summaries": summaries, "sim_check": sim_check,
              "sha_gate": sha_gate, "wall_sec": time.time() - started}
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print("== FINAL means ==", flush=True)
    for name, s in summaries.items():
        c = s.get("cache", {})
        print(f"{name:12s} {s['tps_total']:.2f} t/s ({s['ms_per_tok']:.1f} ms/tok) "
              f"rss={s['peak_rss_mib_max']:.0f}MiB"
              + (f" hit={c['hit_rate']:.4f} miss/t={c['miss_tok']:.1f} "
                 f"B/t={c['bytes_tok']:.0f}" if c else ""), flush=True)
    print(json.dumps({"status": "ok", "summaries": summaries,
                      "sim_check": sim_check}), flush=True)


if __name__ == "__main__":
    main()
