"""Phase-10b remainder-bypass kernel: wall-truth for the 138 ms remainder.

One build, measurement-only bypass arms (env-selected node families are
memset-zeroed before their CPU implementation; outputs invalid by design).
Families from the phase5h tensor inventory (30 GDN-linear + 10 full-attn +
40 MoE+shared layers; routed _exps NEVER bypassed):
  attn_proj x2      sanity: must reproduce 6b ~48 ms
  lm_head x2        sanity: must reproduce 5a-v2 ~25 ms
  gdn_ssm x3        THESIS: GDN recurrence/state wall truth (~11 ms op-est)
  shared_expert x3  THESIS: shared-expert wall truth (~8 ms op-est)
  attn_scores x3    GAP: flash-attn QK^T/softmax/AV (~10 ms unattributed)
  all_dense x2      BOUND: experts+overhead residual (ggml-tax bound)
  control x3        resident exact (hash-checked) + TSC profile
Trace logs every classified node (matcher validation); TSC profile
corroborates wall deltas. Resolves the ~27 ms gap + GDN/shared wall truth
in a single ~20 min kernel. Bypass changes routes/outputs: no quality reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path


SCRATCH = Path("/tmp/native-sparse-remainder-bypass-v1")
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
BENCH = BUILD / "bin" / "llama-bench"
CLI = BUILD / "bin" / "llama-cli"
MODEL = SCRATCH / "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
OUT = Path("/kaggle/working/native-sparse-remainder-bypass-v1-results")

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_SHA256 = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
RESEARCH_BASE_COMMIT = "9ded82cadc5327e6d1a7c6298c2f58f66bb120a6"
PROMPT = "Give one concise reason oral rehydration solution helps a child with watery diarrhoea."
THREADS = 4
N_GEN = 64
N_REPEATS = 3
EXPECTED_CONTROL_SHA256 = "a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c"

SUMMARY_PERF_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+) t/s\s*\|\s*Generation:\s*([0-9.]+) t/s\s*\]"
)
ATTENTION_PROFILE_RE = re.compile(
    r"^REM_BYPASS_PROFILE family=(\S+) cycles=(\d+) calls=(\d+)$"
)


def command_text(cmd: list[str]) -> str:
    return " ".join(str(x) for x in cmd)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(
    cmd: list[str], *, cwd: Path | None = None, log: Path | None = None
) -> subprocess.CompletedProcess:
    print("+", command_text(cmd), flush=True)
    process = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if log is not None:
        log.write_text(process.stdout + "\n--- STDERR ---\n" + process.stderr, encoding="utf-8")
    if process.returncode:
        print(process.stdout[-2000:], flush=True)
        print(process.stderr[-2000:], flush=True)
        raise RuntimeError(f"command exited {process.returncode}: {command_text(cmd)}")
    return process


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_runtime() -> dict:
    """Apply only the existing exact loader change plus Amdahl hooks."""

    cpu = LLAMA / "ggml/src/ggml-cpu/ggml-cpu.c"
    qwen = LLAMA / "src/models/qwen35moe.cpp"

    replace_once(
        cpu,
        "#include <stdint.h>\n",
        "#include <stdint.h>\n"
        "#if defined(__x86_64__) || defined(__i386__)\n"
        "#include <x86intrin.h>\n"
        "#endif\n",
    )

    attention_hook = r'''
enum rem_bypass_family {
    REM_FAM_ATTN_PROJ = 0,  // all attn_* weight MUL_MAT (qkv/gate/q/k/v/output)
    REM_FAM_GDN = 1,        // ssm_* recurrence/state projections
    REM_FAM_SHARED = 2,     // ffn_*_shexp shared-expert MUL_MAT
    REM_FAM_SCORES = 3,     // GGML_OP_FLASH_ATTN_EXT (QK^T/softmax/AV fused)
    REM_FAM_LMHEAD = 4,     // output.weight MUL_MAT
    REM_FAM_DENSE = 5,      // every other dense MUL_MAT (non-_exps)
    REM_FAM_SOFTMAX = 6,    // GGML_OP_SOFT_MAX (profile only)
    REM_FAM_COUNT = 7,
};

static const char * rem_bypass_labels[REM_FAM_COUNT] = {
    "attn_proj", "gdn_ssm", "shared_expert", "attn_scores", "lm_head",
    "dense_other", "softmax",
};

static uint64_t rem_bypass_cycles[REM_FAM_COUNT] = {0};
static uint64_t rem_bypass_calls[REM_FAM_COUNT] = {0};

static inline int rem_bypass_env(const char * name) {
    const char * value = getenv(name);
    return value != NULL && atoi(value) != 0;
}

static inline uint64_t rem_bypass_tsc(void) {
#if defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#else
    return 0;
#endif
}

static int rem_bypass_projection_family(const struct ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_FLASH_ATTN_EXT) return REM_FAM_SCORES;
    if (tensor->op == GGML_OP_SOFT_MAX) return REM_FAM_SOFTMAX;
    if (tensor->op != GGML_OP_MUL_MAT || tensor->src[0] == NULL) {
        return -1;
    }
    const char * name = tensor->src[0]->name;
    // Routed experts are NEVER bypassed (the remainder is non-expert by
    // definition); activation*activation MUL_MATs carry no weight substring.
    if (strstr(name, "_exps") != NULL) return -1;
    if (strstr(name, "attn_norm") != NULL || strstr(name, "q_norm") != NULL ||
            strstr(name, "k_norm") != NULL || strstr(name, "ssm_norm") != NULL) {
        return -1;
    }
    if (strstr(name, "output.weight") != NULL) return REM_FAM_LMHEAD;
    if (strstr(name, "shexp") != NULL) return REM_FAM_SHARED;
    if (strstr(name, "ssm") != NULL) return REM_FAM_GDN;
    if (strstr(name, "attn_") != NULL) return REM_FAM_ATTN_PROJ;
    // Any other dense weight projection (token_embd-adjacent, etc.).
    if (strstr(name, ".weight") != NULL || strstr(name, ".bias") != NULL) {
        return REM_FAM_DENSE;
    }
    return -1;
}

static int rem_bypass_family(const struct ggml_tensor * tensor) {
    return rem_bypass_projection_family(tensor);
}

// Bypass selector: GGML_REMAINDER_BYPASS_FAMILY=attn_proj|gdn_ssm|shared_expert|
// attn_scores|lm_head|dense_other|all_dense|none. all_dense bypasses every
// classified dense family (experts still run); used for the overhead bound.
static int rem_bypass_selected(int family) {
    const char * sel = getenv("GGML_REMAINDER_BYPASS_FAMILY");
    if (sel == NULL || sel[0] == '\0' || strcmp(sel, "none") == 0) return 0;
    if (strcmp(sel, "all_dense") == 0) {
        return family != REM_FAM_SOFTMAX;
    }
    const char * want = NULL;
    if (strcmp(sel, "attn_proj") == 0) want = "attn_proj";
    else if (strcmp(sel, "gdn_ssm") == 0) want = "gdn_ssm";
    else if (strcmp(sel, "shared_expert") == 0) want = "shared_expert";
    else if (strcmp(sel, "attn_scores") == 0) want = "attn_scores";
    else if (strcmp(sel, "lm_head") == 0) want = "lm_head";
    else if (strcmp(sel, "dense_other") == 0) want = "dense_other";
    if (want == NULL) return 0;
    return strcmp(rem_bypass_labels[family], want) == 0;
}

static void rem_bypass_trace(const struct ggml_tensor * tensor,
                                   const char * event) {
    if (tensor->src[0] == NULL ||
            !rem_bypass_env("GGML_REMAINDER_BYPASS_TRACE") ||
            rem_bypass_projection_family(tensor) < 0) {
        return;
    }
    const char * path = getenv("GGML_REMAINDER_BYPASS_TRACE_PATH");
    if (path == NULL || path[0] == '\0') return;
    static FILE * fp = NULL;
    if (fp == NULL) {
        fp = fopen(path, "a");
        if (fp == NULL) return;
        setvbuf(fp, NULL, _IOLBF, 0);
    }
    const int family = rem_bypass_projection_family(tensor);
    // GGUF tensor names contain only the JSON-safe characters emitted here.
    fprintf(fp, "{\"event\":\"%s\",\"family\":\"%s\",\"name\":\"%s\"}\n",
        event, rem_bypass_labels[family], tensor->src[0]->name);
}

static void rem_bypass_report(void) {
    if (!rem_bypass_env("GGML_REMAINDER_BYPASS_PROFILE")) return;
    for (int i = 0; i < REM_FAM_COUNT; ++i) {
        fprintf(stderr,
            "REM_BYPASS_PROFILE family=%s cycles=%llu calls=%llu\n",
            rem_bypass_labels[i],
            (unsigned long long) __atomic_load_n(&rem_bypass_cycles[i], __ATOMIC_RELAXED),
            (unsigned long long) __atomic_load_n(&rem_bypass_calls[i], __ATOMIC_RELAXED));
    }
}

static inline void rem_bypass_register_report(void) {
    static int registered = 0;
    if (rem_bypass_env("GGML_REMAINDER_BYPASS_PROFILE") &&
            __sync_bool_compare_and_swap(&registered, 0, 1)) {
        atexit(rem_bypass_report);
    }
}
'''
    replace_once(
        cpu,
        "static struct ggml_state g_state = {0};\n",
        attention_hook + "\nstatic struct ggml_state g_state = {0};\n",
    )

    # Keep the exact Qwen native-sparse loader intervention used by Phase 0/5A.
    # The benchmark uses -lzm off, so this preserves the resident control while
    # retaining the same loader/runtime basis for later cache comparisons.
    replace_once(
        qwen,
        "        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, \"weight\", il), { n_ff_exp, n_embd, n_expert }, flags);\n"
        "        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);",
        "        // Exact native-sparse research loader: only routed expert\n"
        "        // payload tensors are marked lazy; math and routing are unchanged.\n"
        "        const int expert_flags = flags | TENSOR_READ_LAZY;\n"
        "        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, \"weight\", il), { n_ff_exp, n_embd, n_expert }, expert_flags);\n"
        "        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, expert_flags);",
    )

    # This hook is before extra-buffer dispatch so bypassed destinations are
    # ready for all consumers and no attention weight bytes are read.  The
    # normal control is untouched.  The barrier matches the existing skip arms.
    replace_once(
        cpu,
        "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
        "        return;\n"
        "    }\n\n"
        "    // extra_buffer op?\n",
        "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
        "        return;\n"
        "    }\n\n"
        "    const int rem_bypass_projection =\n"
        "        rem_bypass_projection_family(tensor);\n"
        "    const int rem_bypass_bypass =\n"
        "        rem_bypass_projection >= 0 &&\n"
        "        rem_bypass_selected(rem_bypass_projection);\n"
        "    if (rem_bypass_projection >= 0 && params->ith == 0) {\n"
        "        rem_bypass_trace(tensor, rem_bypass_bypass ?\n"
        "            \"bypassed\" : \"executed\");\n"
        "    }\n"
        "    if (rem_bypass_bypass) {\n"
        "        if (params->ith == 0) {\n"
        "            memset(tensor->data, 0, ggml_nbytes(tensor));\n"
        "        }\n"
        "        ggml_barrier(params->threadpool);\n"
        "        return;\n"
        "    }\n\n"
        "    // extra_buffer op?\n",
    )

    # Attribute operator work by the exact families above.  Like the existing
    # Phase 5A counters, these are summed worker TSC intervals, not wall time.
    replace_once(
        cpu,
        "    switch (tensor->op) {\n",
        "    const int rem_bypass_kind = rem_bypass_family(tensor);\n"
        "    const int rem_bypass_profile =\n"
        "        rem_bypass_env(\"GGML_REMAINDER_BYPASS_PROFILE\") &&\n"
        "        rem_bypass_kind >= 0;\n"
        "    const uint64_t rem_bypass_start =\n"
        "        rem_bypass_profile ? rem_bypass_tsc() : 0;\n"
        "    if (rem_bypass_profile) rem_bypass_register_report();\n\n"
        "    switch (tensor->op) {\n",
    )
    replace_once(
        cpu,
        "    }\n}\n\n// Android's libc implementation",
        "    }\n\n"
        "    if (rem_bypass_profile) {\n"
        "        __atomic_fetch_add(&rem_bypass_cycles[rem_bypass_kind],\n"
        "            rem_bypass_tsc() - rem_bypass_start, __ATOMIC_RELAXED);\n"
        "        __atomic_fetch_add(&rem_bypass_calls[rem_bypass_kind],\n"
        "            1, __ATOMIC_RELAXED);\n"
        "    }\n"
        "}\n\n// Android's libc implementation",
    )

    patch = run_checked(
        ["git", "diff", "--", "ggml/src/ggml-cpu/ggml-cpu.c", "src/models/qwen35moe.cpp"],
        cwd=LLAMA,
    ).stdout
    (OUT / "remainder-bypass-runtime.patch").write_text(patch, encoding="utf-8")
    return {"description": "exact resident control plus remainder-bypass family hooks",
            "sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "files": ["ggml/src/ggml-cpu/ggml-cpu.c", "src/models/qwen35moe.cpp"]}


def clone_patch_build() -> dict:
    if not LLAMA.exists():
        run_checked(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    source_head = run_checked(["git", "rev-parse", "HEAD"], cwd=LLAMA).stdout.strip()
    patch = patch_runtime()
    configure = [
        "cmake", "-S", str(LLAMA), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF",
        "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF",
    ]
    run_checked(configure, log=OUT / "cmake-configure.log")
    build = ["cmake", "--build", str(BUILD), "--config", "Release", "-j4",
             "--target", "llama-bench", "llama-cli"]
    run_checked(build, log=OUT / "cmake-build.log")
    version = run_checked([str(CLI), "--version"])
    (OUT / "llama-version.txt").write_text(version.stdout, encoding="utf-8")
    compiler = run_checked(["c++", "--version"]).stdout.splitlines()[0]
    return {"commit": LLAMA_COMMIT, "source_head": source_head, "patch": patch,
            "configure_command": configure, "build_command": build,
            "compiler": compiler}


def fetch_model() -> dict:
    if not MODEL.exists() or MODEL.stat().st_size != MODEL_SIZE:
        run_checked([
            "curl", "-L", "--fail", "--retry", "8", "--retry-delay", "10",
            "--retry-all-errors", "-C", "-", "-o", str(MODEL), MODEL_URL,
        ], log=OUT / "model-download.log")
    size, digest = MODEL.stat().st_size, sha256(MODEL)
    if size != MODEL_SIZE or digest != MODEL_SHA256:
        raise RuntimeError(f"model identity mismatch: size={size}, sha256={digest}")
    return {"repo": "unsloth/Qwen3.5-35B-A3B-GGUF", "repo_commit": MODEL_REPO_COMMIT,
            "file": MODEL_FILE, "size_bytes": size, "sha256": digest, "url": MODEL_URL}


def read_proc(pid: int) -> dict:
    result = {"mono_ns": time.monotonic_ns(), "rss_kib": 0, "anon_kib": 0,
              "file_kib": 0, "minor_faults": 0, "major_faults": 0, "read_bytes": 0}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"): result["rss_kib"] = int(line.split()[1])
            elif line.startswith("RssAnon:"): result["anon_kib"] = int(line.split()[1])
            elif line.startswith("RssFile:"): result["file_kib"] = int(line.split()[1])
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key == "read_bytes": result[key] = int(value)
        stat_tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        result["minor_faults"], result["major_faults"] = int(stat_tail[7]), int(stat_tail[9])
        result["valid"] = True
    except (OSError, ValueError):
        result["valid"] = False
    return result


def parse_attention_profile(stderr: str) -> dict:
    profile = {}
    for line in stderr.splitlines():
        match = ATTENTION_PROFILE_RE.match(line)
        if match:
            profile[match.group(1)] = {"cycles": int(match.group(2)), "calls": int(match.group(3))}
    return profile


def parse_attention_names(path: Path) -> dict:
    events = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    event_names = sorted({x.get("event") for x in events if x.get("event")})
    return {
        "events": len(events),
        "unique_names": sorted({x.get("name") for x in events if x.get("name")} ),
        "families": sorted({x.get("family") for x in events if x.get("family")} ),
        "event_counts": {event: sum(1 for x in events if x.get("event") == event)
                         for event in event_names},
    }


BYPASS_FAMILIES = ("attn_proj", "gdn_ssm", "shared_expert", "attn_scores",
                  "lm_head", "all_dense")
# 10b expectations (ms/token wall saved): sanity arms must reproduce within
# tolerance; thesis arms resolve open questions.
FAMILY_EXPECTATIONS = {
    "attn_proj": {"expect_ms": 48.0, "tol_ms": 15.0, "kind": "sanity_repro_6b"},
    "lm_head": {"expect_ms": 25.0, "tol_ms": 10.0, "kind": "sanity_repro_5av2"},
    "gdn_ssm": {"expect_ms": 11.0, "tol_ms": None, "kind": "thesis_wall_truth"},
    "shared_expert": {"expect_ms": 8.0, "tol_ms": None, "kind": "thesis_wall_truth"},
    "attn_scores": {"expect_ms": 10.0, "tol_ms": None, "kind": "gap_attribution"},
    "all_dense": {"expect_ms": None, "tol_ms": None, "kind": "overhead_bound"},
}


def amdahl_delta(runs: list[dict]) -> dict:
    by_label = {x["label"]: x for x in runs}
    controls = [x for x in runs if x["label"].startswith("resident_exact_control")]
    control_ms = statistics.mean(x["ms_per_token"] for x in controls if x.get("ms_per_token"))
    out = {"available": True, "control_ms_per_token": control_ms,
           "control_reps": [x.get("ms_per_token") for x in controls],
           "families": {}}
    for fam in BYPASS_FAMILIES:
        rows = [x for x in runs if x.get("bypass_family") == fam and x.get("ms_per_token")]
        if not rows:
            out["families"][fam] = {"available": False}
            continue
        ms = statistics.mean(x["ms_per_token"] for x in rows)
        saved = control_ms - ms
        bypassed = set()
        for x in rows:
            for n in x.get("attention_names", {}).get("unique_names", []):
                bypassed.add(n)
        exp = FAMILY_EXPECTATIONS[fam]
        verdict = "measured"
        if exp["tol_ms"] is not None:
            verdict = ("reproduced" if abs(saved - exp["expect_ms"]) <= exp["tol_ms"]
                       else "MISMATCH_EXPECTATION")
        out["families"][fam] = {
            "available": True, "bypass_ms_per_token": ms,
            "reps_ms_per_token": [x["ms_per_token"] for x in rows],
            "wall_ms_saved_per_token": saved,
            "wall_fraction_of_control": saved / control_ms,
            "free_ceiling_tok_s": 1000.0 / ms,
            "bypassed_unique_names": sorted(bypassed),
            "expectation": exp, "verdict": verdict,
        }
    out["interpretation"] = "measurement-only upper bounds; bypass zeroes activations and may change later routes"
    return out


def run_benchmark(label: str, *, bypass_family: str | None = None, profile: bool = False) -> dict:
    trace = OUT / f"attention_names.{label}.jsonl"
    trace.unlink(missing_ok=True)
    cmd = [str(BENCH), "-m", str(MODEL), "-p", "0", "-n", str(N_GEN),
           "-r", str(N_REPEATS), "-t", str(THREADS), "-ngl", "0", "-lm", "mmap",
           "-lzm", "off", "--poll", "0", "--output", "json"]
    overrides = {
        "GGML_REMAINDER_BYPASS_FAMILY": bypass_family or "none",
        "GGML_REMAINDER_BYPASS_TRACE": "1",
        "GGML_REMAINDER_BYPASS_TRACE_PATH": str(trace),
    }
    if profile:
        overrides["GGML_REMAINDER_BYPASS_PROFILE"] = "1"
    else:
        overrides["GGML_REMAINDER_BYPASS_PROFILE"] = "0"
    env = dict(os.environ); env.update(overrides)
    print("+", command_text(cmd), "ENV", overrides, flush=True)
    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env)
    samples = []
    while process.poll() is None:
        samples.append(read_proc(process.pid))
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    (OUT / f"{label}.stdout.json").write_text(stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(stderr, encoding="utf-8")
    try:
        parsed = json.loads(stdout.strip())
        row = parsed[0] if len(parsed) == 1 else None
    except (json.JSONDecodeError, TypeError, IndexError):
        row = None
    valid = [x for x in samples if x.get("valid")]
    tok_samples = row.get("samples_ts") if row else []
    rss = max((x["rss_kib"] for x in valid), default=0)
    start_io = samples[0]["read_bytes"] if samples else 0
    end_io = samples[-1]["read_bytes"] if samples else 0
    profiles = parse_attention_profile(stderr)
    return {
        "label": label, "bypass_family": bypass_family, "profile": profile,
        "environment": overrides, "command": cmd, "returncode": process.returncode,
        "decode_row": row, "sample_tok_s": tok_samples,
        "mean_tok_s": statistics.mean(tok_samples) if tok_samples else None,
        "median_tok_s": statistics.median(tok_samples) if tok_samples else None,
        "min_tok_s": min(tok_samples) if tok_samples else None,
        "max_tok_s": max(tok_samples) if tok_samples else None,
        "ms_per_token": 1000.0 / row["avg_ts"] if row and row.get("avg_ts") else None,
        "sample_ms_per_token": [1000.0 / x for x in tok_samples if x],
        "peak_rss_mib": rss / 1024,
        "peak_anon_mib": max((x["anon_kib"] for x in valid), default=0) / 1024,
        "peak_file_mib": max((x["file_kib"] for x in valid), default=0) / 1024,
        "minor_faults": max((x["minor_faults"] for x in valid), default=0),
        "major_faults": max((x["major_faults"] for x in valid), default=0),
        "read_bytes_delta": max(0, end_io - start_io),
        "attention_profile": profiles,
        "attention_names": parse_attention_names(trace),
        "proc_samples": samples,
        "stderr_tail": stderr[-3000:],
    }


def generated_response(stdout: str) -> str | None:
    marker = stdout.find("[Start thinking]")
    if marker < 0: return None
    tail = stdout[marker:]
    summary = SUMMARY_PERF_RE.search(tail)
    if summary: tail = tail[:summary.start()]
    if tail.rstrip().endswith("Exiting..."):
        tail = tail.rstrip()[:-len("Exiting...")]
    response = tail.strip()
    return response or None


def run_correctness(label: str, *, bypass_family: str | None = None) -> dict:
    cmd = [str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(THREADS),
           "-c", "512", "-n", "24", "--temp", "0", "--seed", "1234",
           "--single-turn", "--no-display-prompt", "--no-warmup", "--perf",
           "-lm", "mmap", "-lzm", "off", "--poll", "0", "-p", PROMPT]
    trace = OUT / f"attention_names.{label}.jsonl"
    trace.unlink(missing_ok=True)
    overrides = {"GGML_REMAINDER_BYPASS_FAMILY": bypass_family or "none",
                 "GGML_REMAINDER_BYPASS_TRACE": "1",
                 "GGML_REMAINDER_BYPASS_TRACE_PATH": str(trace),
                 "GGML_REMAINDER_BYPASS_PROFILE": "0"}
    env = dict(os.environ); env.update(overrides)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env=env, check=False)
    (OUT / f"{label}.stdout.txt").write_text(p.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(p.stderr, encoding="utf-8")
    response = generated_response(p.stdout)
    return {"label": label, "bypass_family": bypass_family, "environment": overrides, "command": cmd,
            "returncode": p.returncode,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest() if response else None,
            "attention_names": parse_attention_names(trace),
            "output_valid": bypass_family is None and p.returncode == 0}


def hardware_snapshot() -> dict:
    def read(path: str) -> str | None:
        try: return Path(path).read_text(encoding="utf-8")
        except OSError: return None
    try: cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError): cpus = list(range(os.cpu_count() or 0))
    return {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "allowed_cpu_ids": cpus,
            "cpuinfo": read("/proc/cpuinfo"),
            "lscpu": subprocess.run(["lscpu"], text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, check=False).stdout}


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "native-sparse-remainder-bypass/v1", "status": "failed",
        "hypothesis": "Phase-10b remainder decomposition: wall-bypass arms measure GDN (~11ms?) + shared-expert (~8ms?) + attn-scores (~10ms?) + all-dense overhead bound; attn-proj (~48ms) + lm-head (~25ms) sanity-reproduce 6b/5a-v2.",
        "started_unix": time.time(),
        "research_source": {"base_commit": RESEARCH_BASE_COMMIT,
                            "script_sha256": sha256(Path(__file__))},
        "benchmark": {"n_prompt": 0, "n_gen": N_GEN, "repeats": N_REPEATS,
                      "threads": THREADS, "poll": 0, "load_mode": "mmap",
                      "lazy_mode": "off", "gpu_layers": 0, "affinity": "inherited"},
        "family_selection": {
            "attn_proj": "MUL_MAT src0 contains attn_ (not *_norm)",
            "gdn_ssm": "MUL_MAT src0 contains ssm (30 linear layers)",
            "shared_expert": "MUL_MAT src0 contains shexp",
            "attn_scores": "op FLASH_ATTN_EXT (10 full layers)",
            "lm_head": "MUL_MAT src0 == output.weight",
            "all_dense": "every classified dense family at once (experts run)",
            "never_bypassed": ["*_exps routed experts", "*_norm", "SOFT_MAX (profile only)"],
        },
        "model_source_names": {
            "inventory": "research/native_sparse_experiments/results/phase5h_floor_decomp_v1/tensor_inventory.jsonl",
            "linear_30": ["attn_qkv", "attn_gate", "ssm_a/alpha/beta/conv1d/dt/out/norm"],
            "full_attention_10": ["attn_q/k/v/output", "attn_q_norm/k_norm"],
            "shared_expert_40": ["ffn_gate/up/down_shexp", "ffn_gate_inp_shexp"],
            "routed_experts_40": ["ffn_gate/up/down_exps (NEVER bypassed)"],
        },
        "hardware": hardware_snapshot(), "runs": [], "correctness": {},
        "limitations": [
            "The bypass zeroes activations and can change later native routes; it is a wall-clock upper bound, not same-route attribution.",
            "REM_BYPASS_PROFILE is summed worker TSC work, while llama-bench throughput is process wall clock.",
            "The bypass arm is invalid-output by design and must not be used for quality claims.",
        ],
    }
    try:
        result["build"] = clone_patch_build()
        result["model"] = fetch_model()
        result["runs"] = [
            run_benchmark("resident_exact_control_r1"),
            run_benchmark("resident_exact_control_r2"),
            run_benchmark("resident_exact_control_r3", profile=True),
            run_benchmark("bypass_attn_proj_r1", bypass_family="attn_proj"),
            run_benchmark("bypass_attn_proj_r2", bypass_family="attn_proj"),
            run_benchmark("bypass_lm_head_r1", bypass_family="lm_head"),
            run_benchmark("bypass_lm_head_r2", bypass_family="lm_head"),
            run_benchmark("bypass_gdn_r1", bypass_family="gdn_ssm"),
            run_benchmark("bypass_gdn_r2", bypass_family="gdn_ssm"),
            run_benchmark("bypass_gdn_r3", bypass_family="gdn_ssm"),
            run_benchmark("bypass_shared_r1", bypass_family="shared_expert"),
            run_benchmark("bypass_shared_r2", bypass_family="shared_expert"),
            run_benchmark("bypass_shared_r3", bypass_family="shared_expert"),
            run_benchmark("bypass_scores_r1", bypass_family="attn_scores"),
            run_benchmark("bypass_scores_r2", bypass_family="attn_scores"),
            run_benchmark("bypass_scores_r3", bypass_family="attn_scores"),
            run_benchmark("bypass_alldense_r1", bypass_family="all_dense"),
            run_benchmark("bypass_alldense_r2", bypass_family="all_dense"),
        ]
        result["amdahl"] = amdahl_delta(result["runs"])
        exact = run_correctness("correctness_exact_control")
        result["correctness"] = {
            "exact_control": exact,
            "exact_control_matches_established_hash":
                exact["returncode"] == 0 and exact["response_sha256"] == EXPECTED_CONTROL_SHA256,
            "native_k": 8, "layers": 40, "router_or_weights_changed": False,
            "no_drop_or_substitution": True,
        }
        result["status"] = "ok" if (
            all(x["returncode"] == 0 and x["mean_tok_s"] for x in result["runs"])
            and result["correctness"]["exact_control_matches_established_hash"]
        ) else "invalid_result"
    except Exception as exc:
        result["error"] = repr(exc)
        print(f"ERROR: {exc}", flush=True)
    result["finished_unix"] = time.time()
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    amdahl = result.get("amdahl", {}).get("families", {})
    print(json.dumps({"status": result["status"],
                      "runs": [[x["label"], x.get("mean_tok_s")] for x in result.get("runs", [])],
                      "exact_control": result.get("correctness", {}).get("exact_control_matches_established_hash"),
                      "amdahl_ms_saved": {k: (round(v["wall_ms_saved_per_token"], 2), v["verdict"])
                                          for k, v in amdahl.items() if v.get("available")},
                      "error": result.get("error")}), flush=True)
    if result["status"] != "ok": raise SystemExit(2)


if __name__ == "__main__":
    main()
