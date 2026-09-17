"""Native-sparse attention wall-clock Amdahl experiment.

The exact resident control is compared with a measurement-only arm that
zeroes attention-projection MUL_MAT outputs before their CPU implementation.
The bypass is deliberately not a correctness path: it measures the wall-clock
ceiling available if those projection operators were free.
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


SCRATCH = Path("/tmp/native-sparse-attention-amdahl-v1")
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
BENCH = BUILD / "bin" / "llama-bench"
CLI = BUILD / "bin" / "llama-cli"
MODEL = SCRATCH / "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
OUT = Path("/kaggle/working/native-sparse-attention-amdahl-v1-results")

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_SHA256 = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
RESEARCH_BASE_COMMIT = "ba0047b"
PROMPT = "Give one concise reason oral rehydration solution helps a child with watery diarrhoea."
THREADS = 4
N_GEN = 64
N_REPEATS = 3
EXPECTED_CONTROL_SHA256 = "a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c"

SUMMARY_PERF_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+) t/s\s*\|\s*Generation:\s*([0-9.]+) t/s\s*\]"
)
ATTENTION_PROFILE_RE = re.compile(
    r"^PHASE3_ATTENTION_PROFILE family=(\S+) cycles=(\d+) calls=(\d+)$"
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
enum phase3_attention_family {
    PHASE3_ATTN_QKV = 0,
    PHASE3_ATTN_GATE = 1,
    PHASE3_ATTN_Q = 2,
    PHASE3_ATTN_K = 3,
    PHASE3_ATTN_V = 4,
    PHASE3_ATTN_OUTPUT = 5,
    PHASE3_ATTN_OTHER = 6,
    PHASE3_ATTN_FLASH = 7,
    PHASE3_ATTN_COUNT = 8,
};

static const char * phase3_attention_labels[PHASE3_ATTN_COUNT] = {
    "attn_qkv", "attn_gate", "attn_q", "attn_k", "attn_v",
    "attn_output", "attn_other", "flash_attn_ext",
};

static uint64_t phase3_attention_cycles[PHASE3_ATTN_COUNT] = {0};
static uint64_t phase3_attention_calls[PHASE3_ATTN_COUNT] = {0};

static inline int phase3_attention_env(const char * name) {
    const char * value = getenv(name);
    return value != NULL && atoi(value) != 0;
}

static inline uint64_t phase3_attention_tsc(void) {
#if defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#else
    return 0;
#endif
}

static int phase3_attention_projection_family(const struct ggml_tensor * tensor) {
    if (tensor->op != GGML_OP_MUL_MAT || tensor->src[0] == NULL) {
        return -1;
    }
    const char * name = tensor->src[0]->name;
    if (strstr(name, "attn_") == NULL) return -1;
    if (strstr(name, "attn_norm") != NULL || strstr(name, "q_norm") != NULL ||
            strstr(name, "k_norm") != NULL) return -1;
    if (strstr(name, "attn_qkv") != NULL) return PHASE3_ATTN_QKV;
    if (strstr(name, "attn_gate") != NULL) return PHASE3_ATTN_GATE;
    if (strstr(name, "attn_output") != NULL) return PHASE3_ATTN_OUTPUT;
    if (strstr(name, "attn_q") != NULL) return PHASE3_ATTN_Q;
    if (strstr(name, "attn_k") != NULL) return PHASE3_ATTN_K;
    if (strstr(name, "attn_v") != NULL) return PHASE3_ATTN_V;
    return PHASE3_ATTN_OTHER;
}

static int phase3_attention_family(const struct ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_FLASH_ATTN_EXT) return PHASE3_ATTN_FLASH;
    return phase3_attention_projection_family(tensor);
}

static void phase3_attention_trace(const struct ggml_tensor * tensor,
                                   const char * event) {
    if (tensor->src[0] == NULL ||
            !phase3_attention_env("GGML_ATTENTION_AMDAHL_TRACE") ||
            phase3_attention_projection_family(tensor) < 0) {
        return;
    }
    const char * path = getenv("GGML_ATTENTION_AMDAHL_TRACE_PATH");
    if (path == NULL || path[0] == '\0') return;
    static FILE * fp = NULL;
    if (fp == NULL) {
        fp = fopen(path, "a");
        if (fp == NULL) return;
        setvbuf(fp, NULL, _IOLBF, 0);
    }
    const int family = phase3_attention_projection_family(tensor);
    // GGUF tensor names contain only the JSON-safe characters emitted here.
    fprintf(fp, "{\"event\":\"%s\",\"family\":\"%s\",\"name\":\"%s\"}\n",
        event, phase3_attention_labels[family], tensor->src[0]->name);
}

static void phase3_attention_report(void) {
    if (!phase3_attention_env("GGML_ATTENTION_AMDAHL_PROFILE")) return;
    for (int i = 0; i < PHASE3_ATTN_COUNT; ++i) {
        fprintf(stderr,
            "PHASE3_ATTENTION_PROFILE family=%s cycles=%llu calls=%llu\n",
            phase3_attention_labels[i],
            (unsigned long long) __atomic_load_n(&phase3_attention_cycles[i], __ATOMIC_RELAXED),
            (unsigned long long) __atomic_load_n(&phase3_attention_calls[i], __ATOMIC_RELAXED));
    }
}

static inline void phase3_attention_register_report(void) {
    static int registered = 0;
    if (phase3_attention_env("GGML_ATTENTION_AMDAHL_PROFILE") &&
            __sync_bool_compare_and_swap(&registered, 0, 1)) {
        atexit(phase3_attention_report);
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
        "    const int phase3_attention_projection =\n"
        "        phase3_attention_projection_family(tensor);\n"
        "    const int phase3_attention_bypass =\n"
        "        phase3_attention_env(\"GGML_ATTENTION_AMDAHL_BYPASS\") &&\n"
        "        phase3_attention_projection >= 0;\n"
        "    if (phase3_attention_projection >= 0 && params->ith == 0) {\n"
        "        phase3_attention_trace(tensor, phase3_attention_bypass ?\n"
        "            \"bypassed\" : \"executed\");\n"
        "    }\n"
        "    if (phase3_attention_bypass) {\n"
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
        "    const int phase3_attention_kind = phase3_attention_family(tensor);\n"
        "    const int phase3_attention_profile =\n"
        "        phase3_attention_env(\"GGML_ATTENTION_AMDAHL_PROFILE\") &&\n"
        "        phase3_attention_kind >= 0;\n"
        "    const uint64_t phase3_attention_start =\n"
        "        phase3_attention_profile ? phase3_attention_tsc() : 0;\n"
        "    if (phase3_attention_profile) phase3_attention_register_report();\n\n"
        "    switch (tensor->op) {\n",
    )
    replace_once(
        cpu,
        "    }\n}\n\n// Android's libc implementation",
        "    }\n\n"
        "    if (phase3_attention_profile) {\n"
        "        __atomic_fetch_add(&phase3_attention_cycles[phase3_attention_kind],\n"
        "            phase3_attention_tsc() - phase3_attention_start, __ATOMIC_RELAXED);\n"
        "        __atomic_fetch_add(&phase3_attention_calls[phase3_attention_kind],\n"
        "            1, __ATOMIC_RELAXED);\n"
        "    }\n"
        "}\n\n// Android's libc implementation",
    )

    patch = run_checked(
        ["git", "diff", "--", "ggml/src/ggml-cpu/ggml-cpu.c", "src/models/qwen35moe.cpp"],
        cwd=LLAMA,
    ).stdout
    (OUT / "attention-amdahl-runtime.patch").write_text(patch, encoding="utf-8")
    return {"description": "exact resident control plus opt-in attention MUL_MAT Amdahl hooks",
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
            "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
            "-C", "-", "-o", str(MODEL), MODEL_URL,
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


def amdahl_delta(runs: list[dict]) -> dict:
    by_label = {x["label"]: x for x in runs}
    control = by_label.get("resident_exact_control", {})
    bypass = by_label.get("attention_projection_bypassed", {})
    control_ms = control.get("ms_per_token")
    bypass_ms = bypass.get("ms_per_token")
    if not control_ms or not bypass_ms:
        return {"available": False}
    saved = control_ms - bypass_ms
    return {
        "available": True,
        "control_ms_per_token": control_ms,
        "bypass_ms_per_token": bypass_ms,
        "wall_ms_saved_per_token": saved,
        "wall_fraction_of_control": saved / control_ms,
        "attention_free_ceiling_tok_s": 1000.0 / bypass_ms,
        "interpretation": "measurement-only upper bound; bypass may change later routes",
    }


def run_benchmark(label: str, *, bypass: bool = False, profile: bool = False) -> dict:
    trace = OUT / f"attention_names.{label}.jsonl"
    trace.unlink(missing_ok=True)
    cmd = [str(BENCH), "-m", str(MODEL), "-p", "0", "-n", str(N_GEN),
           "-r", str(N_REPEATS), "-t", str(THREADS), "-ngl", "0", "-lm", "mmap",
           "-lzm", "off", "--poll", "0", "--output", "json"]
    overrides = {
        "GGML_ATTENTION_AMDAHL_BYPASS": "1" if bypass else "0",
        "GGML_ATTENTION_AMDAHL_TRACE": "1",
        "GGML_ATTENTION_AMDAHL_TRACE_PATH": str(trace),
    }
    if profile:
        overrides["GGML_ATTENTION_AMDAHL_PROFILE"] = "1"
    else:
        overrides["GGML_ATTENTION_AMDAHL_PROFILE"] = "0"
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
        "label": label, "bypass": bypass, "profile": profile,
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


def run_correctness(label: str, *, bypass: bool = False) -> dict:
    cmd = [str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(THREADS),
           "-c", "512", "-n", "24", "--temp", "0", "--seed", "1234",
           "--single-turn", "--no-display-prompt", "--no-warmup", "--perf",
           "-lm", "mmap", "-lzm", "off", "--poll", "0", "-p", PROMPT]
    trace = OUT / f"attention_names.{label}.jsonl"
    trace.unlink(missing_ok=True)
    overrides = {"GGML_ATTENTION_AMDAHL_BYPASS": "1" if bypass else "0",
                 "GGML_ATTENTION_AMDAHL_TRACE": "1",
                 "GGML_ATTENTION_AMDAHL_TRACE_PATH": str(trace),
                 "GGML_ATTENTION_AMDAHL_PROFILE": "0"}
    env = dict(os.environ); env.update(overrides)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env=env, check=False)
    (OUT / f"{label}.stdout.txt").write_text(p.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(p.stderr, encoding="utf-8")
    response = generated_response(p.stdout)
    return {"label": label, "bypass": bypass, "environment": overrides, "command": cmd,
            "returncode": p.returncode,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest() if response else None,
            "attention_names": parse_attention_names(trace),
            "output_valid": not bypass and p.returncode == 0}


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
        "schema": "native-sparse-attention-amdahl/v1", "status": "failed",
        "hypothesis": "Attention projection MUL_MAT wall time is a material resident decode bottleneck after routed-MoE and LM-head bounds.",
        "started_unix": time.time(),
        "research_source": {"base_commit": RESEARCH_BASE_COMMIT,
                            "script_sha256": sha256(Path(__file__))},
        "benchmark": {"n_prompt": 0, "n_gen": N_GEN, "repeats": N_REPEATS,
                      "threads": THREADS, "poll": 0, "load_mode": "mmap",
                      "lazy_mode": "off", "gpu_layers": 0, "affinity": "inherited"},
        "attention_selection": {
            "op": "GGML_OP_MUL_MAT",
            "predicate": "GGML_OP_MUL_MAT && src[0]->name contains attn_ && name is not *_norm",
            "families": ["attn_qkv", "attn_gate", "attn_q", "attn_k",
                         "attn_v", "attn_output", "attn_other"],
            "excluded": ["GGML_OP_FLASH_ATTN_EXT", "RMS_NORM", "non-attention MUL_MAT"],
        },
        "model_source_names": {
            "inventory": "research/native_sparse_experiments/results/phase5h_floor_decomp_v1/tensor_inventory.jsonl",
            "linear": ["blk.<layer>.attn_qkv.weight", "blk.<layer>.attn_gate.weight",
                       "blk.<layer>.attn_norm.weight"],
            "full_attention": ["blk.<layer>.attn_q.weight", "blk.<layer>.attn_k.weight",
                                "blk.<layer>.attn_v.weight", "blk.<layer>.attn_output.weight",
                                "blk.<layer>.attn_q_norm.weight", "blk.<layer>.attn_k_norm.weight"],
        },
        "hardware": hardware_snapshot(), "runs": [], "correctness": {},
        "limitations": [
            "The bypass zeroes activations and can change later native routes; it is a wall-clock upper bound, not same-route attribution.",
            "PHASE3_ATTENTION_PROFILE is summed worker TSC work, while llama-bench throughput is process wall clock.",
            "The bypass arm is invalid-output by design and must not be used for quality claims.",
        ],
    }
    try:
        result["build"] = clone_patch_build()
        result["model"] = fetch_model()
        result["runs"] = [
            run_benchmark("resident_exact_control"),
            run_benchmark("resident_attention_profile", profile=True),
            run_benchmark("attention_projection_bypassed", bypass=True),
        ]
        result["amdahl"] = amdahl_delta(result["runs"])
        exact = run_correctness("correctness_exact_control")
        bypass_output = run_correctness("measurement_attention_bypassed", bypass=True)
        result["correctness"] = {
            "exact_control": exact,
            "bypass_measurement_output": bypass_output,
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
    print(json.dumps({"status": result["status"],
                      "runs": [[x["label"], x.get("mean_tok_s")] for x in result.get("runs", [])],
                      "exact_control": result.get("correctness", {}).get("exact_control_matches_established_hash"),
                      "error": result.get("error")}), flush=True)
    if result["status"] != "ok": raise SystemExit(2)


if __name__ == "__main__":
    main()
