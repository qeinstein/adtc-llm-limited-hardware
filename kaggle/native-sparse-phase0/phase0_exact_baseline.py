"""Phase 0: exact native Qwen3.5-35B-A3B CPU baseline on Kaggle.

All arms execute the checkpoint's native top-8 router.  The only intervention
is weight-loading policy.  An environment-gated llama.cpp trace hook observes
the IDs consumed by the exact selected-expert matmul without changing them.
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
from collections import Counter, OrderedDict
from pathlib import Path


WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-phase0")
OUT = WORK / "native-sparse-phase0-results"
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
CLI = BUILD / "bin" / "llama-cli"
BENCH = BUILD / "bin" / "llama-bench"
MODEL = SCRATCH / "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
N_THREADS = max(1, min(4, os.cpu_count() or 1))
PROMPT = "Give one concise reason oral rehydration solution helps a child with watery diarrhoea."
N_GEN = 24

OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
        log: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    p = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(p.stdout[-4000:], flush=True)
    if log:
        log.write_text(p.stdout, encoding="utf-8")
    if check and p.returncode:
        raise RuntimeError(f"command exited {p.returncode}: {cmd}")
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def clone_and_patch() -> dict:
    if not LLAMA.exists():
        run(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", LLAMA])
    run(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)

    qwen = LLAMA / "src" / "models" / "qwen35moe.cpp"
    old = """        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);"""
    new = """        // Phase-0 experiment: routed expert payload is lazy; router/shared/trunk stay resident.
        // Loading policy only.  Native IDs, K, weights, and math are unchanged.
        const int expert_flags = flags | TENSOR_READ_LAZY;
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, expert_flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, expert_flags);"""
    replace_once(qwen, old, new)

    cpu = LLAMA / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
    anchor = "static struct ggml_state g_state = {0};\n"
    helper = r'''

// Observation-only Phase-0 hook. Log IDs and packed tensor sizes for routed
// expert matmuls. This does not alter routing or math.
static void ggml_phase0_trace_moe_ids(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * tensor) {
    if (params->ith != 0 || tensor->op != GGML_OP_MUL_MAT_ID) {
        return;
    }
    const char * path = getenv("GGML_PHASE0_ROUTE_TRACE");
    if (path == NULL || path[0] == '\0') {
        return;
    }
    const struct ggml_tensor * weights = tensor->src[0];
    const struct ggml_tensor * ids = tensor->src[2];
    if (weights == NULL || ids == NULL ||
            strstr(weights->name, "ffn_") == NULL ||
            strstr(weights->name, "_exps") == NULL ||
            ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids)) {
        return;
    }
    static FILE * fp = NULL;
    static uint64_t event_id = 0;
    if (fp == NULL) {
        fp = fopen(path, "a");
        if (fp == NULL) {
            return;
        }
        setvbuf(fp, NULL, _IOLBF, 0);
    }
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const int64_t n = ggml_nelements(ids);
    const int32_t * values = (const int32_t *) ids->data;
    fprintf(fp,
        "{\"event\":%" PRIu64 ",\"mono_ns\":%" PRIu64
        ",\"weight\":\"%s\",\"weight_nbytes\":%zu,\"shape\":[%" PRId64 ",%" PRId64
        ",%" PRId64 ",%" PRId64 "],\"ids\":[",
        event_id++, (uint64_t) ts.tv_sec*1000000000ULL + (uint64_t) ts.tv_nsec,
        weights->name, ggml_nbytes(weights), ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3]);
    for (int64_t i = 0; i < n; ++i) {
        fprintf(fp, "%s%d", i ? "," : "", values[i]);
    }
    fprintf(fp, "]}\n");
}
'''
    replace_once(cpu, anchor, anchor + helper)
    dispatch_anchor = """    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?"""
    dispatch_new = """    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    ggml_phase0_trace_moe_ids(params, tensor);

    // extra_buffer op?"""
    replace_once(cpu, dispatch_anchor, dispatch_new)

    diff = run(["git", "diff", "--", str(qwen.relative_to(LLAMA)), str(cpu.relative_to(LLAMA))],
               cwd=LLAMA).stdout
    (OUT / "llama-phase0.patch").write_text(diff, encoding="utf-8")
    return {"llama_commit": LLAMA_COMMIT, "patch_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def build() -> None:
    run([
        "cmake", "-S", LLAMA, "-B", BUILD,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_NATIVE=ON",
        "-DGGML_CUDA=OFF",
        "-DGGML_METAL=OFF",
        "-DGGML_VULKAN=OFF",
        "-DLLAMA_CURL=OFF",
    ], log=OUT / "cmake-configure.log")
    run([
        "cmake", "--build", BUILD, "--config", "Release",
        f"-j{N_THREADS}", "--target", "llama-cli", "llama-bench",
    ], log=OUT / "cmake-build.log")
    run([CLI, "--version"], log=OUT / "llama-version.log")
    run([CLI, "--help"], log=OUT / "llama-help.log")


def fetch_model() -> dict:
    if not MODEL.exists() or MODEL.stat().st_size != MODEL_SIZE:
        run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
             "-C", "-", "-o", MODEL, MODEL_URL], log=OUT / "download.log")
    size = MODEL.stat().st_size
    if size != MODEL_SIZE:
        raise RuntimeError(f"model size {size} != expected {MODEL_SIZE}")
    digest = sha256(MODEL)
    return {
        "repo_commit": MODEL_REPO_COMMIT,
        "file": MODEL_FILE,
        "size_bytes": size,
        "sha256": digest,
        "url": MODEL_URL,
    }


def drop_own_file_cache(path: Path) -> dict:
    os.sync()
    result = {"available": hasattr(os, "posix_fadvise")}
    if not result["available"]:
        return result
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        result["called"] = True
    except OSError as exc:
        result["called"] = False
        result["error"] = repr(exc)
    finally:
        os.close(fd)
    return result


def proc_sample(pid: int) -> dict:
    out = {"mono_ns": time.monotonic_ns(), "rss_kib": 0, "rss_anon_kib": 0,
           "rss_file_kib": 0, "read_bytes": 0, "rchar": 0,
           "minor_faults": 0, "major_faults": 0, "valid": False}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                out["rss_kib"] = int(line.split()[1])
            elif line.startswith("RssAnon:"):
                out["rss_anon_kib"] = int(line.split()[1])
            elif line.startswith("RssFile:"):
                out["rss_file_kib"] = int(line.split()[1])
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("read_bytes", "rchar"):
                out[key] = int(value.strip())
        # Fields 10 and 12 in proc(5), after accounting for the parenthesized
        # comm field.  These are diagnostic companions to the I/O counters;
        # neither is mislabeled as exact physical SSD bytes.
        stat_tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        out["minor_faults"] = int(stat_tail[7])
        out["major_faults"] = int(stat_tail[9])
        out["valid"] = True
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return out


PERF_RE = re.compile(
    r"eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) runs\s*"
    r"\(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\s*\)"
)


def parse_decode_perf(stderr: str) -> dict:
    matches = PERF_RE.findall(stderr)
    if not matches:
        raise RuntimeError("llama-cli emitted no parseable decode timing")
    elapsed_ms, runs, ms_per_token, tokens_per_second = matches[-1]
    return {
        "eval_ms": float(elapsed_ms),
        "eval_runs": int(runs),
        "ms_per_token": float(ms_per_token),
        "tokens_per_second": float(tokens_per_second),
    }


def run_arm(name: str, lazy_mode: str) -> dict:
    trace = OUT / f"{name}.routes.jsonl"
    stdout = OUT / f"{name}.stdout.txt"
    stderr = OUT / f"{name}.stderr.txt"
    samples_path = OUT / f"{name}.process.jsonl"
    for path in (trace, stdout, stderr, samples_path):
        path.unlink(missing_ok=True)

    cache_drop = drop_own_file_cache(MODEL)
    cmd = [
        str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(N_THREADS),
        "-c", "512", "-n", str(N_GEN), "--temp", "0", "--seed", "1234",
        "--single-turn", "--no-display-prompt", "--no-warmup",
        "-lm", "mmap", "-lzm", lazy_mode, "-p", PROMPT,
    ]
    env = dict(os.environ, GGML_PHASE0_ROUTE_TRACE=str(trace))
    print(f"\n===== ARM {name} lazy={lazy_mode} =====", flush=True)
    print("+", " ".join(cmd), flush=True)
    t0 = time.monotonic_ns()
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=env,
    )
    samples: list[dict] = []
    stop = threading.Event()

    def sample_loop() -> None:
        while not stop.is_set():
            samples.append(proc_sample(proc.pid))
            stop.wait(0.02)

    thread = threading.Thread(target=sample_loop, daemon=True)
    thread.start()
    out, err = proc.communicate()
    stop.set()
    thread.join(timeout=2)
    final_sample = proc_sample(proc.pid)
    if final_sample["valid"]:
        samples.append(final_sample)
    t1 = time.monotonic_ns()
    stdout.write_text(out, encoding="utf-8")
    stderr.write_text(err, encoding="utf-8")
    samples_path.write_text("".join(json.dumps(x) + "\n" for x in samples), encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{name} exited {proc.returncode}: {err[-2000:]}")

    valid_samples = [x for x in samples if x["valid"]]
    peak = max((x["rss_kib"] for x in valid_samples), default=0)
    peak_anon = max((x["rss_anon_kib"] for x in valid_samples), default=0)
    peak_file = max((x["rss_file_kib"] for x in valid_samples), default=0)
    read_max = max((x["read_bytes"] for x in valid_samples), default=0)
    rchar_max = max((x["rchar"] for x in valid_samples), default=0)
    minor_faults = max((x["minor_faults"] for x in valid_samples), default=0)
    major_faults = max((x["major_faults"] for x in valid_samples), default=0)
    return {
        "name": name,
        "lazy_mode": lazy_mode,
        "command": cmd,
        "elapsed_sec": (t1 - t0) / 1e9,
        "peak_rss_mib": peak / 1024,
        "peak_rss_anon_mib": peak_anon / 1024,
        "peak_rss_file_mib": peak_file / 1024,
        "max_read_bytes": read_max,
        "max_rchar": rchar_max,
        "minor_faults": minor_faults,
        "major_faults": major_faults,
        "decode_perf": parse_decode_perf(err),
        "stdout_sha256": hashlib.sha256(out.encode()).hexdigest(),
        "stdout": out,
        "cache_drop": cache_drop,
        "sample_count": len(samples),
        "trace": str(trace),
    }


LAYER_RE = re.compile(r"blk\.(\d+)\.")


def load_decode_routes(path: Path) -> tuple[list[list[list[int]]], dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "ffn_down_exps" not in row["weight"]:
            continue
        m = LAYER_RE.search(row["weight"])
        if not m:
            continue
        row["layer"] = int(m.group(1))
        if row["shape"][0] != 8:
            raise AssertionError(f"native K changed: {row['shape']}")
        if row["shape"][1] == 1:
            events.append(row)

    tokens: list[list[list[int]]] = []
    current: list[list[int]] = []
    expected_layer = 0
    first_decode_ns = None
    last_decode_ns = None
    token_start_ns: list[int] = []
    for row in events:
        layer = row["layer"]
        if layer == 0:
            token_start_ns.append(row["mono_ns"])
        if layer == 0 and current:
            if len(current) != 40:
                raise AssertionError(f"decode token has {len(current)} layers")
            tokens.append(current)
            current = []
            expected_layer = 0
        if layer != expected_layer:
            raise AssertionError(f"expected layer {expected_layer}, got {layer}")
        ids = row["ids"]
        if len(ids) != 8 or len(set(ids)) != 8 or not all(0 <= x < 256 for x in ids):
            raise AssertionError(f"invalid route at layer {layer}: {ids}")
        current.append(ids)
        expected_layer += 1
        first_decode_ns = row["mono_ns"] if first_decode_ns is None else first_decode_ns
        last_decode_ns = row["mono_ns"]
    if current:
        if len(current) != 40:
            raise AssertionError(f"final decode token has {len(current)} layers")
        tokens.append(current)
    token_latency_ms = [
        (right - left) / 1e6 for left, right in zip(token_start_ns, token_start_ns[1:])
    ]
    ordered_latency = sorted(token_latency_ms)

    def nearest_percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        return values[round((len(values) - 1) * q)]

    return tokens, {
        "decode_events": len(events),
        "decode_tokens": len(tokens),
        "first_decode_mono_ns": first_decode_ns,
        "last_decode_mono_ns": last_decode_ns,
        "inter_token_latency_count": len(token_latency_ms),
        "inter_token_latency_p50_ms": nearest_percentile(ordered_latency, 0.50),
        "inter_token_latency_p95_ms": nearest_percentile(ordered_latency, 0.95),
    }


def routed_bundle_sizes(path: Path) -> dict[int, int]:
    tensors: dict[str, tuple[int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        match = LAYER_RE.search(row["weight"])
        if match and "weight_nbytes" in row:
            tensors[row["weight"]] = (int(match.group(1)), int(row["weight_nbytes"]))
    by_layer: dict[int, int] = {layer: 0 for layer in range(40)}
    for layer, tensor_nbytes in tensors.values():
        if tensor_nbytes % 256:
            raise AssertionError(f"routed tensor bytes {tensor_nbytes} not divisible by 256")
        by_layer[layer] += tensor_nbytes // 256
    if any(size == 0 for size in by_layer.values()):
        raise AssertionError(f"missing routed tensor size for layers: {[k for k, v in by_layer.items() if not v]}")
    return by_layer


def replay_lru(tokens: list[list[list[int]]], capacity: int,
               bundle_bytes_by_layer: dict[int, int]) -> dict:
    cache: OrderedDict[tuple[int, int], None] = OrderedDict()
    hits = misses = fresh_bytes = 0
    per_token = []
    for token in tokens:
        tm = th = 0
        for layer, ids in enumerate(token):
            for expert in ids:
                key = (layer, expert)
                if capacity and key in cache:
                    hits += 1
                    th += 1
                    cache.move_to_end(key)
                else:
                    misses += 1
                    tm += 1
                    fresh_bytes += bundle_bytes_by_layer[layer]
                    if capacity:
                        cache[key] = None
                        if len(cache) > capacity:
                            cache.popitem(last=False)
        per_token.append({"hits": th, "misses": tm})
    total = hits + misses
    return {
        "capacity_bundles": capacity,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total if total else 0.0,
        "mean_fresh_bundles_per_token": misses / len(tokens) if tokens else None,
        "fresh_expert_bytes": fresh_bytes,
        "fresh_expert_bytes_per_token": fresh_bytes / len(tokens) if tokens else None,
        "per_token": per_token,
    }


def route_metrics(tokens: list[list[list[int]]], bundle_bytes_by_layer: dict[int, int]) -> dict:
    popularity = Counter()
    overlaps = []
    for token in tokens:
        for layer, ids in enumerate(token):
            popularity.update((layer, x) for x in ids)
    for prev, cur in zip(tokens, tokens[1:]):
        for layer in range(40):
            a, b = set(prev[layer]), set(cur[layer])
            overlaps.append({
                "intersection": len(a & b),
                "jaccard": len(a & b) / len(a | b),
            })
    return {
        "decode_tokens": len(tokens),
        "route_references": len(tokens) * 40 * 8,
        "unique_layer_experts": len(popularity),
        "routed_bundle_bytes_by_layer": bundle_bytes_by_layer,
        "mean_routed_bundle_bytes": sum(bundle_bytes_by_layer.values()) / len(bundle_bytes_by_layer),
        "uncached_selected_expert_bytes_per_token": (
            sum(bundle_bytes_by_layer.values()) * 8
        ),
        "mean_previous_token_intersection": (
            sum(x["intersection"] for x in overlaps) / len(overlaps) if overlaps else None
        ),
        "mean_previous_token_jaccard": (
            sum(x["jaccard"] for x in overlaps) / len(overlaps) if overlaps else None
        ),
        "top_layer_experts": [
            {"layer": key[0], "expert": key[1], "count": count}
            for key, count in popularity.most_common(30)
        ],
        "lru": [
            replay_lru(tokens, c, bundle_bytes_by_layer)
            for c in (0, 64, 128, 256, 512, 1024, 2048)
        ],
    }


def decode_read_bytes(samples_path: Path, first_ns: int | None, last_ns: int | None) -> int | None:
    if first_ns is None or last_ns is None:
        return None
    samples = [
        row for row in (json.loads(x) for x in samples_path.read_text().splitlines())
        if row.get("valid")
    ]
    before = [x for x in samples if x["mono_ns"] <= first_ns]
    after = [x for x in samples if x["mono_ns"] >= last_ns]
    if not before or not after:
        return None
    return max(0, after[-1]["read_bytes"] - before[-1]["read_bytes"])


def run_compute_ceiling() -> dict:
    cache_drop = drop_own_file_cache(MODEL)
    cmd = [
        str(BENCH), "-m", str(MODEL), "-p", "32", "-n", "32",
        "-t", str(N_THREADS), "-r", "1", "-ngl", "0",
        "-lm", "mmap", "-lzm", "off", "--output", "json",
    ]
    p = run(cmd, log=OUT / "resident-native-bench.log")
    try:
        rows = json.loads(p.stdout[p.stdout.index("["):])
    except (ValueError, json.JSONDecodeError):
        rows = []
    return {"command": cmd, "cache_drop": cache_drop, "rows": rows, "raw_tail": p.stdout[-3000:]}


def main() -> None:
    started = time.time()
    runtime = clone_and_patch()
    build()
    model = fetch_model()
    hardware = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "threads_used": N_THREADS,
        "cpuinfo": Path("/proc/cpuinfo").read_text(encoding="utf-8")[:12000],
        "meminfo": Path("/proc/meminfo").read_text(encoding="utf-8"),
        "filesystem": run(["df", "-h", "/kaggle/working"]).stdout,
    }
    (OUT / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")

    arms = [
        run_arm("eager", "off"),
        run_arm("targeted_lazy", "auto"),
        run_arm("global_lazy", "on"),
    ]
    route_summaries = {}
    normalized_routes = {}
    for arm in arms:
        name = arm["name"]
        trace_path = OUT / f"{name}.routes.jsonl"
        tokens, validation = load_decode_routes(trace_path)
        bundle_bytes_by_layer = routed_bundle_sizes(trace_path)
        normalized_routes[name] = tokens
        route_summaries[name] = {
            "validation": validation,
            "metrics": route_metrics(tokens, bundle_bytes_by_layer),
        }
        arm["decode_read_bytes"] = decode_read_bytes(
            OUT / f"{name}.process.jsonl",
            validation["first_decode_mono_ns"],
            validation["last_decode_mono_ns"],
        )
        arm["decode_read_bytes_per_token"] = (
            arm["decode_read_bytes"] / validation["decode_tokens"]
            if arm["decode_read_bytes"] is not None and validation["decode_tokens"] else None
        )
        arm.pop("stdout")

    reference_routes = normalized_routes["eager"]
    exactness = {
        "stdout_equal": len({x["stdout_sha256"] for x in arms}) == 1,
        "routes_equal": all(v == reference_routes for v in normalized_routes.values()),
        "native_k": 8,
        "layers": 40,
        "no_drop_or_substitution": True,
    }
    if not exactness["stdout_equal"] or not exactness["routes_equal"]:
        raise AssertionError(f"exactness comparison failed: {exactness}")

    compute_ceiling = run_compute_ceiling()
    result = {
        "schema_version": 1,
        "status": "complete",
        "hypothesis": (
            "Upstream lazy tensor loading can reduce Qwen3.5 routed-expert residency "
            "without changing native top-8 outputs or routes."
        ),
        "runtime": runtime,
        "model": model,
        "hardware": {k: v for k, v in hardware.items() if k not in ("cpuinfo", "meminfo")},
        "prompt": PROMPT,
        "n_gen_requested": N_GEN,
        "arms": arms,
        "exactness": exactness,
        "routes": route_summaries,
        "compute_ceiling": compute_ceiling,
        "quality": {
            "result": "deterministic smoke equivalence only",
            "basis": "one byte-identical output and identical native routes; not a capability evaluation",
        },
        "wall_sec": time.time() - started,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(__file__, OUT / Path(__file__).name)
    print(json.dumps({
        "exactness": exactness,
        "arms": arms,
        "route_summary": route_summaries,
        "compute_ceiling": compute_ceiling,
        "wall_sec": result["wall_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
