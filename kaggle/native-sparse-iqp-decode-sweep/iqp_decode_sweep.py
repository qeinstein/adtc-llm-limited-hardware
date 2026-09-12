"""A/B the IQ panel MUL_MAT_ID path for exact single-token Qwen3.5 decode."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path


SCRATCH = Path("/tmp/native-sparse-iqp-decode")
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
BENCH = BUILD / "bin" / "llama-bench"
CLI = BUILD / "bin" / "llama-cli"
MODEL = SCRATCH / "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
OUT = Path("/kaggle/working/native-sparse-iqp-decode-results")

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_SHA256 = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
RESEARCH_BASE_COMMIT = "6adb36bd63602493902e6f8ce54c87cb71967b21"
PROMPT = "Give one concise reason oral rehydration solution helps a child with watery diarrhoea."
N_GEN = 64
N_REPEATS = 3
THREADS = 4
LOGICAL_SELECTED_EXPERT_BYTES_PER_TOKEN = 280_494_080

SUMMARY_PERF_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+) t/s\s*\|\s*Generation:\s*([0-9.]+) t/s\s*\]"
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
    cmd: list[str],
    *,
    cwd: Path | None = None,
    log: Path | None = None,
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
        log.write_text(
            process.stdout + "\n--- STDERR ---\n" + process.stderr,
            encoding="utf-8",
        )
    if process.returncode:
        print(process.stdout[-1500:], flush=True)
        print(process.stderr[-1500:], flush=True)
        raise RuntimeError(f"command exited {process.returncode}: {command_text(cmd)}")
    return process


def patch_iqp_threshold() -> dict:
    source = LLAMA / "ggml/src/ggml-cpu/iqp.cpp"
    old = """bool ggml_cpu_iqp_mul_mat_id_min_batch(int64_t cne1) {
    return cne1 >= GGML_IQP_MIN_BATCH_ID;
}"""
    new = """bool ggml_cpu_iqp_mul_mat_id_min_batch(int64_t cne1) {
    // Research A/B: preserve the upstream default, but permit a process-local
    // threshold override so single-token MoE decode can exercise the existing
    // IQ panel implementation without changing routing or tensor values.
    const char * threshold_env = getenv("GGML_IQP_MIN_BATCH_ID");
    const int64_t threshold = threshold_env != NULL ? atoll(threshold_env) : GGML_IQP_MIN_BATCH_ID;
    GGML_ASSERT(threshold > 0);
    return cne1 >= threshold;
}"""
    text = source.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected one IQP threshold anchor, found {text.count(old)}")
    source.write_text(text.replace(old, new), encoding="utf-8")
    diff = run_checked(
        ["git", "diff", "--", "ggml/src/ggml-cpu/iqp.cpp"],
        cwd=LLAMA,
    ).stdout
    (OUT / "iqp-threshold.patch").write_text(diff, encoding="utf-8")
    return {
        "description": "runtime-selectable existing IQP MUL_MAT_ID per-expert threshold",
        "sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def clone_patch_build() -> dict:
    if not LLAMA.exists():
        run_checked([
            "git", "clone", "--filter=blob:none",
            "https://github.com/ggml-org/llama.cpp.git", str(LLAMA),
        ])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    source_head = run_checked(["git", "rev-parse", "HEAD"], cwd=LLAMA).stdout.strip()
    patch = patch_iqp_threshold()
    configure = [
        "cmake", "-S", str(LLAMA), "-B", str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF",
        "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF",
    ]
    run_checked(configure, log=OUT / "cmake-configure.log")
    build = [
        "cmake", "--build", str(BUILD), "--config", "Release",
        "-j4", "--target", "llama-bench", "llama-cli",
    ]
    run_checked(build, log=OUT / "cmake-build.log")
    help_result = run_checked([str(BENCH), "--help"])
    (OUT / "llama-bench-help.txt").write_text(help_result.stdout, encoding="utf-8")
    return {
        "commit": LLAMA_COMMIT,
        "source_head": source_head,
        "patch": patch,
        "configure_command": configure,
        "build_command": build,
    }


def fetch_model() -> dict:
    if not MODEL.exists() or MODEL.stat().st_size != MODEL_SIZE:
        run_checked([
            "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
            "-C", "-", "-o", str(MODEL), MODEL_URL,
        ], log=OUT / "model-download.log")
    size = MODEL.stat().st_size
    digest = sha256(MODEL)
    if size != MODEL_SIZE or digest != MODEL_SHA256:
        raise RuntimeError(f"model identity mismatch: size={size}, sha256={digest}")
    return {
        "repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "repo_commit": MODEL_REPO_COMMIT,
        "file": MODEL_FILE,
        "size_bytes": size,
        "sha256": digest,
        "url": MODEL_URL,
    }


def strict_bench_json(stdout: str) -> tuple[dict | None, str | None]:
    text = stdout.strip()
    if not text:
        return None, "empty stdout"
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"stdout is not strict JSON: {exc}"
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        return None, "expected exactly one JSON result row"
    row = value[0]
    required = {"n_threads", "n_prompt", "n_gen", "avg_ts", "samples_ts"}
    if not required.issubset(row):
        return None, "result row is missing required fields"
    if row["n_threads"] != THREADS or row["n_prompt"] != 0 or row["n_gen"] != N_GEN:
        return None, "result row does not match the requested decode configuration"
    if not isinstance(row["avg_ts"], (int, float)) or row["avg_ts"] <= 0:
        return None, "invalid avg_ts"
    samples = row["samples_ts"]
    if (
        not isinstance(samples, list)
        or len(samples) != N_REPEATS
        or not all(isinstance(x, (int, float)) and x > 0 for x in samples)
    ):
        return None, "invalid samples_ts"
    return row, None


def read_peak_rss_kib(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^VmHWM:\s+(\d+)\s+kB$", text, re.MULTILINE)
    if match:
        return int(match.group(1))
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def run_benchmark(label: str, env_overrides: dict[str, str]) -> dict:
    cmd = [
        str(BENCH), "-m", str(MODEL), "-p", "0", "-n", str(N_GEN),
        "-r", str(N_REPEATS), "-t", str(THREADS), "-ngl", "0",
        "-lm", "mmap", "-lzm", "off", "--poll", "0",
        "--output", "json",
    ]
    print("+", command_text(cmd), "ENV", env_overrides, flush=True)
    env = dict(os.environ)
    env.update(env_overrides)
    started = time.monotonic()
    process = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    peak_kib = 0
    while process.poll() is None:
        current = read_peak_rss_kib(process.pid)
        if current is not None:
            peak_kib = max(peak_kib, current)
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    (OUT / f"{label}.stdout.json").write_text(stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(stderr, encoding="utf-8")
    row, parse_error = (
        strict_bench_json(stdout)
        if process.returncode == 0
        else (None, f"nonzero exit {process.returncode}")
    )
    result = {
        "label": label,
        "environment": env_overrides,
        "command": cmd,
        "returncode": process.returncode,
        "elapsed_wall_seconds": time.monotonic() - started,
        "peak_rss_kib": peak_kib or None,
        "parse_ok": row is not None,
        "parse_error": parse_error,
        "decode_row": row,
        "stderr_tail": stderr[-2000:],
    }
    if row is not None:
        result["decode_tok_s"] = float(row["avg_ts"])
        result["sample_tok_s"] = [float(x) for x in row["samples_ts"]]
    print(json.dumps({
        "label": label,
        "returncode": process.returncode,
        "parse_ok": result["parse_ok"],
        "decode_tok_s": result.get("decode_tok_s"),
        "peak_rss_kib": result["peak_rss_kib"],
    }), flush=True)
    return result


def generated_response(stdout: str) -> str:
    marker = stdout.find("[Start thinking]")
    if marker < 0:
        raise RuntimeError("cannot locate generated response marker")
    tail = stdout[marker:]
    summary = SUMMARY_PERF_RE.search(tail)
    if summary:
        tail = tail[:summary.start()]
    if tail.rstrip().endswith("Exiting..."):
        tail = tail.rstrip()[:-len("Exiting...")]
    response = tail.strip()
    if not response:
        raise RuntimeError("generated response payload is empty")
    return response


def run_correctness(label: str, env_overrides: dict[str, str]) -> dict:
    cmd = [
        str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(THREADS),
        "-c", "512", "-n", "24", "--temp", "0", "--seed", "1234",
        "--single-turn", "--no-display-prompt", "--no-warmup", "--perf",
        "-lm", "mmap", "-lzm", "off", "--poll", "0", "-p", PROMPT,
    ]
    env = dict(os.environ)
    env.update(env_overrides)
    print("+", command_text(cmd), "ENV", env_overrides, flush=True)
    process = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    (OUT / f"{label}.stdout.txt").write_text(process.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(process.stderr, encoding="utf-8")
    response = generated_response(process.stdout) if process.returncode == 0 else None
    return {
        "label": label,
        "environment": env_overrides,
        "command": cmd,
        "returncode": process.returncode,
        "response": response,
        "response_sha256": (
            hashlib.sha256(response.encode()).hexdigest()
            if response is not None
            else None
        ),
    }


def hardware_snapshot() -> dict:
    def read(path: str) -> str | None:
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError:
            return None
    try:
        allowed_cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed_cpus = list(range(os.cpu_count() or 0))
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "allowed_cpu_ids": allowed_cpus,
        "cpuinfo": read("/proc/cpuinfo"),
        "meminfo": read("/proc/meminfo"),
        "lscpu": subprocess.run(
            ["lscpu"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout,
    }


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "schema": "native-sparse-iqp-decode-sweep/v1",
        "status": "failed",
        "hypothesis": (
            "Allowing the existing IQ panel MUL_MAT_ID kernel at one routed "
            "row per expert may improve exact single-token IQ2_XXS decode."
        ),
        "started_unix": time.time(),
        "research_source": {
            "base_commit": RESEARCH_BASE_COMMIT,
            "script_sha256": sha256(Path(__file__)),
        },
        "benchmark": {
            "n_prompt": 0,
            "n_gen": N_GEN,
            "repeats": N_REPEATS,
            "threads": THREADS,
            "poll": 0,
            "load_mode": "mmap",
            "lazy_mode": "off",
            "gpu_layers": 0,
        },
        "traffic": {
            "resident_expert_pool": True,
            "scheduled_fresh_expert_bytes_per_token": 0,
            "logical_selected_expert_bytes_per_token": LOGICAL_SELECTED_EXPERT_BYTES_PER_TOKEN,
        },
        "hardware": hardware_snapshot(),
        "runs": [],
    }
    try:
        result["build"] = clone_patch_build()
        result["model"] = fetch_model()
        arms = [
            ("upstream_threshold_8", {}),
            ("forced_iqp_threshold_1", {"GGML_IQP_MIN_BATCH_ID": "1"}),
            (
                "forced_threshold_but_iqp_disabled",
                {"GGML_IQP_MIN_BATCH_ID": "1", "GGML_NO_IQ_PANEL": "1"},
            ),
        ]
        for label, environment in arms:
            result["runs"].append(run_benchmark(label, environment))
        correctness_runs = [
            run_correctness(
                "correctness_generic",
                {"GGML_IQP_MIN_BATCH_ID": "1", "GGML_NO_IQ_PANEL": "1"},
            ),
            run_correctness(
                "correctness_forced_iqp",
                {"GGML_IQP_MIN_BATCH_ID": "1"},
            ),
        ]
        response_equal = (
            all(run["returncode"] == 0 for run in correctness_runs)
            and correctness_runs[0]["response_sha256"]
            == correctness_runs[1]["response_sha256"]
        )
        result["correctness"] = {
            "runs": correctness_runs,
            "response_equal": response_equal,
            "native_k": 8,
            "layers": 40,
            "router_or_weights_changed": False,
            "no_drop_or_substitution": True,
        }
        all_benchmarks_valid = all(run["parse_ok"] for run in result["runs"])
        result["status"] = "ok" if all_benchmarks_valid and response_equal else "invalid_result"
    except Exception as exc:
        result["error"] = repr(exc)
        print(f"ERROR: {exc}", flush=True)
    result["finished_unix"] = time.time()
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "runs": [
            [run["label"], run.get("decode_tok_s")]
            for run in result.get("runs", [])
        ],
        "response_equal": result.get("correctness", {}).get("response_equal"),
        "error": result.get("error"),
    }), flush=True)
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
