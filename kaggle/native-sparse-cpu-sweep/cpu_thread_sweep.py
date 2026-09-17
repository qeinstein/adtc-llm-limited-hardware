"""Exact native Qwen3.5 MoE CPU thread/poll/affinity sweep.

The build and model live under /tmp.  Only a compact JSON report is retained
under /kaggle/working.  No expert is dropped and no routing approximation is
made: this is an unmodified, pinned llama.cpp resident benchmark.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path


SCRATCH = Path("/tmp/native-sparse-cpu-sweep")
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
BENCH = BUILD / "bin" / "llama-bench"
MODEL = SCRATCH / "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
OUT = Path("/kaggle/working/native-sparse-cpu-sweep-results")

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
N_GEN = 64
N_REPEATS = 3
THREADS = (1, 2, 3, 4)
RESEARCH_BASE_COMMIT = "58594d2d8f53ad7628ece93c17b636fb153e812f"

def command_text(cmd: list[str]) -> str:
    return " ".join(str(x) for x in cmd)


def run_checked(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("+", command_text(cmd), flush=True)
    p = subprocess.run(
        [str(x) for x in cmd], cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if p.returncode:
        print(p.stdout[-1500:], flush=True)
        print(p.stderr[-1500:], flush=True)
        raise RuntimeError(f"command exited {p.returncode}: {command_text(cmd)}")
    return p


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clone_and_build() -> tuple[dict, str]:
    if not LLAMA.exists():
        run_checked([
            "git", "clone", "--filter=blob:none",
            "https://github.com/ggml-org/llama.cpp.git", str(LLAMA),
        ])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    configure = [
        "cmake", "-S", str(LLAMA), "-B", str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF",
        "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF",
    ]
    run_checked(configure)
    build = [
        "cmake", "--build", str(BUILD), "--config", "Release",
        "-j4", "--target", "llama-bench",
    ]
    run_checked(build)
    source_head = run_checked(["git", "rev-parse", "HEAD"], cwd=LLAMA)
    help_result = run_checked([str(BENCH), "--help"])
    (OUT / "llama-help.txt").write_text(help_result.stdout, encoding="utf-8")
    return {
        "commit": LLAMA_COMMIT,
        "source_head": source_head.stdout.strip(),
        "configure_command": configure,
        "build_command": build,
    }, help_result.stdout


def fetch_model() -> dict:
    if not MODEL.exists() or MODEL.stat().st_size != MODEL_SIZE:
        run_checked([
            "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
            "-C", "-", "-o", str(MODEL), MODEL_URL,
        ])
    size = MODEL.stat().st_size
    if size != MODEL_SIZE:
        raise RuntimeError(f"model size {size} != expected {MODEL_SIZE}")
    return {
        "repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "repo_commit": MODEL_REPO_COMMIT,
        "file": MODEL_FILE,
        "size_bytes": size,
        "sha256": sha256(MODEL),
        "url": MODEL_URL,
    }


def strict_bench_json(stdout: str) -> tuple[list[dict] | None, str | None]:
    """Fail closed: do not salvage JSON from mixed or truncated stdout."""
    text = stdout.strip()
    if not text:
        return None, "empty stdout"
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"stdout is not strict JSON: {exc}"
    if not isinstance(value, list) or not value or not all(isinstance(row, dict) for row in value):
        return None, "JSON root is not a non-empty array of objects"
    required = {"n_threads", "n_prompt", "n_gen", "avg_ts", "samples_ts"}
    for index, row in enumerate(value):
        if not required.issubset(row):
            return None, f"row {index} missing required fields"
        if row["n_prompt"] != 0 or row["n_gen"] != N_GEN:
            return None, f"row {index} is not requested decode n={N_GEN}"
        if not isinstance(row["avg_ts"], (int, float)) or row["avg_ts"] <= 0:
            return None, f"row {index} has invalid avg_ts"
        samples = row["samples_ts"]
        if not isinstance(samples, list) or len(samples) != N_REPEATS:
            return None, f"row {index} does not contain {N_REPEATS} samples"
        if not all(isinstance(x, (int, float)) and x > 0 for x in samples):
            return None, f"row {index} has invalid samples_ts"
    return value, None


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


def run_benchmark(label: str, extra: list[str], threads: int) -> dict:
    cmd = [
        str(BENCH), "-m", str(MODEL), "-p", "0", "-n", str(N_GEN),
        "-r", str(N_REPEATS), "-t", str(threads), "-ngl", "0",
        "-lm", "mmap", "-lzm", "off", "--output", "json", *extra,
    ]
    print("+", command_text(cmd), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    peak_kib = 0
    while process.poll() is None:
        current = read_peak_rss_kib(process.pid)
        if current is not None:
            peak_kib = max(peak_kib, current)
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    current = read_peak_rss_kib(process.pid)
    if current is not None:
        peak_kib = max(peak_kib, current)
    rows, parse_error = (
        strict_bench_json(stdout) if process.returncode == 0
        else (None, "nonzero exit")
    )
    decoded = None
    if rows is not None:
        matches = [row for row in rows if row["n_threads"] == threads]
        if len(matches) != 1:
            rows = None
            parse_error = f"expected one row for n_threads={threads}, got {len(matches)}"
        else:
            decoded = matches[0]
    result = {
        "label": label,
        "threads": threads,
        "extra_args": extra,
        "command": cmd,
        "command_text": command_text(cmd),
        "returncode": process.returncode,
        "elapsed_wall_seconds": time.monotonic() - started,
        "peak_rss_kib": peak_kib or None,
        "parse_ok": decoded is not None,
        "parse_error": parse_error,
        "decode_row": decoded,
        "stderr_tail": stderr[-2000:],
    }
    if decoded is not None:
        result["decode_tok_s"] = float(decoded["avg_ts"])
        result["sample_tok_s"] = [float(x) for x in decoded["samples_ts"]]
    print(json.dumps({
        "label": label, "returncode": process.returncode,
        "parse_ok": result["parse_ok"],
        "decode_tok_s": result.get("decode_tok_s"),
        "peak_rss_kib": result["peak_rss_kib"],
        "parse_error": parse_error,
    }), flush=True)
    return result


def supported_options(help_text: str) -> dict[str, bool]:
    return {
        "poll": bool(re.search(r"(?:^|\s)--poll(?:\s|$)", help_text, re.MULTILINE)),
        "cpu_mask": "--cpu-mask" in help_text,
        "cpu_range": "--cpu-range" in help_text,
        "cpu_strict": "--cpu-strict" in help_text,
    }


def allowed_cpu_ids() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 0))


def physical_first_cpu_ids(cpu_ids: list[int]) -> list[int]:
    """Prefer one logical CPU per physical core before SMT siblings."""
    by_core: dict[tuple[str, str], list[int]] = {}
    unknown: list[int] = []
    for cpu in cpu_ids:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            unknown.append(cpu)
            continue
        by_core.setdefault((package, core), []).append(cpu)
    ordered: list[int] = []
    max_siblings = max((len(ids) for ids in by_core.values()), default=0)
    for sibling_index in range(max_siblings):
        for key in sorted(by_core):
            ids = sorted(by_core[key])
            if sibling_index < len(ids):
                ordered.append(ids[sibling_index])
    return ordered + [cpu for cpu in unknown if cpu not in ordered]


def strict_affinity_args(cpu_ids: list[int], options: dict[str, bool]) -> list[str] | None:
    if not cpu_ids or not options["cpu_strict"]:
        return None
    contiguous = cpu_ids == list(range(cpu_ids[0], cpu_ids[-1] + 1))
    if contiguous and options["cpu_range"]:
        return ["--cpu-range", f"{cpu_ids[0]}-{cpu_ids[-1]}", "--cpu-strict", "1"]
    if options["cpu_mask"]:
        mask = sum(1 << cpu for cpu in cpu_ids)
        return ["--cpu-mask", hex(mask), "--cpu-strict", "1"]
    return None


def hardware_snapshot() -> dict:
    def read(path: str) -> str | None:
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError:
            return None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "allowed_cpu_ids": allowed_cpu_ids(),
        "cpuinfo": read("/proc/cpuinfo"),
        "meminfo": read("/proc/meminfo"),
        "lscpu": subprocess.run(["lscpu"], text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, check=False).stdout,
        "df_working": subprocess.run(["df", "-h", "/kaggle/working"], text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      check=False).stdout,
    }


def best_of(runs: list[dict]) -> dict | None:
    valid = [r for r in runs if r.get("parse_ok") and r.get("decode_tok_s", 0) > 0]
    if not valid:
        return None
    row = max(valid, key=lambda r: r["decode_tok_s"])
    return {
        "label": row["label"], "threads": row["threads"],
        "decode_tok_s": row["decode_tok_s"],
        "peak_rss_kib": row["peak_rss_kib"], "command": row["command"],
    }


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "schema": "native-sparse-cpu-sweep/v1",
        "status": "failed",
        "started_unix": time.time(),
        "benchmark": {
            "n_prompt": 0, "n_gen": N_GEN, "repeats": N_REPEATS,
            "load_mode": "mmap", "lazy_mode": "off", "gpu_layers": 0,
            "description": "unmodified llama.cpp exact native router/top-K resident decode",
        },
        "research_source": {
            "base_commit": RESEARCH_BASE_COMMIT,
            "script_sha256": sha256(Path(__file__)),
        },
        "hardware": hardware_snapshot(),
        "runs": [],
    }
    try:
        result["build"], help_text = clone_and_build()
        result["supported_options"] = supported_options(help_text)
        result["model"] = fetch_model()

        # Required thread sweep, exactly t=1,2,3,4 and decode n=64, r=3.
        for threads in THREADS:
            result["runs"].append(run_benchmark(f"threads_{threads}_default", [], threads))

        valid = [r for r in result["runs"] if r.get("parse_ok")]
        if valid:
            best_threads = max(valid, key=lambda r: r["decode_tok_s"])["threads"]
            options = result["supported_options"]
            # Optional arms are run only when the pinned llama-bench advertises them.
            if options["poll"]:
                result["runs"].append(run_benchmark("best_poll_0", ["--poll", "0"], best_threads))
                result["runs"].append(run_benchmark("best_poll_100", ["--poll", "100"], best_threads))
            affinity_order = physical_first_cpu_ids(allowed_cpu_ids())
            selected_cpus = affinity_order[:best_threads]
            affinity_args = strict_affinity_args(selected_cpus, options)
            result["affinity_selection"] = {
                "order": affinity_order,
                "selected_cpu_ids": selected_cpus,
                "args": affinity_args,
            }
            if len(selected_cpus) == best_threads and affinity_args is not None:
                result["runs"].append(run_benchmark(
                    "best_affinity_strict",
                    affinity_args,
                    best_threads,
                ))
        result["best"] = best_of(result["runs"])
        result["status"] = "ok" if result["best"] else "no_valid_measurement"
    except Exception as exc:
        result["error"] = repr(exc)
        print(f"ERROR: {exc}", flush=True)
    result["finished_unix"] = time.time()
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "best": result.get("best"),
                      "error": result.get("error")}), flush=True)
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
