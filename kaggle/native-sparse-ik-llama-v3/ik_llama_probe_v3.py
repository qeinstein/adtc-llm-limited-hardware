"""Kaggle-only compatibility probe for the official ik_llama.cpp baseline.

This script intentionally does not patch ik_llama.cpp.  It checks out an
immutable upstream commit, records source anchors, builds the CPU binaries,
downloads the exact Phase-0 checkpoint into /tmp, and runs a single-stream
smoke/bench.  It is an external resident baseline: Phase-0 TENSOR_READ_LAZY
semantics are not assumed to exist here.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path


SCRATCH = Path("/tmp/native-sparse-ik-llama-v3")
UPSTREAM = SCRATCH / "ik_llama.cpp"
BUILD = UPSTREAM / "build-native"
MODEL = SCRATCH / "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
OUT = Path("/kaggle/working/native-sparse-ik-llama-v3-results")

IK_REPO = "https://github.com/ikawrakow/ik_llama.cpp.git"
IK_COMMIT = "3bb386eb68ffee0a5dc7db21da0735d594929eeb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_SHA256 = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
RESEARCH_BASE_COMMIT = "e5bb66e"
PROMPT = "Give one concise reason oral rehydration solution helps a child with watery diarrhoea."
THREADS = 4
CONTEXT = 512
N_GEN = 64
BENCH_GEN = 64
BENCH_REPEATS = 3

SOURCE_PATTERNS = {
    "qwen35moe": re.compile(r"qwen35moe", re.IGNORECASE),
    "routed_gate": re.compile(r"ffn_gate_exps"),
    "routed_up": re.compile(r"ffn_up_exps"),
    "routed_down": re.compile(r"ffn_down_exps"),
    "iq2_xxs": re.compile(r"IQ2_XXS|iq2_xxs"),
    "mul_mat_id": re.compile(r"GGML_OP_MUL_MAT_ID|ggml_mul_mat_id"),
    "lazy_tensor": re.compile(r"TENSOR_READ_LAZY|READ_LAZY"),
    "one_sequence": re.compile(r"one sequence|single.sequence|n_seq", re.IGNORECASE),
}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hh", ".hpp", ".md", ".cmake", ".txt"}


def command_text(command: list[str]) -> str:
    return " ".join(str(item) for item in command)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_capture(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log: Path | None = None,
) -> subprocess.CompletedProcess:
    print("+", command_text(command), flush=True)
    process = subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        env=env,
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
    return process


def require_ok(process: subprocess.CompletedProcess, command: list[str]) -> None:
    if process.returncode:
        raise RuntimeError(
            f"command exited {process.returncode}: {command_text(command)}"
        )


def hardware_snapshot() -> dict:
    def read(path: str) -> str | None:
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError:
            return None

    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = list(range(os.cpu_count() or 0))
    lscpu = run_capture(["lscpu"])
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "allowed_cpu_ids": allowed,
        "lscpu": lscpu.stdout,
        "cpuinfo": read("/proc/cpuinfo"),
        "meminfo": read("/proc/meminfo"),
        "filesystem": run_capture(["df", "-h", "/tmp"]).stdout,
    }


def setup_upstream() -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not UPSTREAM.exists():
        clone = run_capture(
            ["git", "clone", "--filter=blob:none", "--no-checkout", IK_REPO, str(UPSTREAM)],
            log=OUT / "git-clone.log",
        )
        require_ok(clone, ["git", "clone", "--filter=blob:none", "--no-checkout", IK_REPO, str(UPSTREAM)])
    fetch_cmd = ["git", "fetch", "--depth", "1", "origin", IK_COMMIT]
    fetch = run_capture(fetch_cmd, cwd=UPSTREAM, log=OUT / "git-fetch.log")
    require_ok(fetch, fetch_cmd)
    checkout_cmd = ["git", "checkout", "--detach", IK_COMMIT]
    checkout = run_capture(checkout_cmd, cwd=UPSTREAM, log=OUT / "git-checkout.log")
    require_ok(checkout, checkout_cmd)

    head_cmd = ["git", "rev-parse", "HEAD"]
    head = run_capture(head_cmd, cwd=UPSTREAM)
    require_ok(head, head_cmd)
    source_status = run_capture(["git", "status", "--short", "--branch"], cwd=UPSTREAM).stdout
    source_log = run_capture(["git", "log", "--oneline", "--decorate", "-5"], cwd=UPSTREAM).stdout
    source_matches = inspect_sources()

    configure = [
        "cmake", "-S", str(UPSTREAM), "-B", str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF",
        "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF",
    ]
    configured = run_capture(configure, log=OUT / "cmake-configure.log")
    require_ok(configured, configure)
    build_cli = [
        "cmake", "--build", str(BUILD), "--config", "Release", "-j4",
        "--target", "llama-cli",
    ]
    built_cli = run_capture(build_cli, log=OUT / "cmake-build-cli.log")
    require_ok(built_cli, build_cli)
    build_bench = [
        "cmake", "--build", str(BUILD), "--config", "Release", "-j4",
        "--target", "llama-bench",
    ]
    built_bench = run_capture(build_bench, log=OUT / "cmake-build-bench.log")

    cli = find_binary("llama-cli")
    bench = find_binary("llama-bench", required=False)
    version = run_capture([str(cli), "--version"], log=OUT / "llama-version.log")
    require_ok(version, [str(cli), "--version"])
    # ik_llama's CLI prints a complete usage page but returns status 1 for
    # --help. Treat that conventional behavior as success when the usage
    # text is present; the v1 probe incorrectly rejected a usable build here.
    help_result = run_capture([str(cli), "--help"], log=OUT / "llama-cli-help.log")
    if help_result.returncode and "usage:" not in help_result.stdout.lower():
        require_ok(help_result, [str(cli), "--help"])
    if bench:
        bench_help_result = run_capture([str(bench), "--help"], log=OUT / "llama-bench-help.log")
        bench_help = bench_help_result.stdout + "\n--- STDERR ---\n" + bench_help_result.stderr
    else:
        bench_help = ""

    return {
        "repo": IK_REPO,
        "requested_commit": IK_COMMIT,
        "checked_out_commit": head.stdout.strip(),
        "status": source_status,
        "log": source_log,
        "source_probe": source_matches,
        "configure_command": configure,
        "build_commands": {"cli": build_cli, "bench": build_bench},
        "build_results": {
            "cli_returncode": built_cli.returncode,
            "bench_returncode": built_bench.returncode,
            "bench_stderr_tail": built_bench.stderr[-3000:],
        },
        "compiler": run_capture(["c++", "--version"]).stdout.splitlines()[0],
        "cmake": run_capture(["cmake", "--version"]).stdout.splitlines()[0],
        "cli": str(cli),
        "bench": str(bench) if bench else None,
        "cli_help": help_result.stdout,
        "bench_help": bench_help,
        "version": version.stdout + "\n--- STDERR ---\n" + version.stderr,
        "help_returncode": help_result.returncode,
    }


def find_binary(name: str, *, required: bool = True) -> Path | None:
    candidates = [BUILD / "bin" / name, BUILD / name]
    candidates.extend(sorted(BUILD.rglob(name)))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    if required:
        raise RuntimeError(f"built binary not found: {name}")
    return None


def inspect_sources() -> dict:
    matches: dict[str, list[dict]] = {key: [] for key in SOURCE_PATTERNS}
    paths: set[str] = set()
    for root, dirs, files in os.walk(UPSTREAM):
        dirs[:] = [item for item in dirs if item not in {".git", "build", "build-native"}]
        for filename in files:
            path = Path(root) / filename
            if path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            relative = str(path.relative_to(UPSTREAM))
            for number, line in enumerate(lines, 1):
                for key, pattern in SOURCE_PATTERNS.items():
                    if pattern.search(line):
                        paths.add(relative)
                        if len(matches[key]) < 80:
                            matches[key].append({"path": relative, "line": number, "text": line[:400]})
    return {"matching_paths": sorted(paths), "matches": matches}


def fetch_model() -> dict:
    if not MODEL.exists() or MODEL.stat().st_size != MODEL_SIZE:
        command = [
            "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
            "-C", "-", "-o", str(MODEL), MODEL_URL,
        ]
        downloaded = run_capture(command, log=OUT / "model-download.log")
        require_ok(downloaded, command)
    size = MODEL.stat().st_size
    digest = sha256(MODEL)
    if size != MODEL_SIZE or digest != MODEL_SHA256:
        raise RuntimeError(f"model identity mismatch: size={size}, sha256={digest}")
    return {
        "repo": "unsloth/Qwen3.5-35B-A3B-GGUF",
        "revision": MODEL_REPO_COMMIT,
        "file": MODEL_FILE,
        "size_bytes": size,
        "sha256": digest,
        "url": MODEL_URL,
    }


def parse_proc_status(pid: int) -> dict:
    result: dict[str, int] = {}
    text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    wanted = {"VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSwap"}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in wanted:
            fields = rest.split()
            if fields:
                result[key] = int(fields[0])
    return result


def parse_proc_smaps_rollup(pid: int) -> dict:
    result: dict[str, int] = {}
    text = Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="utf-8")
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if fields and fields[0].isdigit():
            result[key] = int(fields[0])
    return result


def parse_proc_io(pid: int) -> dict:
    result: dict[str, int] = {}
    text = Path(f"/proc/{pid}/io").read_text(encoding="utf-8")
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if value.strip().isdigit():
            result[key] = int(value.strip())
    return result


def parse_proc_stat(pid: int) -> dict:
    text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = text.rsplit(")", 1)[1].split()
    return {"minor_faults": int(fields[7]), "major_faults": int(fields[9])}


def proc_sample(pid: int) -> dict:
    sample: dict = {"mono_ns": time.monotonic_ns()}
    try:
        sample.update(parse_proc_status(pid))
        sample["smaps_rollup"] = parse_proc_smaps_rollup(pid)
        sample["io"] = parse_proc_io(pid)
        sample.update(parse_proc_stat(pid))
        sample["valid"] = True
    except (OSError, ValueError):
        sample["valid"] = False
    return sample


def capture_smaps(pid: int, label: str) -> dict:
    try:
        text = Path(f"/proc/{pid}/smaps").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"available": False}
    destination = OUT / f"{label}.smaps.sampled.txt"
    destination.write_text(text, encoding="utf-8")
    model_rows = []
    current_header = None
    current: dict[str, int | str] = {}
    for line in text.splitlines():
        if line and line[0] not in " \t" and "-" in line.split()[0]:
            if current_header and "Qwen3.5" in current_header:
                model_rows.append({"header": current_header, **current})
            current_header = line
            current = {}
            continue
        key, _, rest = line.partition(":")
        fields = rest.split()
        if fields and fields[0].isdigit() and key in {"Rss", "Pss", "Private_Clean", "Shared_Clean", "Private_Dirty", "Shared_Dirty", "Anonymous"}:
            current[key] = int(fields[0])
    if current_header and "Qwen3.5" in current_header:
        model_rows.append({"header": current_header, **current})
    return {
        "available": True,
        "path": str(destination),
        "trigger": "first in-flight sample with VmRSS >= 1 GiB",
        "model_vmas": model_rows,
    }


def drop_model_file_cache() -> dict:
    result = {"available": hasattr(os, "posix_fadvise"), "called": False}
    if not result["available"]:
        return result
    try:
        os.sync()
        fd = os.open(MODEL, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            result["called"] = True
        finally:
            os.close(fd)
    except OSError as exc:
        result["error"] = repr(exc)
    return result


def option_supported(help_text: str, option: str) -> bool:
    return option in help_text


def first_supported(help_text: str, *options: str) -> str:
    for option in options:
        if option_supported(help_text, option):
            return option
    raise RuntimeError(f"none of the CLI option aliases are supported: {options}")


def smoke_command(cli: Path, help_text: str) -> list[str]:
    model_option = first_supported(help_text, "-m", "--model")
    thread_option = first_supported(help_text, "-t", "--threads")
    context_option = first_supported(help_text, "-c", "--ctx-size")
    predict_option = first_supported(help_text, "-n", "--n-predict")
    prompt_option = first_supported(help_text, "-p", "--prompt")
    temp_option = first_supported(help_text, "--temp", "--temperature")
    seed_option = first_supported(help_text, "--seed")
    command = [str(cli), model_option, str(MODEL)]
    # ik_llama's CLI is CPU-only by default at this build and does not expose
    # llama.cpp's -ngl alias. Add a GPU override only when the help page has it.
    if option_supported(help_text, "-ngl") or option_supported(help_text, "--gpu-layers"):
        gpu_option = first_supported(help_text, "-ngl", "--gpu-layers")
        command += [gpu_option, "0"]
    command += [thread_option, str(THREADS), context_option, str(CONTEXT),
                predict_option, str(N_GEN), temp_option, "0", seed_option, "1234",
                prompt_option, PROMPT]
    optional = [
        ("--single-turn", ["--single-turn"]),
        ("--no-display-prompt", ["--no-display-prompt"]),
        ("--no-warmup", ["--no-warmup"]),
        ("--perf", ["--perf"]),
        ("-lm", ["-lm", "mmap"]),
    ]
    for marker, args in optional:
        if option_supported(help_text, marker):
            command[-2:-2] = args
    return command


def generated_text(stdout: str, stderr: str) -> str | None:
    # Keep parsing deliberately conservative.  We only need a non-empty smoke
    # payload and same-binary determinism, not a cross-runtime format claim.
    del stderr
    text = stdout
    marker = text.find("[Start thinking]")
    if marker >= 0:
        text = text[marker:]
    text = re.split(r"\n(?:llama_|prompt eval time|eval time|total time|load time)", text, maxsplit=1)[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    ignored = {"Exiting..."}
    payload = [line for line in lines if line not in ignored]
    return "\n".join(payload) if payload else None


def run_process(label: str, command: list[str], *, cache_drop: dict | None = None) -> dict:
    stdout_path = OUT / f"{label}.stdout.txt"
    stderr_path = OUT / f"{label}.stderr.txt"
    samples_path = OUT / f"{label}.process.jsonl"
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(os.environ, OMP_NUM_THREADS=str(THREADS)),
    )
    samples = []
    sampled_smaps = {"available": False}
    smaps_captured = False
    with samples_path.open("w", encoding="utf-8") as stream:
        while process.poll() is None:
            sample = proc_sample(process.pid)
            samples.append(sample)
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            if (
                not smaps_captured
                and sample.get("valid")
                and int(sample.get("VmRSS", 0)) >= 1_048_576
            ):
                sampled_smaps = capture_smaps(process.pid, label)
                smaps_captured = sampled_smaps.get("available", False)
            time.sleep(0.02)
    stdout, stderr = process.communicate()
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    valid = [item for item in samples if item.get("valid")]
    peak = lambda key: max((int(item.get(key, 0)) for item in valid), default=0)
    peak_rollup = lambda key: max((int(item.get("smaps_rollup", {}).get(key, 0)) for item in valid), default=0)
    response = generated_text(stdout, stderr)
    combined = stdout + "\n" + stderr
    return {
        "label": label,
        "command": command,
        "returncode": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "cache_drop": cache_drop,
        "peak_status_kib": {
            "VmRSS": peak("VmRSS"), "VmHWM": peak("VmHWM"),
            "RssAnon": peak("RssAnon"), "RssFile": peak("RssFile"), "VmSwap": peak("VmSwap"),
        },
        "peak_smaps_rollup_kib": {
            key: peak_rollup(key)
            for key in ["Rss", "Pss", "Anonymous", "AnonHugePages", "Private_Clean", "Shared_Clean", "Private_Dirty", "Shared_Dirty", "Referenced", "Swap"]
        },
        "samples": len(valid),
        "process_sample_file": str(samples_path),
        "sampled_smaps": sampled_smaps,
        "minor_faults": peak("minor_faults"),
        "major_faults": peak("major_faults"),
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "response_sha256": hashlib.sha256(response.encode()).hexdigest() if response else None,
        "response_nonempty": bool(response),
        "contains_nan_or_inf": bool(re.search(r"\b(?:nan|inf)\b", combined, re.IGNORECASE)),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_tail": stderr[-3000:],
    }


def parse_bench(stdout: str, stderr: str) -> dict:
    text = stdout.strip()
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return {"format": "json", "rows": value}
    except json.JSONDecodeError:
        pass
    rates = [float(item) for item in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*t/s", stdout + "\n" + stderr)]
    return {"format": "text", "tok_s_mentions": rates}


def run_bench(bench: Path, help_text: str) -> dict:
    required = ["-m", "-p", "-n", "-r", "-t", "-ngl"]
    missing = [item for item in required if not option_supported(help_text, item)]
    if missing:
        return {"status": "unsupported", "missing_options": missing}
    command = [
        str(bench), "-m", str(MODEL), "-p", "0", "-n", str(BENCH_GEN),
        "-r", str(BENCH_REPEATS), "-t", str(THREADS), "-ngl", "0",
    ]
    if option_supported(help_text, "--output"):
        command += ["--output", "json"]
    run = run_process("resident_bench", command, cache_drop=drop_model_file_cache())
    run["bench_parse"] = parse_bench(
        (OUT / "resident_bench.stdout.txt").read_text(encoding="utf-8"),
        (OUT / "resident_bench.stderr.txt").read_text(encoding="utf-8"),
    )
    run["status"] = "ok" if run["returncode"] == 0 else "failed"
    return run


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "schema": "native-sparse-ik-llama-v3",
        "status": "failed",
        "hypothesis": "The official ik_llama.cpp AVX2 CPU path can load and single-stream run the exact Phase-0 Qwen3.5-35B-A3B IQ2_XXS checkpoint.",
        "research_source": {"base_commit": RESEARCH_BASE_COMMIT, "script_sha256": sha256(Path(__file__))},
        "upstream": {"repo": IK_REPO, "commit": IK_COMMIT},
        "model": {},
        "configuration": {
            "threads": THREADS, "context": CONTEXT, "n_gen": N_GEN,
            "bench_gen": BENCH_GEN, "bench_repeats": BENCH_REPEATS,
            "gpu_layers": 0, "prompt": PROMPT, "rtr": False,
            "native_routing_changes": False, "model_changes": False,
        },
        "hardware": hardware_snapshot(),
        "runs": [],
        "blockers": [],
    }
    try:
        result["build"] = setup_upstream()
        result["model"] = fetch_model()
        cli = Path(result["build"]["cli"])
        help_text = result["build"]["cli_help"]
        smoke = smoke_command(cli, help_text)

        cold_drop = drop_model_file_cache()
        cold = run_process("cold_smoke", smoke, cache_drop=cold_drop)
        warm = run_process("warm_smoke", smoke)
        result["runs"].extend([cold, warm])
        bench_path = result["build"].get("bench")
        result["bench"] = (
            run_bench(Path(bench_path), result["build"]["bench_help"])
            if bench_path
            else {"status": "unsupported", "reason": "llama-bench binary was not built"}
        )

        result["correctness"] = {
            "same_binary_seeded_smoke_response_equal": (
                cold["returncode"] == 0 and warm["returncode"] == 0
                and cold["response_nonempty"] and warm["response_nonempty"]
                and cold["response_sha256"] == warm["response_sha256"]
            ),
            "cold_returncode": cold["returncode"],
            "warm_returncode": warm["returncode"],
            "route_equality": "unmeasured_without_observation_hook",
            "cross_runtime_phase0_response_hash": "a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c",
            "cross_runtime_hash_interpretation": "reference_only; mismatch is not itself a route failure",
            "native_k": "parse_from_runtime_or_unverified",
            "no_drop_or_substitution": True,
        }
        if not result["correctness"]["same_binary_seeded_smoke_response_equal"]:
            result["blockers"].append("same-binary deterministic smoke did not pass")
        if cold["contains_nan_or_inf"] or warm["contains_nan_or_inf"]:
            result["blockers"].append("captured smoke logs contain nan/inf")
        if result["bench"].get("status") != "ok":
            result["blockers"].append("llama-bench resident decode probe unavailable or failed")
        source_probe = result["build"].get("source_probe", {})
        for key in ["qwen35moe", "iq2_xxs", "routed_gate", "routed_up", "routed_down"]:
            if not source_probe.get("matches", {}).get(key):
                result["blockers"].append(f"source anchor not found: {key}")
        result["status"] = "ok" if not result["blockers"] else "invalid_result"
    except Exception as exc:
        result["error"] = repr(exc)
        print(f"ERROR: {exc}", flush=True)
    result["finished_unix"] = time.time()
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    print(json.dumps({
        "status": result["status"],
        "blockers": result.get("blockers"),
        "smoke_equal": result.get("correctness", {}).get("same_binary_seeded_smoke_response_equal"),
        "bench_status": result.get("bench", {}).get("status"),
        "error": result.get("error"),
    }), flush=True)
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
