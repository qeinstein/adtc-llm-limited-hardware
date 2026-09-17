"""Resident Qwen3.5 dense Q5_K repack A/B measurement."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path


SCRATCH = Path("/tmp/native-sparse-dense-repack-v1")
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
BENCH = BUILD / "bin" / "llama-bench"
CLI = BUILD / "bin" / "llama-cli"
MODEL = SCRATCH / "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
OUT = Path("/kaggle/working/native-sparse-dense-repack-v1-results")

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_SHA256 = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
RESEARCH_BASE_COMMIT = "2c9e965"
PROMPT = "Give one concise reason oral rehydration solution helps a child with watery diarrhoea."
THREADS = 4
N_GEN = 64
N_REPEATS = 3
EXPECTED_CONTROL_SHA256 = "a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c"


def command_text(cmd: list[str]) -> str:
    return " ".join(str(x) for x in cmd)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(cmd: list[str], *, cwd: Path | None = None, log: Path | None = None,
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("+", command_text(cmd), flush=True)
    process = subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if log is not None:
        log.write_text(process.stdout + "\n--- STDERR ---\n" + process.stderr, encoding="utf-8")
    if process.returncode:
        print(process.stdout[-2000:], flush=True)
        print(process.stderr[-2000:], flush=True)
        raise RuntimeError(f"command exited {process.returncode}: {command_text(cmd)}")
    return process


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"{path}: patch anchor count != 1")
    path.write_text(source.replace(old, new), encoding="utf-8")


def patch_qwen_loader() -> dict:
    path = LLAMA / "src/models/qwen35moe.cpp"
    old = (
        '        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);\n'
        "        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);"
    )
    new = (
        "        // Exact native-sparse control loader: routed expert payloads\n"
        "        // remain mathematically unchanged; only their residency is lazy.\n"
        "        const int expert_flags = flags | TENSOR_READ_LAZY;\n"
        '        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, expert_flags);\n'
        "        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, expert_flags);"
    )
    replace_once(path, old, new)
    patch = subprocess.run(["git", "diff", "--", "src/models/qwen35moe.cpp"], cwd=LLAMA,
                           text=True, stdout=subprocess.PIPE, check=True).stdout
    (OUT / "runtime.patch").write_text(patch, encoding="utf-8")
    return {"description": "existing exact native-sparse routed-loader patch",
            "sha256": hashlib.sha256(patch.encode()).hexdigest()}


def build() -> dict:
    if not LLAMA.exists():
        run_checked(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    source_head = run_checked(["git", "rev-parse", "HEAD"], cwd=LLAMA).stdout.strip()
    patch = patch_qwen_loader()
    configure = ["cmake", "-S", str(LLAMA), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
                 "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF",
                 "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF"]
    run_checked(configure, log=OUT / "cmake-configure.log")
    build_cmd = ["cmake", "--build", str(BUILD), "--config", "Release", "-j4",
                 "--target", "llama-bench", "llama-cli"]
    run_checked(build_cmd, log=OUT / "cmake-build.log")
    version = run_checked([str(CLI), "--version"]).stdout
    (OUT / "llama-version.txt").write_text(version, encoding="utf-8")
    compiler = run_checked(["c++", "--version"]).stdout.splitlines()[0]
    return {"commit": LLAMA_COMMIT, "source_head": source_head, "patch": patch,
            "configure": configure, "build": build_cmd, "compiler": compiler}


def fetch_model() -> dict:
    if not MODEL.exists() or MODEL.stat().st_size != MODEL_SIZE:
        run_checked(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
                     "-C", "-", "-o", str(MODEL), MODEL_URL], log=OUT / "model-download.log")
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
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        result["minor_faults"], result["major_faults"] = int(tail[7]), int(tail[9])
        result["valid"] = True
    except (OSError, ValueError):
        result["valid"] = False
    return result


def run_benchmark(label: str, repack: bool) -> dict:
    cmd = [str(BENCH), "-m", str(MODEL), "-p", "0", "-n", str(N_GEN), "-r", str(N_REPEATS),
           "-t", str(THREADS), "-ngl", "0", "-lm", "mmap", "-lzm", "off", "--poll", "0",
           "--output", "json", "--repack" if repack else "--no-repack"]
    env = dict(os.environ)
    print("+", command_text(cmd), flush=True)
    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
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
    tok = row.get("samples_ts") if row else []
    return {"label": label, "repack": repack, "command": cmd, "returncode": process.returncode,
            "decode_row": row, "sample_tok_s": tok,
            "mean_tok_s": statistics.mean(tok) if tok else None,
            "median_tok_s": statistics.median(tok) if tok else None,
            "min_tok_s": min(tok) if tok else None, "max_tok_s": max(tok) if tok else None,
            "ms_per_token": 1000.0 / row["avg_ts"] if row and row.get("avg_ts") else None,
            "peak_rss_mib": max((x["rss_kib"] for x in valid), default=0) / 1024,
            "peak_anon_mib": max((x["anon_kib"] for x in valid), default=0) / 1024,
            "peak_file_mib": max((x["file_kib"] for x in valid), default=0) / 1024,
            "minor_faults": max((x["minor_faults"] for x in valid), default=0),
            "major_faults": max((x["major_faults"] for x in valid), default=0),
            "read_bytes_delta": max(0, (samples[-1]["read_bytes"] if samples else 0) -
                                     (samples[0]["read_bytes"] if samples else 0)),
            "stderr_tail": stderr[-3000:]}


def generated_response(stdout: str) -> str | None:
    marker = stdout.find("[Start thinking]")
    if marker < 0: return None
    text = stdout[marker:]
    marker2 = text.find("\nllama_perf_", 1)
    if marker2 >= 0: text = text[:marker2]
    return text.strip() or None


def run_correctness(label: str, repack: bool) -> dict:
    cmd = [str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(THREADS), "-c", "512",
           "-n", "24", "--temp", "0", "--seed", "1234", "--single-turn",
           "--no-display-prompt", "--no-warmup", "--perf", "-lm", "mmap", "-lzm", "off",
           "--poll", "0", "--repack" if repack else "--no-repack", "-p", PROMPT]
    process = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             check=False)
    (OUT / f"{label}.stdout.txt").write_text(process.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(process.stderr, encoding="utf-8")
    response = generated_response(process.stdout)
    digest = hashlib.sha256(response.encode()).hexdigest() if response else None
    return {"label": label, "repack": repack, "command": cmd, "returncode": process.returncode,
            "response_sha256": digest, "matches_established_hash": digest == EXPECTED_CONTROL_SHA256}


def hardware() -> dict:
    try: affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError): affinity = []
    return {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "allowed_cpu_ids": affinity,
            "lscpu": subprocess.run(["lscpu"], text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, check=False).stdout}


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"schema": "native-sparse-dense-repack/v1", "status": "failed",
              "hypothesis": "The standard CPU weight-repacking path improves Q5_K attention and Gated-DeltaNet projection decode on Qwen-shaped batch-1 workloads.",
              "started_unix": time.time(), "research_source": {"base_commit": RESEARCH_BASE_COMMIT,
              "script_sha256": sha256(Path(__file__))},
              "benchmark": {"n_prompt": 0, "n_gen": N_GEN, "repeats": N_REPEATS,
                             "threads": THREADS, "poll": 0, "load_mode": "mmap", "lazy_mode": "off",
                             "gpu_layers": 0, "affinity": "inherited"}, "hardware": hardware(), "runs": [],
              "correctness": {}, "limitations": [
                  "--repack is the generic backend donor arm, not a bespoke Qwen dense kernel.",
                  "The model remains the IQ2_XXS exact control; attention projection tensors are Q5_K."]}
    try:
        result["build"] = build()
        result["model"] = fetch_model()
        result["runs"] = [run_benchmark("dense_no_repack", False), run_benchmark("dense_repack", True)]
        result["correctness"] = {"no_repack": run_correctness("correctness_no_repack", False),
                                  "repack": run_correctness("correctness_repack", True),
                                  "native_k": 8, "router_or_weights_changed": False}
        result["status"] = "ok" if all(x["returncode"] == 0 and x["mean_tok_s"] for x in result["runs"]) and all(
            x["matches_established_hash"] for x in result["correctness"].values() if isinstance(x, dict) and "matches_established_hash" in x) else "invalid_result"
    except Exception as exc:
        result["error"] = repr(exc)
        print(f"ERROR: {exc}", flush=True)
    result["finished_unix"] = time.time()
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "runs": [[x["label"], x.get("mean_tok_s")] for x in result.get("runs", [])],
                      "correctness": result.get("correctness"), "error": result.get("error")}), flush=True)
    if result["status"] != "ok": raise SystemExit(2)


if __name__ == "__main__":
    main()
