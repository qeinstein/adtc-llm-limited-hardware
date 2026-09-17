"""Selective Q5_K -> Q4_K challenger for measured Qwen projection families.

The input IQ2_XXS GGUF is the control. Kaggle requantizes only tensors whose
names match the measured attention/GDN projection family; every other tensor
is copied at its original GGML type. This is a controlled representation
challenger, not an exact-runtime arm: the native router and K=8 graph remain
unchanged, but selected Q5_K weight values are requantized to Q4_K.
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


SCRATCH = Path("/tmp/native-sparse-selective-q4-v1")
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
BENCH = BUILD / "bin" / "llama-bench"
CLI = BUILD / "bin" / "llama-cli"
QUANTIZE = BUILD / "bin" / "llama-quantize"
MODEL = SCRATCH / "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
CHALLENGER = SCRATCH / "Qwen3.5-35B-A3B-selective-attn-Q4_K.gguf"
OUT = Path("/kaggle/working/native-sparse-selective-q4-v1-results")

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_SHA256 = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
RESEARCH_BASE_COMMIT = "ce5c119"
PROMPT = "Give one concise reason oral rehydration solution helps a child with watery diarrhoea."
THREADS = 4
N_GEN = 64
N_REPEATS = 3
EXPECTED_CONTROL_SHA256 = "a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c"
SELECTIVE_PATTERN = r"attn_(qkv|gate|q|k|v|output)\.weight=q4_k"

SUMMARY_RE = re.compile(
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


def run_checked(cmd: list[str], *, cwd: Path | None = None,
                log: Path | None = None, env: dict[str, str] | None = None,
                check: bool = True) -> subprocess.CompletedProcess:
    print("+", command_text(cmd), flush=True)
    process = subprocess.run([str(x) for x in cmd], cwd=cwd, env=env,
                             text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
    if log is not None:
        log.write_text(process.stdout, encoding="utf-8")
    if check and process.returncode:
        print(process.stdout[-4000:], flush=True)
        raise RuntimeError(f"command exited {process.returncode}: {command_text(cmd)}")
    return process


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"{path}: patch anchor count != 1")
    path.write_text(source.replace(old, new), encoding="utf-8")


def patch_quantizer() -> dict:
    path = LLAMA / "src" / "llama-quant.cpp"
    replace_once(path, "#include <cstring>\n", "#include <cstring>\n#include <cstdlib>\n")
    old = """        // if not manual - use the standard logic for choosing the quantization type based on the selected mixture
        if (!manual && !params->pure) {
"""
    new = """        // Controlled requantization arm: an explicit --tensor-type override
        // selects the challenger tensors, while all other quantized source
        // tensors retain their original GGML type byte-for-byte.
        if (!manual && std::getenv(\"JAMII_SELECTIVE_REQUANT_ONLY\") != nullptr) {
            return tensor->type;
        }

        // if not manual - use the standard logic for choosing the quantization type based on the selected mixture
        if (!manual && !params->pure) {
"""
    replace_once(path, old, new)
    replace_once(path,
        "        metadata[i].requires_imatrix = tensor_requires_imatrix(tensor->name, metadata[i].target_type, ftype);\n\n"
        "        if (params->imatrix) {",
        "        metadata[i].requires_imatrix = tensor_requires_imatrix(tensor->name, metadata[i].target_type, ftype);\n\n"
        "        // In the selective-only arm, an unmatched tensor is copied\n"
        "        // because its target type equals its source type.  Do not\n"
        "        // require an imatrix for a tensor that is not requantized.\n"
        "        if (std::getenv(\"JAMII_SELECTIVE_REQUANT_ONLY\") != nullptr &&\n"
        "                metadata[i].target_type == tensor->type) {\n"
        "            metadata[i].requires_imatrix = false;\n"
        "        }\n\n"
        "        if (params->imatrix) {")
    diff = subprocess.run(["git", "diff", "--", "src/llama-quant.cpp"], cwd=LLAMA,
                          text=True, stdout=subprocess.PIPE, check=True).stdout
    (OUT / "quantizer.patch").write_text(diff, encoding="utf-8")
    return {"description": "selective-only requantizer: unmatched tensors keep source type",
            "sha256": hashlib.sha256(diff.encode()).hexdigest()}


def build() -> dict:
    if not LLAMA.exists():
        run_checked(["git", "clone", "--filter=blob:none",
                     "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    source_head = run_checked(["git", "rev-parse", "HEAD"], cwd=LLAMA).stdout.strip()
    patch = patch_quantizer()
    configure = ["cmake", "-S", str(LLAMA), "-B", str(BUILD),
                 "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
                 "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF",
                 "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF"]
    run_checked(configure, log=OUT / "cmake-configure.log")
    build_cmd = ["cmake", "--build", str(BUILD), "--config", "Release", "-j4",
                 "--target", "llama-bench", "llama-cli", "llama-quantize"]
    run_checked(build_cmd, log=OUT / "cmake-build.log")
    version = run_checked([str(CLI), "--version"]).stdout
    (OUT / "llama-version.txt").write_text(version, encoding="utf-8")
    compiler = run_checked(["c++", "--version"]).stdout.splitlines()[0]
    return {"commit": LLAMA_COMMIT, "source_head": source_head, "patch": patch,
            "configure": configure, "build": build_cmd, "compiler": compiler}


def fetch_model() -> dict:
    if not MODEL.exists() or MODEL.stat().st_size != MODEL_SIZE:
        run_checked(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
                     "-C", "-", "-o", str(MODEL), MODEL_URL],
                    log=OUT / "model-download.log")
    size, digest = MODEL.stat().st_size, sha256(MODEL)
    if size != MODEL_SIZE or digest != MODEL_SHA256:
        raise RuntimeError(f"model identity mismatch: size={size} sha256={digest}")
    return {"repo": "unsloth/Qwen3.5-35B-A3B-GGUF", "repo_commit": MODEL_REPO_COMMIT,
            "file": MODEL_FILE, "size_bytes": size, "sha256": digest, "url": MODEL_URL}


def quantize_challenger() -> dict:
    if CHALLENGER.exists() and CHALLENGER.stat().st_size > 1_000_000_000:
        return {"reused": True, "size_bytes": CHALLENGER.stat().st_size,
                "sha256": sha256(CHALLENGER), "pattern": SELECTIVE_PATTERN}
    env = dict(os.environ, JAMII_SELECTIVE_REQUANT_ONLY="1")
    cmd = [str(QUANTIZE), "--allow-requantize", "--tensor-type", SELECTIVE_PATTERN,
           str(MODEL), str(CHALLENGER), "Q5_K_M", "4"]
    process = run_checked(cmd, env=env, log=OUT / "selective-quantize.log")
    if not CHALLENGER.exists():
        raise RuntimeError("selective quantizer returned without output file")
    result = {"reused": False, "command": cmd, "pattern": SELECTIVE_PATTERN,
              "size_bytes": CHALLENGER.stat().st_size, "sha256": sha256(CHALLENGER),
              "log_tail": process.stdout[-5000:]}
    if result["size_bytes"] < 1_000_000_000:
        raise RuntimeError(f"candidate output is implausibly small: {result['size_bytes']}")
    return result


def read_proc(pid: int) -> dict:
    result = {"mono_ns": time.monotonic_ns(), "rss_kib": 0, "anon_kib": 0,
              "file_kib": 0, "read_bytes": 0, "minor_faults": 0,
              "major_faults": 0, "valid": False}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"): result["rss_kib"] = int(line.split()[1])
            elif line.startswith("RssAnon:"): result["anon_kib"] = int(line.split()[1])
            elif line.startswith("RssFile:"): result["file_kib"] = int(line.split()[1])
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key == "read_bytes": result[key] = int(value)
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        result["minor_faults"], result["major_faults"] = int(fields[7]), int(fields[9])
        result["valid"] = True
    except (OSError, ValueError):
        pass
    return result


def response_text(stdout: str) -> str | None:
    marker = stdout.find("[Start thinking]")
    if marker < 0:
        return None
    text = stdout[marker:]
    match = SUMMARY_RE.search(text)
    if match:
        text = text[:match.start()]
    if text.rstrip().endswith("Exiting..."):
        text = text.rstrip()[:-len("Exiting...")]
    return text.strip() or None


def run_benchmark(label: str, model: Path) -> dict:
    repetitions = []
    for rep in range(1, N_REPEATS + 1):
        cmd = [str(CLI), "-m", str(model), "-ngl", "0", "-t", str(THREADS), "-c", "512",
               "-n", str(N_GEN), "--temp", "0", "--seed", "1234", "--single-turn",
               "--no-display-prompt", "--no-warmup", "--perf", "-lm", "mmap",
               "-lzm", "off", "--poll", "0", "--no-repack", "-p", PROMPT]
        print("+", command_text(cmd), flush=True)
        process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=dict(os.environ))
        samples = []
        while process.poll() is None:
            samples.append(read_proc(process.pid))
            time.sleep(0.02)
        stdout, stderr = process.communicate()
        (OUT / f"{label}_rep{rep}.stdout.txt").write_text(stdout, encoding="utf-8")
        (OUT / f"{label}_rep{rep}.stderr.txt").write_text(stderr, encoding="utf-8")
        matches = SUMMARY_RE.findall(stdout + "\n" + stderr)
        generation = float(matches[-1][1]) if matches else None
        valid = [x for x in samples if x["valid"]]
        repetitions.append({"rep": rep, "command": cmd, "returncode": process.returncode,
                            "prompt_tok_s": float(matches[-1][0]) if matches else None,
                            "generation_tok_s": generation,
                            "ms_per_token": 1000.0 / generation if generation else None,
                            "peak_rss_mib": max((x["rss_kib"] for x in valid), default=0) / 1024,
                            "peak_anon_mib": max((x["anon_kib"] for x in valid), default=0) / 1024,
                            "peak_file_mib": max((x["file_kib"] for x in valid), default=0) / 1024,
                            "read_bytes_delta": max(0, (samples[-1]["read_bytes"] - samples[0]["read_bytes"]) if samples else 0),
                            "minor_faults": max((x["minor_faults"] for x in valid), default=0),
                            "major_faults": max((x["major_faults"] for x in valid), default=0),
                            "stderr_tail": stderr[-1200:]})
    values = [x["generation_tok_s"] for x in repetitions if x["generation_tok_s"]]
    return {"label": label, "model": str(model), "repetitions": repetitions,
            "sample_tok_s": values, "mean_tok_s": statistics.mean(values) if values else None,
            "median_tok_s": statistics.median(values) if values else None,
            "min_tok_s": min(values) if values else None, "max_tok_s": max(values) if values else None,
            "ms_per_token": 1000.0 / statistics.mean(values) if values else None,
            "peak_rss_mib": max((x["peak_rss_mib"] for x in repetitions), default=0),
            "peak_anon_mib": max((x["peak_anon_mib"] for x in repetitions), default=0),
            "peak_file_mib": max((x["peak_file_mib"] for x in repetitions), default=0),
            "read_bytes_delta": sum(x["read_bytes_delta"] for x in repetitions),
            "returncode": max((x["returncode"] for x in repetitions), default=1)}


def run_smoke(label: str, model: Path) -> dict:
    cmd = [str(CLI), "-m", str(model), "-ngl", "0", "-t", str(THREADS), "-c", "512",
           "-n", "24", "--temp", "0", "--seed", "1234", "--single-turn",
           "--no-display-prompt", "--no-warmup", "--perf", "-lm", "mmap", "-lzm", "off",
           "--poll", "0", "--no-repack", "-p", PROMPT]
    process = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=False)
    (OUT / f"{label}.stdout.txt").write_text(process.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(process.stderr, encoding="utf-8")
    response = response_text(process.stdout)
    digest = hashlib.sha256(response.encode()).hexdigest() if response else None
    return {"label": label, "model": str(model), "command": cmd,
            "returncode": process.returncode, "response_sha256": digest,
            "response_chars": len(response) if response else 0,
            "matches_established_control": digest == EXPECTED_CONTROL_SHA256,
            "response_tail": response[-800:] if response else None}


def hardware() -> dict:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = []
    return {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "allowed_cpu_ids": affinity,
            "lscpu": subprocess.run(["lscpu"], text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, check=False).stdout}


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"schema": "native-sparse-selective-q4/v1", "status": "failed",
              "hypothesis": "Measured dominant Q5_K attention/GDN projections may benefit from a genuinely lower-bit Q4_K arithmetic path enough to move whole-runtime throughput.",
              "started_unix": time.time(), "research_source": {"base_commit": RESEARCH_BASE_COMMIT,
              "script_sha256": sha256(Path(__file__))},
              "challenger": {"pattern": SELECTIVE_PATTERN, "target_type": "Q4_K",
                             "unmatched_tensor_policy": "copy source GGML type",
                             "allow_requantize": True},
              "benchmark": {"n_prompt": 0, "n_gen": N_GEN, "repeats": N_REPEATS,
                             "threads": THREADS, "poll": 0, "load_mode": "mmap",
                             "lazy_mode": "off", "gpu_layers": 0,
                             "affinity": "inherited", "repack": False},
              "model": {}, "hardware": hardware(), "runs": [], "correctness": {},
              "quality": {"status": "initial deterministic smoke only"}}
    try:
        result["build"] = build()
        result["model"] = {"control": fetch_model(),
                            "challenger": quantize_challenger()}
        result["runs"] = [run_benchmark("control", MODEL),
                          run_benchmark("selective_q4", CHALLENGER)]
        result["correctness"] = {
            "control": run_smoke("control_smoke", MODEL),
            "selective_q4": run_smoke("selective_q4_smoke", CHALLENGER),
            "native_k": 8, "router_changed": False,
            "expert_weights_changed": False,
            "selected_projection_weights_requantized": True,
        }
        control, candidate = result["runs"]
        result["analysis"] = {
            "throughput_delta_fraction": candidate["mean_tok_s"] / control["mean_tok_s"] - 1.0,
            "storage_delta_fraction": result["model"]["challenger"]["size_bytes"] / MODEL_SIZE - 1.0,
            "candidate_output_hash_differs": result["correctness"]["selective_q4"]["response_sha256"] != EXPECTED_CONTROL_SHA256,
            "quality_gate_required_if_kept": True,
        }
        result["status"] = "ok" if all(x["returncode"] == 0 and x["mean_tok_s"] for x in result["runs"]) and \
            result["correctness"]["control"]["matches_established_control"] else "invalid_result"
    except Exception as exc:
        result["error"] = repr(exc)
        print(f"ERROR: {exc}", flush=True)
    result["finished_unix"] = time.time()
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "runs": [[x["label"], x.get("mean_tok_s")] for x in result.get("runs", [])],
                      "correctness": result.get("correctness"),
                      "analysis": result.get("analysis"), "error": result.get("error")}), flush=True)
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
