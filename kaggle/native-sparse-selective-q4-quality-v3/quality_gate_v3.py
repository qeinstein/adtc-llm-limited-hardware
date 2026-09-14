"""Likelihood-based broad quality gate for the selective-Q4 challenger."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-selective-q4-quality-v3")
OUT = WORK / "native-sparse-selective-q4-quality-v3-results"
BASE = SCRATCH / "quality_gate_v2_base.py"
RESEARCH_COMMIT = "cae43e5fb930ec562bf5749c7df5cb0d8827a911"
BASE_URL = (
    "https://raw.githubusercontent.com/qeinstein/adtc-llm-limited-hardware/"
    f"{RESEARCH_COMMIT}/kaggle/native-sparse-selective-q4-quality-v2/quality_gate_v2.py"
)
DATA_REVISION = "37884b81b4957f1950a53b6ff48d77c8dd5e430c"
DATA_URL = (
    "https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/"
    f"resolve/{DATA_REVISION}/mmlu-test.bin"
)
N_TASKS = 100

SCRATCH.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base():
    urllib.request.urlretrieve(BASE_URL, BASE)
    spec = importlib.util.spec_from_file_location("quality_v2", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load quality-v2 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_eval(binary: Path, model: Path, dataset: Path, label: str) -> dict:
    cmd = [
        str(binary), "-m", str(model), "-f", str(dataset), "-ngl", "0",
        "-t", "4", "-c", "2048", "-b", "2048", "-ub", "512", "-np", "16",
        "--multiple-choice", "--multiple-choice-tasks", str(N_TASKS),
        "--no-repack", "--no-warmup",
    ]
    print("+", " ".join(cmd), flush=True)
    started = time.monotonic()
    process = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=False)
    elapsed = time.monotonic() - started
    (OUT / f"{label}.stdout.txt").write_text(process.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(process.stderr, encoding="utf-8")
    text = process.stdout + "\n" + process.stderr
    finals = re.findall(r"Final result:\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", text)
    rows = re.findall(r"^\s*(\d+)\s+([0-9.]+)\s*$", text, flags=re.MULTILINE)
    if process.returncode or not finals:
        raise RuntimeError(f"{label} evaluation failed rc={process.returncode}: {text[-3000:]}")
    score, sigma = finals[-1]
    return {
        "label": label,
        "command": cmd,
        "returncode": process.returncode,
        "elapsed_sec": elapsed,
        "score_percent": float(score),
        "sigma_percent": float(sigma),
        "running_accuracy": [[int(n), float(value)] for n, value in rows],
        "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
    }


def main() -> None:
    started = time.time()
    base = load_base()
    base.SCRATCH = SCRATCH
    base.OUT = OUT
    base.LLAMA = SCRATCH / "llama.cpp"
    base.BUILD = base.LLAMA / "build-native"
    base.CLI = base.BUILD / "bin" / "llama-cli"
    base.QUANTIZE = base.BUILD / "bin" / "llama-quantize"
    base.MODEL = SCRATCH / base.MODEL_FILE
    base.CHALLENGER = SCRATCH / "Qwen3.5-35B-A3B-selective-attn-Q4_K.gguf"
    perplexity = base.BUILD / "bin" / "llama-perplexity"
    dataset = SCRATCH / "mmlu-test.bin"

    build_info = base.build()
    subprocess.run([
        "cmake", "--build", str(base.BUILD), "--config", "Release", "-j4",
        "--target", "llama-perplexity",
    ], check=True)
    model_info = base.fetch_model()
    challenger_info = base.quantize_challenger()
    urllib.request.urlretrieve(DATA_URL, dataset)
    dataset_info = {"url": DATA_URL, "revision": DATA_REVISION,
                    "size_bytes": dataset.stat().st_size, "sha256": sha256(dataset)}

    control = run_eval(perplexity, base.MODEL, dataset, "control")
    candidate = run_eval(perplexity, base.CHALLENGER, dataset, "selective_q4")
    delta = candidate["score_percent"] - control["score_percent"]
    decision = "KEEP" if delta >= -2.0 else "REJECT_QUALITY_REGRESSION"
    result = {
        "schema": "native-sparse-selective-q4-quality/v3",
        "status": "complete",
        "hypothesis": "Selective attention/GDN Q4_K preserves matched broad multiple-choice likelihood accuracy within two percentage points.",
        "research_source_commit": RESEARCH_COMMIT,
        "runtime": {"llama_commit": base.LLAMA_COMMIT, "threads": 4,
                    "context": 2048, "batch": 2048, "ubatch": 512,
                    "parallel": 16, "repack": False},
        "build": build_info,
        "model": model_info,
        "challenger": challenger_info,
        "dataset": dataset_info,
        "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "tasks": N_TASKS,
        "control": control,
        "selective_q4": candidate,
        "score_delta_percentage_points": delta,
        "decision": decision,
        "decision_rule": "reject if matched score drops by more than 2.0 percentage points",
        "wall_sec": time.time() - started,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(__file__, OUT / Path(__file__).name)
    print(json.dumps({"control": control["score_percent"],
                      "selective_q4": candidate["score_percent"],
                      "delta_pp": delta, "decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
