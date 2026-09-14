#!/usr/bin/env python3
"""Finalize the first frozen-gate-passed Falcon adapter on Kaggle.

This worker is intentionally fail-closed: it will not upload an adapter unless
the persisted promotion manifest, merged evaluation, and exact Q4_K_M gate all
pass.  All large files live under /kaggle/working.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path("/kaggle/working/adtc-llm-limited-hardware")
WORK = Path("/kaggle/working/falcon-finalize")
CHECKPOINT_DATASET = os.environ.get(
    "FALCON_CHECKPOINT_DATASET",
    "toheebogunade/jamii-afya-falcon-production-checkpoints",
)
HOST_DATASET = os.environ.get(
    "FALCON_HOST_DATASET", "toheebogunade/jamii-afya-falcon-submission"
)


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"timestamp_utc": stamp(), "event": event, **fields}), flush=True)


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    emit("command_start", command=command)
    merged = os.environ.copy()
    merged.update(env or {})
    merged["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{stamp()}] {line}", end="", flush=True)
    code = proc.wait()
    emit("command_end", exit_code=code)
    if code:
        raise RuntimeError(f"command failed ({code}): {' '.join(command)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_promoted_adapter(root: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for manifest_path in root.rglob("promotion_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "selected_and_frozen_gate_passed":
            continue
        adapter = manifest_path.parent
        if not (adapter / "adapter_config.json").is_file():
            continue
        if not list(adapter.glob("adapter_model.*")):
            continue
        candidates.append((int(manifest.get("selected_step", -1)), adapter))
    if not candidates:
        raise RuntimeError("no frozen-gate-passed promotion manifest in persisted checkpoint dataset")
    return max(candidates, key=lambda item: item[0])[1]


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    if not (ROOT / ".git").is_dir():
        run(["git", "clone", "--depth", "1", "--branch", "research/edge35-adaptive-streaming", "https://github.com/qeinstein/adtc-llm-limited-hardware.git", str(ROOT)], cwd=WORK)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-falcon-production.txt"], cwd=ROOT)
    checkpoint_root = WORK / "checkpoint"
    if checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)
    run([sys.executable, "scripts/verify_persisted_checkpoint.py", "--dataset", CHECKPOINT_DATASET, "--out-dir", str(checkpoint_root)], cwd=ROOT)
    adapter = find_promoted_adapter(checkpoint_root)
    emit("promoted_adapter_selected", adapter=str(adapter))

    export_dir = WORK / "export"
    run(["bash", "scripts/export_falcon_gguf.sh", str(adapter), str(export_dir)], cwd=ROOT)
    export_manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
    deployment = Path(export_manifest["deployment_model"])
    if not deployment.is_file():
        raise RuntimeError(f"missing deployment model: {deployment}")
    emit("export_ready", path=str(deployment), bytes=deployment.stat().st_size, sha256=sha256(deployment))

    # The exact quantized evaluator is intentionally installed only on Kaggle.
    run([sys.executable, "-m", "pip", "install", "-q", "--prefer-binary", "llama-cpp-python==0.3.16"], cwd=ROOT)
    quant_eval = export_dir / "quantized-eval"
    run([
        sys.executable, "scripts/evaluate_falcon_candidate.py",
        "--model", str(deployment), "--out-dir", str(quant_eval),
        "--limit", os.environ.get("FALCON_FINAL_MCQA_LIMIT", "100"),
        "--n-ctx", "2048", "--threads", os.environ.get("FALCON_EVAL_THREADS", "4"),
        "--generation-mode", "chat",
        "--system-prompt", "You are Jamii Afya, an offline medical decision-support assistant for community health workers in rural African clinics. Answer in the question's language. Surface danger signs and when to refer.",
        "--battery", "docs/research/falcon_probe_heldout.json",
    ], cwd=ROOT)
    frozen_quality = quant_eval / "frozen-quality.json"
    run([
        sys.executable, "scripts/score_falcon_battery.py",
        "--battery", "docs/research/falcon_probe_heldout.json",
        "--generation-dir", str(quant_eval / "falcon_probe_heldout"),
        "--out", str(frozen_quality), "--report-only",
    ], cwd=ROOT)
    quality = json.loads(frozen_quality.read_text(encoding="utf-8"))
    if float(quality.get("pass_rate_percent", 0.0)) < 100.0 or quality.get("critical_failures"):
        raise RuntimeError(f"exact Q4_K_M frozen gate failed: {quality}")

    bench_dir = export_dir / "deployment-benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(1, 4):
        out = bench_dir / f"rep-{rep}.json"
        timing = bench_dir / f"rep-{rep}.time"
        command = ["/usr/bin/time", "-v", "./llama.cpp/build/bin/llama-bench", "-m", str(deployment), "-p", "512", "-n", "128", "-ngl", "0", "--output", "json"]
        with out.open("w", encoding="utf-8") as stdout, timing.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"llama-bench failed on repetition {rep}")
        rows.append(json.loads(out.read_text(encoding="utf-8")))
    benchmark = {"model": str(deployment), "bytes": deployment.stat().st_size, "sha256": sha256(deployment), "repetitions": rows, "llama_cpp_revision": export_manifest.get("llama_cpp_revision"), "command": "llama-bench -p 512 -n 128 -ngl 0 --output json"}
    (bench_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    emit("benchmark_ready", benchmark=str(bench_dir / "benchmark.json"))

    submission = WORK / "submission"
    if submission.exists():
        shutil.rmtree(submission)
    submission.mkdir(parents=True)
    model_name = "Falcon-H1-1.5B-Deep-Instruct-Q4_K_M.gguf"
    shutil.copy2(deployment, submission / model_name)
    (submission / "dataset-metadata.json").write_text(json.dumps({
        "title": "Jamii Afya Falcon submission artifact",
        "id": HOST_DATASET,
        "subtitle": "Public Falcon-H1 deployment GGUF for ADTC submission",
        "description": "Frozen-gate-passed Jamii Afya Falcon-H1-1.5B-Deep-Instruct Q4_K_M deployment artifact.",
        "isPrivate": False,
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n", encoding="utf-8")
    run(["kaggle", "datasets", "version", "-p", str(submission), "-m", "publish frozen-gate-passed Falcon Q4_K_M", "--dir-mode", "zip"], cwd=ROOT)
    emit("hosting_uploaded", dataset=HOST_DATASET, file=model_name)
    summary = {"status": "SUBMISSION_ARTIFACT_READY", "deployment_model": str(deployment), "deployment_bytes": deployment.stat().st_size, "deployment_sha256": sha256(deployment), "quality": quality, "benchmark": benchmark, "host_dataset": HOST_DATASET, "host_file": model_name}
    (WORK / "final-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
