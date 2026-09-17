#!/usr/bin/env python3
"""Evaluate a persisted Falcon checkpoint and promote it only on a clean gate."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/kaggle/working/adtc-llm-limited-hardware")
WORK = Path("/kaggle/working/falcon-gate-persisted")
DATASET = os.environ.get("FALCON_CHECKPOINT_DATASET", "toheebogunade/jamii-afya-falcon-production-checkpoints")
sys.path.insert(0, str(ROOT))


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"timestamp_utc": now(), "event": event, **fields}), flush=True)


def run(command: list[str], cwd: Path = ROOT) -> None:
    emit("command_start", command=command)
    proc = subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{now()}] {line}", end="", flush=True)
    code = proc.wait()
    emit("command_end", exit_code=code)
    if code:
        raise RuntimeError(f"command failed: {' '.join(command)}")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    if not (ROOT / ".git").is_dir():
        run(["git", "clone", "--depth", "1", "--branch", "research/edge35-adaptive-streaming", "https://github.com/qeinstein/adtc-llm-limited-hardware.git", str(ROOT)], cwd=WORK)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-falcon-production.txt"])
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, capture_output=True, check=False).stdout.strip()
    if "P100" in gpu:
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "numpy<2"])
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu118"])
        run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao", "torchvision", "torchaudio", "bitsandbytes"])
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "--no-deps", "transformers==4.53.3", "tokenizers==0.21.4", "peft==0.15.2", "accelerate==1.7.0"])
    emit("environment", gpu=gpu, repo_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())

    config = json.loads((ROOT / "configs/falcon-production-v1.json").read_text())
    config["training"]["mcqa_context_max_tokens"] = 320
    config["data"]["max_length"] = 384
    config_path = WORK / "bounded-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    mcqa = ROOT / "output/accuracy_sft.jsonl"
    if mcqa.exists():
        mcqa.unlink()
    run([sys.executable, "-u", "scripts/build_accuracy_sft.py", "--datasets", "arc_easy", "arc_challenge", "openbookqa", "mmlu_aux", "medmcqa", "medqa", "pubmedqa", "headqa", "--max-per-dataset", "250", "--letter-permutations", "2", "--seed", "3407", "--fail-on-source-error", "--out", str(mcqa)])
    data_dir = WORK / "data"
    run([sys.executable, "-u", "scripts/build_falcon_dataset.py", "--config", str(config_path), "--out-dir", str(data_dir)])

    checkpoint_root = WORK / "checkpoint"
    run([sys.executable, "scripts/verify_persisted_checkpoint.py", "--dataset", DATASET, "--out-dir", str(checkpoint_root)])
    states = sorted(checkpoint_root.rglob("trainer_state.json"), key=lambda p: int(json.loads(p.read_text())["global_step"]))
    if not states:
        raise RuntimeError("no persisted trainer state")
    checkpoint = states[-1].parent
    step = int(json.loads(states[-1].read_text())["global_step"])
    emit("checkpoint_selected", checkpoint=str(checkpoint), step=step)

    eval_dir = WORK / "evaluation"
    run([sys.executable, "-u", "scripts/evaluate_falcon_hf.py", "--config", str(config_path), "--data-dir", str(data_dir), "--output-dir", str(eval_dir), "--adapter", str(checkpoint), "--max-dev", "8", "--max-new-tokens", "96", "--battery", "docs/research/falcon_prompt_dev.json", "--battery", "docs/research/falcon_prompt_validation.json", "--battery", "docs/research/falcon_probe_heldout.json"])
    reports = {}
    for label, battery in (("dev", "docs/research/falcon_prompt_dev.json"), ("validation", "docs/research/falcon_prompt_validation.json"), ("frozen", "docs/research/falcon_probe_heldout.json")):
        out = eval_dir / f"{label}-quality.json"
        run([sys.executable, "-u", "scripts/score_falcon_battery.py", "--battery", battery, "--generation-dir", str(eval_dir / Path(battery).stem), "--out", str(out), "--report-only"])
        reports[label] = json.loads(out.read_text())
    summary = {"gpu": gpu, "checkpoint": str(checkpoint), "step": step, "reports": reports}
    (WORK / "gate-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if any(reports[label].get("critical_failures") for label in ("dev", "validation", "frozen")) or any(float(reports[label].get("pass_rate_percent", 0)) < 75 for label in ("dev", "validation")) or float(reports["frozen"].get("pass_rate_percent", 0)) < 100:
        emit("promotion_rejected", **summary)
        return 2

    from scripts.verify_falcon_promotion import build_promotion_manifest, verify_promoted_adapter
    selection = {"status": "selected_and_frozen_gate_passed", "selected_checkpoint": str(checkpoint), "selected_step": step, "selected_eval_loss": None, "selection_criterion": "bounded-persisted-checkpoint-clean-dev-validation-and-frozen-gate", "minimum_pass_rate_percent": 75, "selected_dev_validation_pass_rate_percent": min(reports["dev"].get("pass_rate_percent", 0), reports["validation"].get("pass_rate_percent", 0)), "frozen_gate": {"status": "passed"}}
    frozen = reports["frozen"]
    manifest = build_promotion_manifest(checkpoint, quality_selection=selection, frozen_report={**frozen, "report_sha256": sha(eval_dir / "frozen-quality.json")}, experiment_id=config["experiment_id"], stage="stage_a_capability_preserving", repo_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), config_sha256=sha(config_path), data_manifest_sha256=sha(data_dir / "data_manifest.json"), minimum_pass_rate=100)
    (checkpoint / "promotion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checkpoint_manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    checkpoint_manifest["files"] = sorted(str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file())
    checkpoint_manifest["promotion_status"] = "promoted_after_frozen_gate"
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest, indent=2) + "\n")
    verify_promoted_adapter(checkpoint, minimum_pass_rate=100)
    run([sys.executable, "scripts/persist_checkpoint.py", "--checkpoint", str(checkpoint), "--dataset", DATASET, "--message", f"Falcon bounded stage-A step {step} frozen-gate promotion"])
    summary["status"] = "PROMOTED"
    (WORK / "gate-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit("promotion_complete", **summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
