#!/usr/bin/env python3
"""Run one bounded Falcon quality probe entirely on a Kaggle worker.

This is deliberately not the production stage: it uses an ephemeral local
checkpoint, 16 optimizer steps, a bounded public-train MCQA replay, and only
development/validation generation batteries.  The purpose is to compare a
lower-rate attention+Mamba adapter against the rejected all-module follow-up
without spending a production-scale run.
"""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


WORK = Path("/kaggle/working")
REPO = WORK / "adtc-llm-limited-hardware"
RUN_ID = os.environ.get("FALCON_MICRO_PROBE_ID", time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
OUT = WORK / "falcon-micro-probe-v2" / RUN_ID
CONFIG = REPO / "configs" / "falcon-production-v1.json"
DATA_DIR = OUT / "data"


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short(command: list[str], cwd: Path = REPO) -> str:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True).stdout.strip()


def run_stream(command: list[str], name: str, cwd: Path = REPO, extra_env: dict[str, str] | None = None) -> None:
    """Run a child with line streaming and a periodic heartbeat."""
    log_path = OUT / "logs" / name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(extra_env or {})
    env["PYTHONUNBUFFERED"] = "1"
    rendered_command = " ".join(str(part) for part in command)
    print(f"[{stamp()}] STREAM {rendered_command}", flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        proc = subprocess.Popen(
            [str(part) for part in command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        started = time.monotonic()
        try:
            while True:
                ready = selector.select(timeout=30)
                if ready:
                    line = proc.stdout.readline()
                    if line:
                        output = f"[{stamp()}] {line}"
                        print(output, end="", flush=True)
                        handle.write(output)
                        handle.flush()
                    elif proc.poll() is not None:
                        break
                else:
                    heartbeat = f"[{stamp()}] HEARTBEAT child={name} elapsed={time.monotonic() - started:.1f}s\n"
                    print(heartbeat, end="", flush=True)
                    handle.write(heartbeat)
                    handle.flush()
                if proc.poll() is not None:
                    for line in proc.stdout:
                        output = f"[{stamp()}] {line}"
                        print(output, end="", flush=True)
                        handle.write(output)
                        handle.flush()
                    break
        finally:
            selector.close()
        code = proc.wait()
        ending = f"[{stamp()}] EXIT={code}\n"
        print(ending, end="", flush=True)
        handle.write(ending)
        handle.flush()
    if code:
        raise RuntimeError(f"child failed with exit {code}: {rendered_command}")


def install_worker_stack() -> str:
    run_stream([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-falcon-production.txt"], "pip-base.log")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    if "P100" in gpu:
        run_stream([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "numpy<2"], "numpy-p100.log")
        run_stream(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu118"],
            "torch-p100.log",
        )
        run_stream([sys.executable, "-m", "pip", "uninstall", "-y", "torchao", "torchvision", "torchaudio", "bitsandbytes"], "optional-uninstall.log")
        run_stream(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "--no-deps", "transformers==4.53.3", "tokenizers==0.21.4", "peft==0.15.2", "accelerate==1.7.0"],
            "hf-stack-final.log",
        )
    else:
        run_stream([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], "bitsandbytes.log")
    return gpu


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def changed_generation_count(stock_dir: Path, adapter_dir: Path, battery_stems: list[str]) -> int:
    changed = 0
    for stem in battery_stems:
        stock_index = load_json(stock_dir / stem / "_index.json")
        adapter_index = load_json(adapter_dir / stem / "_index.json")
        adapter_by_id = {str(item["id"]): item for item in adapter_index}
        for item in stock_index:
            prompt_id = str(item["id"])
            stock_text = (stock_dir / stem / f"{prompt_id}.txt").read_text(encoding="utf-8")
            adapter_text = (adapter_dir / stem / f"{prompt_id}.txt").read_text(encoding="utf-8")
            if stock_text != adapter_text or item.get("new_tokens") != adapter_by_id[prompt_id].get("new_tokens"):
                changed += 1
    return changed


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = stamp()
    if not REPO.is_dir():
        run_stream(["git", "clone", "--depth", "1", "--branch", "research/edge35-adaptive-streaming", "https://github.com/qeinstein/adtc-llm-limited-hardware.git", str(REPO)], name="clone.log", cwd=WORK)
    repo_sha = short(["git", "rev-parse", "HEAD"])
    gpu = install_worker_stack()
    write_json(OUT / "environment.json", {
        "timestamp_utc": stamp(),
        "gpu": gpu,
        "repo_sha": repo_sha,
        "python": sys.version,
        "torch": short([sys.executable, "-c", "import torch; print(torch.__version__)"]),
        "cuda": short([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"]),
        "cuda_capability": short([sys.executable, "-c", "import torch; print(torch.cuda.get_device_capability(0))"]),
    })

    mcqa = REPO / "output" / "accuracy_sft.jsonl"
    run_stream(
        [sys.executable, "-u", "scripts/build_accuracy_sft.py", "--datasets", "arc_easy", "arc_challenge", "openbookqa", "mmlu_aux", "medmcqa", "medqa", "pubmedqa", "headqa", "--max-per-dataset", "32", "--letter-permutations", "2", "--seed", "3407", "--fail-on-source-error", "--out", str(mcqa)],
        "mcqa-build.log",
    )
    run_stream(
        [sys.executable, "-u", "scripts/build_falcon_dataset.py", "--config", str(CONFIG), "--out-dir", str(DATA_DIR)],
        "dataset-build.log",
    )
    manifest = load_json(DATA_DIR / "data_manifest.json")
    write_json(OUT / "data_manifest_summary.json", {
        "counts": manifest["counts"],
        "token_totals": manifest["token_totals"],
        "loss_token_totals": manifest["loss_token_totals"],
        "loss_token_shares_percent": manifest["loss_token_shares_percent"],
        "facets": manifest["facets"],
        "source_manifest_sha256": sha256_file(DATA_DIR / "data_manifest.json"),
    })

    training_dir = OUT / "training"
    micro_steps = os.environ.get("FALCON_MICRO_STEPS", "16")
    # This candidate must be independently launchable.  Stage B requires a
    # persisted Stage-A adapter, which this bounded screen intentionally does
    # not assume.  Safety weighting is carried by the data mixture and the
    # explicit low-LR/all-target ablation below.
    micro_stage = os.environ.get("FALCON_MICRO_STAGE", "stage_a_capability_preserving")
    micro_lr = os.environ.get("FALCON_MICRO_LR", "0.00002")
    micro_rank = os.environ.get("FALCON_MICRO_LORA_R", "4")
    micro_max_length = os.environ.get("FALCON_MICRO_MAX_LENGTH", "384")
    micro_targets = os.environ.get(
        "FALCON_MICRO_TARGETS",
        "q_proj,k_proj,v_proj,o_proj,in_proj,out_proj,gate_proj,up_proj,down_proj",
    )
    run_stream(
        [
            sys.executable, "-u", "scripts/train_falcon_production.py",
            "--config", str(CONFIG), "--data-dir", str(DATA_DIR), "--run-dir", str(training_dir),
            "--stage", micro_stage, "--max-steps", micro_steps, "--save-steps", "4", "--eval-steps", "4",
            "--learning-rate", micro_lr, "--lora-r", micro_rank,
            "--lora-alpha", str(int(micro_rank) * 2), "--lora-dropout", "0.05",
            "--target-modules", micro_targets,
            "--max-length", micro_max_length,
            "--quantize", "none", "--compute-dtype", "fp16", "--allow-ephemeral",
        ],
        "training.log",
    )
    adapter = training_dir / "stage_a_capability_preserving" / "checkpoints" / "final-adapter"
    if not adapter.is_dir():
        raise RuntimeError(f"training produced no final adapter: {adapter}")

    batteries = [item.strip() for item in os.environ.get(
        "FALCON_MICRO_BATTERIES",
        "docs/research/falcon_prompt_dev.json,docs/research/falcon_prompt_validation.json,docs/research/falcon_probe_heldout.json",
    ).split(",") if item.strip()]
    stems = [Path(item).stem for item in batteries]
    stock_dir = OUT / "stock-eval"
    adapter_dir = OUT / "adapter-eval"
    for label, destination, extra in (
        ("stock", stock_dir, []),
        ("adapter", adapter_dir, ["--adapter", str(adapter)]),
    ):
        command = [sys.executable, "-u", "scripts/evaluate_falcon_hf.py", "--config", str(CONFIG), "--data-dir", str(DATA_DIR), "--output-dir", str(destination), "--max-dev", "64", "--max-new-tokens", "128"]
        for battery in batteries:
            command += ["--battery", battery]
        command += extra
        run_stream(command, f"{label}-eval.log")
        for battery, stem in zip(batteries, stems):
            run_stream(
                [sys.executable, "-u", "scripts/score_falcon_battery.py", "--battery", battery, "--generation-dir", str(destination / stem), "--out", str(destination / f"{stem}-quality.json"), "--report-only"],
                f"{label}-{stem}-quality.log",
            )

    stock_metrics = load_json(stock_dir / "eval_metrics.json")
    adapter_metrics = load_json(adapter_dir / "eval_metrics.json")
    quality = {}
    for stem in stems:
        quality[stem] = {
            "stock": load_json(stock_dir / f"{stem}-quality.json"),
            "adapter": load_json(adapter_dir / f"{stem}-quality.json"),
        }
    summary = {
        "schema_version": "1.0.0",
            "experiment_id": os.environ.get("FALCON_MICRO_PROBE_ID", "falcon-sprint-safety-candidate"),
        "status": "COMPLETE",
        "decision": "FOLLOW-UP",
        "started_utc": started,
        "completed_utc": stamp(),
        "repo_sha": repo_sha,
        "gpu": gpu,
        "model": load_json(CONFIG)["model"],
            "hypothesis": "A safety-weighted all-module Falcon adapter at the bounded learning rate should improve refusal, uncertainty, and urgent-disposition behavior without the previous MCQA-only specialization failure.",
        "training": {
                "steps": int(micro_steps),
                "learning_rate": float(micro_lr),
                "max_length": int(micro_max_length),
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
                "target_modules": [item for item in micro_targets.split(",") if item],
            "persistence": "bounded probe used --allow-ephemeral; no production checkpoint promotion",
        },
        "data": {
            "mcqa_max_per_dataset": 32,
            "letter_permutations": 2,
            "manifest_summary": manifest["counts"],
            "loss_token_shares_percent": manifest["loss_token_shares_percent"],
        },
        "evaluation": {
            "batteries": batteries,
            "frozen_final_holdout_touched": False,
            "stock": stock_metrics,
            "adapter": adapter_metrics,
            "quality": quality,
            "changed_generation_count": changed_generation_count(stock_dir, adapter_dir, stems),
        },
        "conclusion": "No promotion decision is made by this bounded probe; inspect safety and usefulness reports before authorizing any longer run.",
    }
    write_json(OUT / "micro-probe-summary.json", summary)
    print(f"[{stamp()}] MICRO_PROBE_COMPLETE {json.dumps({'summary': str(OUT / 'micro-probe-summary.json'), 'changed_generation_count': summary['evaluation']['changed_generation_count']}, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
