#!/usr/bin/env python3
"""Persist a small Falcon adapter/checkpoint through Kaggle Dataset versioning.

The production notebook must set ``FALCON_CHECKPOINT_DATASET`` to an existing
private dataset slug. This script refuses to guess credentials or create a
remote destination implicitly. The uploaded directory is deliberately limited
to adapter/trainer state and manifests, never the base model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_streamed(command: list[str]) -> int:
    """Stream remote-upload output; never hide a slow Kaggle API call."""
    print("STREAM " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{now()}] {line}", end="", flush=True)
    return process.wait()


def wait_for_dataset_files(dataset: str, timeout_seconds: int = 600,
                           retry_seconds: int = 15) -> None:
    """Wait until the newly-created version exposes resumable trainer state."""
    deadline = time.monotonic() + timeout_seconds
    command = ["kaggle", "datasets", "files", "-d", dataset]
    attempt = 0
    while True:
        attempt += 1
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        listing = (result.stdout or "") + (result.stderr or "")
        ready = result.returncode == 0 and all(
            name in listing for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        )
        if ready:
            print(f"PERSISTENCE_READY dataset={dataset} attempt={attempt}", flush=True)
            return
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise RuntimeError(
                f"Kaggle dataset version did not expose resumable files within {timeout_seconds}s; "
                f"last listing: {listing[-1000:]}"
            )
        delay = min(max(1, retry_seconds), remaining)
        print(f"PERSISTENCE_WAIT dataset={dataset} attempt={attempt} retry_in={delay}s", flush=True)
        time.sleep(delay)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default=os.environ.get("FALCON_CHECKPOINT_DATASET"))
    ap.add_argument("--message", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait-seconds", type=int, default=600)
    ap.add_argument("--retry-seconds", type=int, default=15)
    args = ap.parse_args(argv)
    if not args.dataset:
        raise SystemExit("FALCON_CHECKPOINT_DATASET/--dataset is required; refusing ephemeral-only persistence")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint directory does not exist: {checkpoint}")
    if not (checkpoint / "trainer_state.json").exists():
        raise SystemExit(f"checkpoint has no trainer_state.json: {checkpoint}")
    try:
        trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"checkpoint trainer_state.json is invalid: {exc}") from exc
    if not isinstance(trainer_state.get("global_step"), int):
        raise SystemExit("checkpoint trainer_state.json has no integer global_step")
    adapter_files = [path for path in checkpoint.glob("adapter_model.*") if path.is_file()]
    required_state = [checkpoint / "optimizer.pt", checkpoint / "scheduler.pt", checkpoint / "rng_state.pth"]
    if not adapter_files:
        raise SystemExit(f"checkpoint has no adapter_model.* file: {checkpoint}")
    missing_state = [str(path.name) for path in required_state if not path.exists()]
    if missing_state:
        raise SystemExit(f"checkpoint is not resumable; missing: {', '.join(missing_state)}")
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"checkpoint_manifest.json is invalid: {exc}") from exc
        if manifest.get("complete") is not True or int(manifest.get("global_step", -1)) != trainer_state["global_step"]:
            raise SystemExit("checkpoint manifest is not marked complete or disagrees with trainer_state global_step")
        if manifest.get("scaler_required") is True and not (checkpoint / "scaler.pt").is_file():
            raise SystemExit("checkpoint manifest requires scaler.pt but it is missing")
    with tempfile.TemporaryDirectory(prefix="falcon-persist-") as tmp:
        staging = Path(tmp) / "checkpoint"
        shutil.copytree(checkpoint, staging)
        metadata = {
            "title": "Jamii Afya Falcon production checkpoints",
            "id": args.dataset,
            "licenses": [{"name": "other"}],
            "subtitle": "Private resumable adapter and trainer state; base model is not stored here."
        }
        (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        command = ["kaggle", "datasets", "version", "-p", str(staging), "-m", args.message, "--dir-mode", "zip"]
        print("PERSIST:", " ".join(command), flush=True)
        if args.dry_run:
            return 0
        code = run_streamed(command)
        if code:
            return code
        wait_for_dataset_files(args.dataset, args.wait_seconds, args.retry_seconds)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
