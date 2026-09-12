#!/usr/bin/env python3
"""Download and validate a persisted Falcon Trainer checkpoint from Kaggle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=os.environ.get("FALCON_CHECKPOINT_DATASET"), required=False)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wait-seconds", type=int, default=600,
                    help="wait for an asynchronously-created Kaggle version to become visible")
    ap.add_argument("--retry-seconds", type=int, default=15)
    args = ap.parse_args(argv)
    if not args.dataset:
        raise SystemExit("FALCON_CHECKPOINT_DATASET/--dataset is required")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0, args.wait_seconds)
    last_reason = "unknown"
    with tempfile.TemporaryDirectory(prefix="falcon-checkpoint-download-") as tmp:
        tmp_path = Path(tmp)
        command = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(tmp_path)]
        attempt = 0
        while True:
            attempt += 1
            for old_archive in tmp_path.glob("*.zip"):
                old_archive.unlink()
            print(f"DOWNLOAD attempt={attempt}: {' '.join(command)}", flush=True)
            result = subprocess.run(command, check=False)
            archives = sorted(tmp_path.glob("*.zip"))
            if result.returncode == 0 and archives:
                with zipfile.ZipFile(archives[-1]) as archive:
                    names = archive.namelist()
                    if any(name.endswith("trainer_state.json") for name in names):
                        archive.extractall(out)
                        break
                    last_reason = "archive is visible but has no trainer_state.json yet"
            elif result.returncode:
                last_reason = f"download exited {result.returncode}"
            else:
                last_reason = f"no archive found in {tmp_path}"
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                raise SystemExit(f"Kaggle checkpoint did not become visible within wait window: {last_reason}")
            delay = min(max(1, args.retry_seconds), remaining)
            print(f"PERSISTENCE_WAIT reason={last_reason} retry_in={delay}s remaining={remaining}s", flush=True)
            time.sleep(delay)
    trainer_states = sorted(out.rglob("trainer_state.json"))
    if not trainer_states:
        raise SystemExit("downloaded dataset has no trainer_state.json; persistence is not resumable")
    checkpoints = []
    for state_path in trainer_states:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        checkpoint = state_path.parent
        required = ["optimizer.pt", "scheduler.pt", "rng_state.pth"]
        missing = [name for name in required if not (checkpoint / name).is_file()]
        adapter_files = [path for path in checkpoint.glob("adapter_model.*") if path.is_file()]
        manifest_path = checkpoint / "checkpoint_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        if manifest.get("scaler_required") is True and not (checkpoint / "scaler.pt").is_file():
            missing.append("scaler.pt")
        if missing or not adapter_files or manifest.get("complete") is not True:
            raise SystemExit(f"incomplete persisted checkpoint: {checkpoint}; missing={missing}")
        checkpoints.append({"path": str(state_path), "global_step": state.get("global_step"), "files": sorted(x.name for x in checkpoint.iterdir() if x.is_file())})
    summary = {"dataset": args.dataset, "checkpoint_count": len(checkpoints), "checkpoints": checkpoints}
    (out / "persistence-verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
