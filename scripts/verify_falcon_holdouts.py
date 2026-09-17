#!/usr/bin/env python3
"""Verify the committed Falcon holdout hashes and row counts.

The check is intentionally small and dependency-free so it can run before any
Kaggle model download or dataset construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path: Path) -> int:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("prompts") if isinstance(value, dict) else value
    return len(rows) if isinstance(rows, list) else -1


def verify(root: Path, manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"missing: {item['path']}")
            continue
        actual_hash = sha256(path)
        if actual_hash != item["sha256"]:
            errors.append(f"hash mismatch: {item['path']} expected={item['sha256']} actual={actual_hash}")
        actual_rows = row_count(path)
        if actual_rows != int(item["rows"]):
            errors.append(f"row-count mismatch: {item['path']} expected={item['rows']} actual={actual_rows}")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    manifest = Path(args.manifest).resolve() if args.manifest else root / "docs/research/falcon_holdout_manifest.json"
    errors = verify(root, manifest)
    if errors:
        for error in errors:
            print(f"[holdout] ERROR {error}")
        return 1
    print(f"[holdout] OK: {len(json.loads(manifest.read_text(encoding='utf-8'))['files'])} frozen files verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
