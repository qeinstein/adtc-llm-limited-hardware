#!/usr/bin/env python3
"""Audit a Jamii Afya training manifest by effective tokenizer tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "experiments" / "data" / "current_default.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "experiments" / "data" / "reports" / "current_default"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)

    from transformers import AutoTokenizer

    from src.data_audit import audit_manifest, write_outputs

    config = json.loads(manifest.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"],
        revision=config.get("tokenizer_revision"),
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    report = audit_manifest(manifest, tokenizer)
    output_dir = Path(args.output_dir)
    write_outputs(report, output_dir)
    print(f"Audit JSON: {output_dir / 'audit.json'}")
    print(f"Audit CSV : {output_dir / 'audit.csv'}")
    print(f"Audit MD  : {output_dir / 'audit.md'}")
    if any(item["required"] for item in report["missing_sources"]):
        print("WARNING: required inputs are missing; report is partial.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
