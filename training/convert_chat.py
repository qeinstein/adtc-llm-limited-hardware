#!/usr/bin/env python3
"""Convert SFT mixtures to Axolotl chat_template JSONL. Deterministic.

Reads data/manifests/{pilot,full}_mixture.jsonl + prompts/system.json and
writes data/chat/{pilot,full}_chat.jsonl with rows:
  {"messages": [{"role": "system", ...}, {"role": "user", ...},
                {"role": "assistant", ...}], "src": ..., "src_id": ...}

Refuses to convert when:
- the system prompt version is not the pinned one (train/serve skew guard)
- a row's prompt/response is empty
- a row contains chat-template control strings (<think>, im_start)

Usage:
  python3 training/convert_chat.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINNED_SYSTEM_VERSION = "1.0.0"
FORBIDDEN = ("<think>", "im_start", "im_end", "<tool_call>")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    sys_prompt = json.loads((ROOT / "prompts" / "system.json").read_text())
    if sys_prompt.get("version") != PINNED_SYSTEM_VERSION:
        raise SystemExit(
            f"system prompt v{sys_prompt.get('version')} != pinned "
            f"v{PINNED_SYSTEM_VERSION}; refusing (train/serve skew)")
    system_text = sys_prompt["text"]
    out_dir = ROOT / "data" / "chat"
    out_dir.mkdir(exist_ok=True)
    manifest = {"system_version": PINNED_SYSTEM_VERSION,
                "system_sha256": hashlib.sha256(
                    system_text.encode()).hexdigest(),
                "files": {}}
    for name in ("pilot", "full"):
        src = ROOT / "data" / "manifests" / f"{name}_mixture.jsonl"
        dst = out_dir / f"{name}_chat.jsonl"
        n = 0
        with open(src, encoding="utf-8") as fin, open(dst, "w",
                                                      encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                r = json.loads(line)
                prompt, response = r["prompt"], r["response"]
                if not prompt.strip() or not response.strip():
                    raise SystemExit(f"{name}: empty prompt/response "
                                     f"(src_id={r.get('src_id')})")
                blob = prompt + response
                for bad in FORBIDDEN:
                    if bad in blob:
                        raise SystemExit(f"{name}: forbidden {bad!r} "
                                         f"(src_id={r.get('src_id')})")
                row = {"messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}],
                    "src": r["src"], "src_id": r["src_id"]}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        manifest["files"][dst.name] = {"rows": n,
                                       "sha256": sha256_file(dst)}
        print(f"{dst.name}: {n} rows", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", out_dir / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
