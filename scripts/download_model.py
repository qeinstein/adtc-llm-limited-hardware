#!/usr/bin/env python3
"""Cross-platform, hash-verified downloader for the shipped GGUF artifact."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
MODEL_FILE = MODEL_DIR / "Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"
MODEL_URL = (
    "https://huggingface.co/Fluxx08/jamii-afya-qwen36-35b-q2k/resolve/main/"
    "Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"
)
MIN_SIZE = 12_000_000_000


def expected_hash() -> str:
    configured = os.environ.get("MODEL_SHA256", "").strip().lower()
    if configured:
        return configured
    with urllib.request.urlopen(MODEL_URL + ".sha256") as response:
        text = response.read().decode("utf-8", "replace")
    match = re.search(r"[0-9a-fA-F]{64}", text)
    if not match:
        raise RuntimeError("host did not provide a SHA256 sidecar; set MODEL_SHA256")
    return match.group(0).lower()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    if path.stat().st_size < MIN_SIZE:
        raise RuntimeError("artifact is too small")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch expected={expected} actual={actual}")
    print(f"[download_model] verified {path} ({path.stat().st_size} bytes) sha256={actual}")


def download(path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(MODEL_URL, headers=headers)
    with urllib.request.urlopen(request) as response:
        resumed = existing and response.status == 206
        mode = "ab" if resumed else "wb"
        if not resumed:
            existing = 0
        total = response.headers.get("Content-Length")
        total_bytes = existing + int(total) if total else None
        with partial.open(mode) as output:
            copied = existing
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                copied += len(chunk)
                if total_bytes:
                    print(f"\r[download_model] {copied}/{total_bytes} bytes", end="", flush=True)
        print()
    partial.replace(path)


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    expected = expected_hash()
    if MODEL_FILE.exists():
        verify(MODEL_FILE, expected)
        return 0
    print(f"[download_model] downloading {MODEL_URL}")
    download(MODEL_FILE)
    verify(MODEL_FILE, expected)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, urllib.error.URLError) as exc:
        print(f"[download_model] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
