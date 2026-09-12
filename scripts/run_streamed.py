#!/usr/bin/env python3
"""Run a long command with line-buffered live output and a durable text log."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--cwd", default=None)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        ap.error("a command is required after --")
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        header = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] COMMAND {' '.join(command)}\n"
        print(header, end="", flush=True)
        log.write(header)
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\n")
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            rendered = f"[{stamp}] +{time.monotonic() - started:.1f}s {line}\n"
            print(rendered, end="", flush=True)
            log.write(rendered)
        code = process.wait()
        footer = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] EXIT={code} elapsed_seconds={time.monotonic() - started:.1f}\n"
        print(footer, end="", flush=True)
        log.write(footer)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
