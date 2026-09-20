#!/usr/bin/env python3
"""Build the frozen sparse llama.cpp runtime on any supported host.

This is the platform-neutral equivalent of ``build_runtime.sh``.  Keeping the
actual checkout/patch/CMake orchestration in Python lets native Windows users
run it from PowerShell or cmd without requiring a POSIX shell; Linux and macOS
continue to use the small shell wrapper for backwards compatibility.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PIN = "3057bb66c86c46d5781e50e85462a760ba7d1feb"


def run(args: list[str], *, cwd: Path | None = None) -> None:
    print("[build_runtime]", " ".join(str(arg) for arg in args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def binary_candidates(build: Path, name: str) -> list[Path]:
    return [
        build / "bin" / name,
        build / "bin" / f"{name}.exe",
        build / "bin" / "Release" / name,
        build / "bin" / "Release" / f"{name}.exe",
    ]


def header_sha() -> str:
    return hashlib.sha256(
        (ROOT / "probes" / "edge0_port" / "join4_phase6.h").read_bytes()
    ).hexdigest()


def main(argv: list[str]) -> int:
    llama_dir = Path(
        argv[1] if len(argv) > 1 else
        os.environ.get("ADTC_LLAMA_DIR", str(ROOT / "runtime" / "llama.cpp"))
    ).expanduser().resolve()
    build = llama_dir / "build-native"
    stamp = llama_dir / ".edge0-stamp"

    if not (llama_dir / ".git").is_dir():
        print(f"[build_runtime] cloning llama.cpp -> {llama_dir}", flush=True)
        llama_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none",
             "https://github.com/ggml-org/llama.cpp.git", str(llama_dir)])

    run(["git", "fetch", "--depth", "1", "origin", PIN], cwd=llama_dir)
    run(["git", "checkout", "--detach", PIN], cwd=llama_dir)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=llama_dir, text=True
    ).strip()
    if head != PIN:
        raise RuntimeError(f"[build_runtime] ERROR: NOT on pin: {head}")
    print(f"[build_runtime] on pin {head}", flush=True)

    want = f"{PIN} {header_sha()}"
    if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == want:
        print("[build_runtime] patches already applied (stamp match), skipping",
              flush=True)
    else:
        if stamp.is_file():
            print("[build_runtime] stamp mismatch: resetting tracked changes and re-patching",
                  flush=True)
            run(["git", "checkout", "--", "."], cwd=llama_dir)
        run([sys.executable, str(ROOT / "probes" / "edge0_port" / "join4_apply.py"),
             str(llama_dir)])
        stamp.write_text(want + "\n", encoding="utf-8")

    run(["cmake", "-S", str(llama_dir), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DLLAMA_CURL=ON"])
    jobs = str(max(1, os.cpu_count() or 4))
    run(["cmake", "--build", str(build), "--config", "Release",
         "--parallel", jobs, "--target", "llama-server", "llama-cli",
         "llama-bench"])

    for name in ("llama-server", "llama-cli", "llama-bench"):
        paths = binary_candidates(build, name)
        found = next((path for path in paths if path.is_file()), None)
        if found is None:
            raise FileNotFoundError(
                f"[build_runtime] ERROR: missing {name}; checked "
                + ", ".join(str(path) for path in paths)
            )
        print(f"[build_runtime] {name}: {found}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
