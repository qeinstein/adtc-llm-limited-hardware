#!/usr/bin/env python3
"""Build the self-contained baseline kernel: injects join4_phase6.h into
base_src.py -> edge0base_v1.py (the pushed code_file).

Run from the repo root: python3 kaggle/native-sparse-edge0base-v1/mk_base_kernel.py
"""
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
KDIR = ROOT / "kaggle" / "native-sparse-edge0base-v1"


def main():
    src = (KDIR / "base_src.py").read_text()
    c_src = (ROOT / "probes/edge0_port/join4_phase6.h").read_text()
    out = src.replace('"@@JOIN4_C@@"', repr(c_src))
    assert "@@" not in out, "unreplaced marker left"
    dest = KDIR / "edge0base_v1.py"
    dest.write_text(out)
    py_compile.compile(str(dest), doraise=True)
    print(f"wrote {dest} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
