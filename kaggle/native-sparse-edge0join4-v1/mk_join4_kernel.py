#!/usr/bin/env python3
"""Build the self-contained JOIN4 kernel: injects join4_phase6.h + locked
pin sets into join4_src.py -> edge0join4_v1.py (the pushed code_file).

Run from the repo root: python3 kaggle/native-sparse-edge0join4-v1/mk_join4_kernel.py
"""
import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
KDIR = ROOT / "kaggle" / "native-sparse-edge0join4-v1"


def main():
    src = (KDIR / "join4_src.py").read_text()
    c_src = (ROOT / "probes/edge0_port/join4_phase6.h").read_text()
    cfg = json.loads((ROOT / "probes/edge0_port/cache_config_k4.json").read_text())
    pins = {}
    for tag in ("3.0", "4.0", "5.0", "6.0"):
        keys = cfg[tag]["pins_global"]
        assert len(keys) == cfg[tag]["npins"], tag
        assert all(0 <= k < 10240 for k in keys), tag
        pins[tag] = "\n".join(str(k) for k in keys) + "\n"
    out = src.replace('"@@JOIN4_C@@"', repr(c_src))
    out = out.replace('"@@PINS_3@@"', repr(pins["3.0"]))
    out = out.replace('"@@PINS_4@@"', repr(pins["4.0"]))
    out = out.replace('"@@PINS_5@@"', repr(pins["5.0"]))
    out = out.replace('"@@PINS_6@@"', repr(pins["6.0"]))
    assert "@@" not in out, "unreplaced marker left"
    dest = KDIR / "edge0join4_v1.py"
    dest.write_text(out)
    py_compile.compile(str(dest), doraise=True)
    print(f"wrote {dest} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
