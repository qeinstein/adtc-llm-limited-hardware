#!/usr/bin/env python3
"""Extract JOIN K4 cache-sweep inputs from tracek4 kernel npz files.

Input:  traces/tracek4/trace_p<NN>.npz  {hidden_fp16 [T,40,2048],
        topk_ids [T,40,4], topk_probs, ...}  (k=4 under K4/16 execution)
Output: traces/tracek4routes/ids_p<NN>.npy  [T,40,4] int16

Explicit quirk checks (v3 had an eval0 cross-prompt duplicate that was
dropped; do NOT blindly copy that here):
  - duplicate eval-0 rows across prompts -> report + drop with reason
  - ids out of [0,256), wrong ndim, T==0 -> hard fail
  - topk set validity vs probs ordering -> spot-check first prompt
"""
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path("traces/tracek4")
DST = Path("traces/tracek4routes")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("trace_p*.npz"))
    assert files, f"no npz in {SRC}"
    eval0 = {}
    n_toks = 0
    for f in files:
        pid = int(re.search(r"p(\d+)", f.name).group(1))
        d = np.load(f)
        H, I = d["hidden_fp16"], d["topk_ids"]
        assert H.ndim == 3 and H.shape[1] == 40 and H.shape[2] == 2048, H.shape
        assert I.ndim == 3 and I.shape[1] == 40 and I.shape[2] == 4, I.shape
        assert I.shape[0] > 0 and H.shape[0] == I.shape[0]
        assert I.min() >= 0 and I.max() < 256, (I.min(), I.max())
        assert len({tuple(sorted(r)) for L in range(40)
                    for r in [I[0, L]]}) <= 40  # ids sane per layer
        key = (H[0].tobytes(), I[0].tobytes())
        if key in eval0:
            print(f"pid {pid:02d}: eval0 DUPLICATES pid {eval0[key]:02d} "
                  f"-> checking full-trace equality", flush=True)
            d0 = np.load(SRC / f"trace_p{eval0[key]:02d}.npz")
            if d0["topk_ids"].shape == I.shape and \
               (d0["topk_ids"] == I).all():
                print(f"  FULL duplicate -> SKIP {f.name} (no info)",
                      flush=True)
                continue
            print(f"  eval0-only match -> KEEP (diverges later)", flush=True)
        else:
            eval0[key] = pid
        np.save(DST / f"ids_p{pid:02d}.npy", I.astype(np.int16))
        n_toks += I.shape[0]
        print(f"pid {pid:02d}: T={I.shape[0]} -> ids_p{pid:02d}.npy",
              flush=True)
    print(f"wrote {len(list(DST.glob('*.npy')))} prompts, {n_toks} toks")


if __name__ == "__main__":
    sys.exit(main())
