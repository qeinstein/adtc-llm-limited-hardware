"""S2-Agent2 04: SVD spectra of R8, D4, D2 (uncentered, deployable target).

Reports effective rank exp(H(p)) and variance explained at
16,32,64,128,256 + bootstrap CI on var-exp@64 for D4/D2.
Output: results_svd.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights")
RANKS = (16, 32, 64, 128, 256)


def spec(M):
    s = np.linalg.svd(M, compute_uv=False)
    e = s ** 2
    p = e / e.sum()
    erank = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    cum = np.cumsum(e) / e.sum()
    return s, erank, {r: float(cum[r - 1]) for r in RANKS if r <= len(s)}


def main():
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0]
    res_path = os.path.join(HERE, "results_svd.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    rng = np.random.default_rng(11)
    for L in layers:
        d = np.load(os.path.join(OUT, f"outputs_L{L:02d}.npz"))
        N = d["R8"].shape[0]
        row = {"N": N}
        for nm in ("R8", "D4", "D2"):
            s, er, ve = spec(d[nm].astype(np.float64))
            row[nm] = {"erank": er, "var_exp": ve, "top_sv_ratio": float(s[0] / s[-1]),
                       "rank_full": int(len(s))}
            print(f"L{L:02d} {nm}: erank={er:.1f} " +
                  " ".join(f"r{r}={ve.get(r, float('nan')):.3f}" for r in RANKS),
                  flush=True)
        for nm in ("D4", "D2"):
            M = d[nm].astype(np.float64)
            v = []
            for _ in range(30):
                idx = rng.integers(0, N, N)
                _, _, ve = spec(M[idx] - M[idx].mean(axis=0) + M.mean(axis=0))
                v.append(ve[64])
            v = np.array(v)
            row[nm]["var64_boot"] = [float(v.mean()), float(v.min()), float(v.max())]
            print(f"  {nm} var64 boot: {v.mean():.3f} [{v.min():.3f},{v.max():.3f}]",
                  flush=True)
        res[str(L)] = row
    json.dump(res, open(res_path, "w"), indent=1)
    print("OK s2_04", flush=True)


if __name__ == "__main__":
    main()
