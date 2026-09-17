"""fb_01b: contextual PCA via stratified subsample (OOM-safe).

fb_01 died in TruncatedSVD on the full 521k x 2048 M (Q factor 625 MB).
A 2048-dim covariance does not need 521k rows: stratified 32k-row sample
(128/expert, fixed rng) estimates the same subspace. Stability control:
second independent subsample, subspace overlap ||V1V1'-V2V2'||_F reported.
Also writes TOTAL-variance-normalized cumulative variance (chunked ||M||^2).

Same outputs as fb_01 (context): Vt_R256_context.npy, S_top_context.npy,
spectrum_context.json
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fb_assets")
RMAX = 256
PER_EXP = 128


def main():
    t0 = time.time()
    M = np.load(os.path.join(OUT, "M_stack_context.npy"), mmap_mode="r")
    A, D = M.shape
    Ne = A // 256
    print(f"M={M.shape}", flush=True)
    Vt = {}
    for rep, seed in (("a", 0), ("b", 1)):
        rng = np.random.default_rng(seed)
        rows = np.concatenate(
            [e * Ne + rng.choice(Ne, PER_EXP, replace=False)
             for e in range(256)])
        Ms = np.asarray(M[rows], dtype=np.float32)
        svd = TruncatedSVD(n_components=RMAX, random_state=0)
        svd.fit(Ms)
        Vt[rep] = svd.components_.astype(np.float64)
        if rep == "a":
            S = svd.singular_values_.astype(np.float64)
        del Ms, svd
        print(f"  rep {rep} done ({time.time()-t0:.0f}s)", flush=True)
    P1 = Vt["a"].T @ Vt["a"]
    P2 = Vt["b"].T @ Vt["b"]
    # subspace overlap per r: mean squared cosine of principal angles
    ov = {}
    for r in (16, 64, 128, 256):
        C = Vt["a"][:r] @ Vt["b"][:r].T
        ov[r] = float((np.linalg.svd(C, compute_uv=False) ** 2).mean())
    print("overlap r16/64/128/256: " + "/".join(f"{ov[r]:.4f}" for r in ov),
          flush=True)
    np.save(os.path.join(OUT, "Vt_R256_context.npy"),
            Vt["a"].astype(np.float32))
    np.save(os.path.join(OUT, "S_top_context.npy"), S)
    tot = 0.0
    for a in range(0, A, 16384):
        b = M[a:a + 16384].astype(np.float64)
        tot += (b ** 2).sum()
    cum = np.cumsum(S ** 2) / tot
    ev = (S ** 2) / (S ** 2).sum()
    erank = float(np.exp(-(ev * np.log(np.maximum(ev, 1e-300))).sum()))
    spec = {"RMAX": RMAX, "Nc": Ne, "erank_top256": erank,
            "subsample_rows": 256 * PER_EXP, "subspace_overlap": ov,
            "cumvar_total": {r: float(cum[r - 1]) for r in
                             (16, 32, 64, 96, 128, 160, 192, 224, 256)},
            "top_sv_ratio": float(S[0] / S[-1])}
    json.dump(spec, open(os.path.join(OUT, "spectrum_context.json"), "w"),
              indent=1)
    print(f"erank={erank:.1f} totvar64={cum[63]:.4f} totvar128={cum[127]:.4f} "
          f"totvar256={cum[255]:.4f}", flush=True)
    print(f"OK fb01b ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
