"""04: GLOBAL dictionary -- one shared atom dictionary for all experts.

Pipeline (all fits on CALIB only; held-out purely for eval):
  1. signatures: F = [s_hat/sqrt(Nc), P(d_hat)/sqrt(64)] (A,384)
     P = fixed Gaussian random projection 2048->64 (JL; avoids holding
     A x 2272 dense features in RAM).
  2. MiniBatchKMeans over A=131072 atoms, K in {512,...,8192}.
  3. medoid per cluster = REAL atom (evaluable SwiGLU neuron).
  4. 1D function-space rescale per atom onto its medoid:
       c_j = <phi_j,phi_k>/<phi_k,phi_k>  (calib inner products)
     <phi_j,phi_k> = (s_j.s_k)(d_j.d_k) -- no giant tensors.
  5. expert codes: c[e,k] = sum of c_j over expert e's atoms in cluster k.
  6. held-out eval with ROUTING-COMPOSED coefficients:
       beta_k(x) = sum_{e in top8} alpha_e(x) c[e,k]
     selection rule 'beta':  top-M by |beta_k|      (pure composition:
                             nonlinear evals = M only -- the hypothesis)
     selection rule 'betas': top-M by |beta_k s_k(x)| ||d_k||
                             (needs K linear pre-scores -- headroom control)
     Rhat = sum_selected beta_k s_k(x) d_k.

Outputs (assets/): dict_global_K{k}.npz, dict_global_summary.json
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from lib import (OUT, build_features, eval_beta, medoid_weights, medoids_for,
                 rescale_and_codes)

HERE = os.path.dirname(os.path.abspath(__file__))
K_GRID = [512, 1024, 2048, 4096, 8192]
M_GRID = [128, 256, 512, 1024, 2048]


def main():
    t0 = time.time()
    split = json.load(open(os.path.join(OUT, "split.json")))
    n_calib, N = split["n_calib"], split["N"]
    H = N - n_calib
    Xh = np.load(os.path.join(OUT, "inputs_L20.npy"))[n_calib:].astype(
        np.float32)
    r = np.load(os.path.join(OUT, "routing_L20.npz"))
    top8h, w8h = r["top8"][n_calib:], r["weights"][n_calib:].astype(np.float64)
    R8h = np.load(os.path.join(OUT, "teacher_L20.npz"))["R8"][
        n_calib:].astype(np.float64)

    print("building features...", flush=True)
    F, Nc, A = build_features()
    print(f"F={F.shape} ({F.nbytes/1e6:.0f} MB, {time.time()-t0:.0f}s)",
          flush=True)

    summary = {"M_grid": M_GRID, "K": {}}
    for K in K_GRID:
        tk = time.time()
        km = MiniBatchKMeans(n_clusters=K, batch_size=4096, max_iter=100,
                             n_init=3 if K <= 2048 else 2, random_state=0)
        assign = km.fit_predict(F).astype(np.int32)
        med = medoids_for(F, assign, K)
        dead = int((med == -1).sum())
        Gm, Um, Dm, med_e, med_j = medoid_weights(med, K)
        coef, r2, codes = rescale_and_codes(assign, med, K, A)
        nnz_code = (codes != 0).sum(axis=1).mean()

        np.savez_compressed(os.path.join(OUT, f"dict_global_K{K}.npz"),
                            assign=assign, coef=coef, r2=r2,
                            med_e=med_e, med_j=med_j, codes=codes)

        te = time.time()
        res, _ = eval_beta(top8h, w8h, Xh, R8h, codes, Gm, Um, Dm, M_GRID)
        res["eval_ms_per_token"] = round(
            (time.time() - te) * 1000 / H, 2)
        summary["K"][K] = {
            "dead_clusters": dead,
            "code_nnz_per_expert": float(nnz_code),
            "r2_mean": float(r2.mean()), "r2_median": float(np.median(r2)),
            "clust_seconds": round(time.time() - tk, 1),
            **res,
        }
        b = res["beta"][512]
        print(f"K={K:5d} dead={dead} nnzcode={nnz_code:.0f} "
              f"R2={r2.mean():.3f} beta@512: mean={b['rel_mean']:.4f} "
              f"med={b['rel_median']:.4f} p99={b['rel_p99']:.4f} "
              f"max={b['rel_max']:.4f} ({time.time()-tk:.0f}s)", flush=True)
    json.dump(summary, open(os.path.join(OUT, "dict_global_summary.json"),
                            "w"), indent=1)
    print(f"OK 04 ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
