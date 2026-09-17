"""05: COMMUNITY dictionaries -- per-community atom dictionaries.

1. Cluster the 256 experts by FUNCTIONAL behavior (calib only):
     affinity = 0.5 * co-routing Jaccard + 0.5 * (1+cos(Emean))/2
   Emean_calib[e] = sum_j mean_t(s_j) d_j, from saved S_all/D_all (no restream).
   SpectralClustering with C in {4, 8}.
2. Per community: own MiniBatchKMeans dictionary over its member atoms,
   budget split proportional to member atoms; total K matches a global K
   for fair comparison. Medoids + 1D rescale + expert codes as in 04.
3. Same held-out beta/betas eval.

KILL criterion relevance: community model MUST be materially better than
global to justify the extra structure; else that criterion fires.

Outputs: dict_comm_C{c}_K{K}.npz, dict_comm_summary.json
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans, SpectralClustering

from lib import (OUT, build_features, eval_beta, medoid_weights, medoids_for,
                 rescale_and_codes)

HERE = os.path.dirname(os.path.abspath(__file__))
C_GRID = [4, 8]
K_GRID = [1024, 4096]
M_GRID = [128, 256, 512, 1024, 2048]


def expert_communities(n_calib):
    r = np.load(os.path.join(OUT, "routing_L20.npz"))
    top8 = r["top8"][:n_calib]
    # co-routing Jaccard over calib tokens
    pres = np.zeros((n_calib, 256), bool)
    for t in range(n_calib):
        pres[t, top8[t]] = True
    inter = pres.T @ pres
    cnt = pres.sum(axis=0)
    union = cnt[:, None] + cnt[None, :] - inter
    jac = inter / np.maximum(union, 1)
    # calib mean expert outputs from saved signatures
    S = np.load(os.path.join(OUT, "S_all.npy"), mmap_mode="r")
    D = np.load(os.path.join(OUT, "D_all.npy"), mmap_mode="r")
    Emean = np.zeros((256, 2048), np.float64)
    for e in range(256):
        sl = slice(e * 512, (e + 1) * 512)
        sm = S[sl].astype(np.float64).mean(axis=1)  # (512,)
        Emean[e] = (sm[:, None] * D[sl].astype(np.float64)).sum(axis=0)
    del S, D
    En = Emean / np.maximum(np.linalg.norm(Emean, axis=1, keepdims=True),
                            1e-30)
    cos = En @ En.T
    aff = 0.5 * jac + 0.5 * (1.0 + cos) / 2.0
    return aff


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

    print("expert affinity...", flush=True)
    aff = expert_communities(n_calib)
    print("building features...", flush=True)
    F, Nc, A = build_features()

    summary = {"M_grid": M_GRID, "configs": {}}
    for C in C_GRID:
        sc = SpectralClustering(n_clusters=C, affinity="precomputed",
                                n_init=10, random_state=0)
        clab = sc.fit_predict(aff)
        sizes = [int((clab == c).sum()) for c in range(C)]
        print(f"C={C} expert sizes={sizes}", flush=True)
        for K in K_GRID:
            tk = time.time()
            # budget proportional to member atoms (=512 x experts)
            Kc = [max(1, round(K * sizes[c] / 256)) for c in range(C)]
            # fix rounding drift on largest community
            Kc[int(np.argmax(sizes))] += K - sum(Kc)
            assign = np.zeros(A, np.int32)
            med = np.zeros(K, np.int64)
            off = 0
            for c in range(C):
                members = np.where(clab[np.arange(A) // 512] == c)[0]
                Fc = F[members]
                km = MiniBatchKMeans(n_clusters=Kc[c], batch_size=4096,
                                     max_iter=100, n_init=2, random_state=0)
                ac = km.fit_predict(Fc).astype(np.int32)
                assign[members] = ac + off
                med_c = medoids_for(Fc, ac, Kc[c])
                for k in range(Kc[c]):
                    med[off + k] = (members[med_c[k]]
                                    if med_c[k] >= 0 else -1)
                off += Kc[c]
            dead = int((med == -1).sum())
            Gm, Um, Dm, med_e, med_j = medoid_weights(med, K)
            coef, r2, codes = rescale_and_codes(assign, med, K, A)
            np.savez_compressed(
                os.path.join(OUT, f"dict_comm_C{C}_K{K}.npz"),
                assign=assign, coef=coef, r2=r2, med_e=med_e, med_j=med_j,
                codes=codes, clab=clab.astype(np.int32))
            res, _ = eval_beta(top8h, w8h, Xh, R8h, codes, Gm, Um, Dm,
                               M_GRID)
            key = f"C{C}_K{K}"
            summary["configs"][key] = {
                "C": C, "K": K, "expert_sizes": sizes,
                "K_per_comm": Kc, "dead_clusters": dead,
                "r2_mean": float(r2.mean()),
                "clust_seconds": round(time.time() - tk, 1),
                **res,
            }
            b = res["beta"][512]
            print(f"{key}: R2={r2.mean():.3f} beta@512: "
                  f"mean={b['rel_mean']:.4f} med={b['rel_median']:.4f} "
                  f"p99={b['rel_p99']:.4f} max={b['rel_max']:.4f} "
                  f"({time.time()-tk:.0f}s)", flush=True)
    json.dump(summary, open(os.path.join(OUT, "dict_comm_summary.json"),
                            "w"), indent=1)
    print(f"OK 05 ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
