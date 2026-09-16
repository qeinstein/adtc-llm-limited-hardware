"""fb_02c: contextual community-subspace check at r=256 (OOM-safe).

Mirrors fb_02b on contextual data: C=4 co-routing communities, same total
budget r=256, per-community PCA on SUBSAMPLED rows (32k/community max --
the degenerate giant community has 499k rows; full fancy-index copy OOMs).
Held R8 accumulated on the fly (no big arrays).

Outputs: fb_comm_context.json
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fb_assets")
DEQ = "/tmp/agent1_f32"
L = 20
R = 256


def main():
    t0 = time.time()
    d = np.load(os.path.join(OUT, "context_inputs.npz"))
    t = np.load(os.path.join(OUT, "context_teacher.npz"))
    keep = np.load(os.path.join(OUT, "context_te_keep.npy"))
    Xh = d["Xte"][keep].astype(np.float32)
    top8, w8 = d["top8te"][keep], d["w8te"][keep].astype(np.float64)
    R8 = t["R8te"][keep].astype(np.float64)
    top8tr = d["top8tr"]
    H = Xh.shape[0]
    nR = np.linalg.norm(R8, axis=1)

    pres = np.zeros((top8tr.shape[0], 256), bool)
    for i in range(top8tr.shape[0]):
        pres[i, top8tr[i]] = True
    inter = pres.T @ pres
    cnt = pres.sum(axis=0)
    union = cnt[:, None] + cnt[None, :] - inter
    clab = SpectralClustering(
        n_clusters=4, affinity="precomputed", n_init=10,
        random_state=0).fit_predict(inter / np.maximum(union, 1))
    sizes = [int((clab == c).sum()) for c in range(4)]
    print(f"sizes={sizes}", flush=True)
    rc = [max(1, round(R * sizes[c] / 256)) for c in range(4)]
    rc[int(np.argmax(sizes))] += R - sum(rc)

    MM = np.load(os.path.join(OUT, "M_stack_context.npy"), mmap_mode="r")
    Nc = MM.shape[0] // 256
    rng = np.random.default_rng(3)
    Vc = {}
    for c in range(4):
        rows = np.concatenate([np.arange(e * Nc, (e + 1) * Nc)
                               for e in range(256) if clab[e] == c])
        if len(rows) > 32768:
            rows = rng.choice(rows, 32768, replace=False)
        svd = TruncatedSVD(n_components=rc[c], random_state=0)
        svd.fit(np.asarray(MM[rows], dtype=np.float32))
        Vc[c] = svd.components_.T.astype(np.float64)
        print(f"  comm {c}: nexperts={sizes[c]} r={rc[c]}", flush=True)
    del MM

    Rh = np.zeros((H, 2048), np.float64)
    use = defaultdict(list)
    for i in range(H):
        for s in range(8):
            use[int(top8[i, s])].append((i, s))
    for e in sorted(use):
        G = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32",
                        dtype=np.float32).reshape(512, 2048)
        U = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32",
                        dtype=np.float32).reshape(512, 2048)
        D = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_down.f32",
                        dtype=np.float32).reshape(2048, 512)
        gv = Xh @ G.T
        uv = Xh @ U.T
        Y = ((((gv / (1.0 + np.exp(-gv))) * uv) @ D.T)).astype(np.float64)
        P = Vc[clab[e]]
        P = P @ P.T
        for (i, s) in use[e]:
            Rh[i] += w8[i, s] * (P @ Y[i])
        del G, U, D, gv, uv, Y
    per = np.linalg.norm(Rh - R8, axis=1) / nR
    out = {"C": 4, "sizes": sizes, "rc": rc, "R": R,
           "rel_median": float(np.median(per)),
           "rel_mean": float(per.mean()),
           "rel_p95": float(np.quantile(per, 0.95)),
           "rel_max": float(per.max())}
    print(f"CTX COMM r={R}: med={out['rel_median']:.4f} "
          f"mean={out['rel_mean']:.4f} (global med=0.8724)", flush=True)
    json.dump(out, open(os.path.join(OUT, "fb_comm_context.json"), "w"), indent=1)
    print(f"OK fb02c ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
