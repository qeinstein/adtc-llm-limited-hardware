"""Functional-basis oracle 02: held-out routed reconstruction at ranks.

For each held-out token: gather TRUE E_e(x) for its 8 routed experts
(small: H x 8 x 2048), project onto calib-fitted V_r, compose with TRUE a:
  Rhat_r = sum_e a_e V_r V_r^T E_e
Report per-rank routed-output errors + tails + cosine + per-expert errors
+ beta stats. Also: community-subspace comparison (C=4 co-routing groups,
same total r) at r in {64,128,256}; mean/full-rank controls.

Outputs: fb_oracle_<TAG>.json, fb_beta_<TAG>.npz (per-rank beta on held)
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
ATOM = "/home/fluxx/Workspace/adtc-llm-native-sparse/probes/atom_factorization_one_layer/assets"
DEQ = "/tmp/agent1_f32"
L = 20
RANKS = [16, 32, 64, 96, 128, 160, 192, 224, 256]
DATA_TAG = os.environ.get("FB_DATA", "proxy")


def load_held():
    if DATA_TAG == "proxy":
        split = json.load(open(os.path.join(ATOM, "split.json")))
        n_calib = split["n_calib"]
        X = np.load(os.path.join(ATOM, "inputs_L20.npy")).astype(np.float32)
        r = np.load(os.path.join(ATOM, "routing_L20.npz"))
        R8 = np.load(os.path.join(ATOM, "teacher_L20.npz"))["R8"].astype(
            np.float64)
        return (X[n_calib:], r["top8"][n_calib:],
                r["weights"][n_calib:].astype(np.float64), R8[n_calib:],
                r["top8"][:n_calib])
    else:
        d = np.load(os.path.join(OUT, "context_inputs.npz"))
        t = np.load(os.path.join(OUT, "context_teacher.npz"))
        keep = np.load(os.path.join(OUT, "context_te_keep.npy"))
        print(f"context: kept {int(keep.sum())}/{len(keep)} held rows "
              f"(near-dup excluded)", flush=True)
        return (d["Xte"][keep].astype(np.float32), d["top8te"][keep],
                d["w8te"][keep].astype(np.float64),
                t["R8te"][keep].astype(np.float64), d["top8tr"])


def main():
    t0 = time.time()
    Xh, top8, w8, R8, top8tr = load_held()
    H = Xh.shape[0]
    nR = np.linalg.norm(R8, axis=1)
    Vt = np.load(os.path.join(OUT, f"Vt_R256_{DATA_TAG}.npy")).astype(np.float64)
    print(f"DATA={DATA_TAG} H={H}", flush=True)

    # gather TRUE per-expert held outputs (H,8,2048)
    E = np.zeros((H, 8, 2048), np.float32)
    use = defaultdict(list)
    for t in range(H):
        for s in range(8):
            use[int(top8[t, s])].append((t, s))
    eids = sorted(use)
    print(f"streaming {len(eids)} experts for held E...", flush=True)
    for i, e in enumerate(eids):
        G = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32",
                        dtype=np.float32).reshape(512, 2048)
        U = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32",
                        dtype=np.float32).reshape(512, 2048)
        D = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_down.f32",
                        dtype=np.float32).reshape(2048, 512)
        gv = Xh @ G.T
        uv = Xh @ U.T
        Y = ((((gv / (1.0 + np.exp(-gv))) * uv) @ D.T)).astype(np.float64)
        for (t, s) in use[e]:
            E[t, s] = Y[t]
        del G, U, D, gv, uv, Y
        if (i + 1) % 80 == 0:
            print(f"  {i+1}/{len(eids)}", flush=True)
    # sanity: alpha-composed E == R8
    chk = (w8[:, :, None] * E.astype(np.float64)).sum(axis=1)
    assert np.allclose(chk, R8, rtol=1e-4, atol=1e-3), "E compose != R8"
    E64 = E.astype(np.float64)
    del E

    res = {"H": H, "ranks": {}, "controls": {}}
    betas = {}
    for r in RANKS:
        V = Vt[:r].T  # (2048,r)
        P = V @ V.T
        Eh = E64 @ P.T  # project each expert output
        Rh = (w8[:, :, None] * Eh).sum(axis=1)
        per = np.linalg.norm(Rh - R8, axis=1) / nR
        cos = ((Rh * R8).sum(1)
               / np.maximum(np.linalg.norm(Rh, axis=1) * nR, 1e-30))
        # per-expert reconstruction error (over all routed slots)
        slot_err = (np.linalg.norm(Eh - E64, axis=2)
                    / np.maximum(np.linalg.norm(E64, axis=2), 1e-30))
        # beta stats: beta_r = V^T R8 (== sum_e a_e V^T E_e by linearity)
        beta = R8 @ V
        betas[r] = beta.astype(np.float32)
        ab = np.abs(beta)
        pr = ((ab ** 2).sum(1) ** 2) / np.maximum(((ab ** 4).sum(1)), 1e-30)
        res["ranks"][r] = {
            "rel_mean": float(per.mean()), "rel_median": float(np.median(per)),
            "rel_p95": float(np.quantile(per, 0.95)),
            "rel_p99": float(np.quantile(per, 0.99)),
            "rel_max": float(per.max()),
            "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "expert_rel_mean": float(slot_err.mean()),
            "expert_rel_max": float(slot_err.max()),
            "beta_partratio_mean": float(pr.mean()),
            "beta_nnz_frac": float((ab > 1e-6).mean()),
        }
        g = res["ranks"][r]
        print(f"r={r:4d}: med={g['rel_median']:.4f} mean={g['rel_mean']:.4f} "
              f"p95={g['rel_p95']:.4f} p99={g['rel_p99']:.4f} "
              f"max={g['rel_max']:.4f} cos={g['cos_mean']:.4f} "
              f"exprt={g['expert_rel_mean']:.4f}", flush=True)
    np.savez_compressed(os.path.join(OUT, f"fb_beta_{DATA_TAG}.npz"), **{
        f"r{r}": betas[r] for r in RANKS})
    # interim dump: community section is OOM-prone; ranks must survive
    json.dump({"H": H, "TAG": DATA_TAG, "ranks": res["ranks"]},
              open(os.path.join(OUT, f"fb_oracle_{DATA_TAG}.json"), "w"),
              indent=1)

    # controls: mean predictor + full-rank exactness
    mu = R8.mean(axis=0)
    mp = np.broadcast_to(mu, R8.shape).copy()
    res["controls"]["mean_pred_pooled"] = float(
        np.linalg.norm(mp - R8) / np.linalg.norm(R8))
    Vfull = Vt  # 256 of 2048 -- NOT exact; verify r=2048 path on 8 tokens
    M = np.load(os.path.join(OUT, f"M_stack_{DATA_TAG}.npy"), mmap_mode="r")
    Q = M[:4096].astype(np.float64)
    Qc = Q - Q.mean(axis=0)
    U, S, W = np.linalg.svd(Qc, full_matrices=False)
    rec = (Qc @ W.T) @ W
    res["controls"]["full_rank_maxerr"] = float(
        np.linalg.norm(rec - Qc) / np.linalg.norm(Qc))
    assert res["controls"]["full_rank_maxerr"] < 1e-10

    # community comparison: C=4 co-routing groups, same total r
    pres = np.zeros((top8tr.shape[0], 256), bool)
    for t in range(top8tr.shape[0]):
        pres[t, top8tr[t]] = True
    inter = pres.T @ pres
    cnt = pres.sum(axis=0)
    union = cnt[:, None] + cnt[None, :] - inter
    jac = inter / np.maximum(union, 1)
    clab = SpectralClustering(n_clusters=4, affinity="precomputed",
                              n_init=10, random_state=0).fit_predict(jac)
    sizes = [int((clab == c).sum()) for c in range(4)]
    res["comm"] = {"sizes": sizes, "ranks": {}}
    MM = np.load(os.path.join(OUT, f"M_stack_{DATA_TAG}.npy"), mmap_mode="r")
    Nc = MM.shape[0] // 256
    for r in (64, 128, 256):
        rc = [max(1, round(r * sizes[c] / 256)) for c in range(4)]
        rc[int(np.argmax(sizes))] += r - sum(rc)
        Vc = {}
        for c in range(4):
            rows = np.concatenate([np.arange(e * Nc, (e + 1) * Nc)
                                   for e in range(256) if clab[e] == c])
            svd = TruncatedSVD(n_components=rc[c], random_state=0)
            svd.fit(MM[rows])
            Vc[c] = svd.components_.T.astype(np.float64)
        Eh = np.zeros_like(E64)
        for t in range(H):
            for s in range(8):
                c = clab[int(top8[t, s])]
                P = Vc[c] @ Vc[c].T
                Eh[t, s] = P @ E64[t, s]
        Rh = (w8[:, :, None] * Eh).sum(axis=1)
        per = np.linalg.norm(Rh - R8, axis=1) / nR
        res["comm"]["ranks"][r] = {
            "rel_median": float(np.median(per)),
            "rel_mean": float(per.mean()), "rel_max": float(per.max())}
        print(f"comm r={r}: med={np.median(per):.4f} "
              f"mean={per.mean():.4f} (global med="
              f"{res['ranks'][r]['rel_median']:.4f})", flush=True)
    json.dump(res, open(os.path.join(OUT, f"fb_oracle_{DATA_TAG}.json"), "w"),
              indent=1)
    print(f"OK fb02 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
