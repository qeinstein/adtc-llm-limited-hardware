"""fb_02b: community-subspace check at r=256 only (OOM-safe rerun).

fb_02 died (OOM) in the community section after printing all global ranks.
This answers the one remaining question: can COMMUNITY-LOCALIZED subspaces
(C=4 co-routing groups, same total budget r=256) rescue the oracle where
the global 256-dim subspace fails (87% median)? If community-256 is still
far above 5%, the community branch is dead (smaller r only worse).

Also reports per-expert effective rank (each expert's own calib outputs):
if experts are individually high-rank, no sharing scheme can work.

Memory: f32 throughout, one community SVD at a time.
Outputs: fb_comm_proxy.json
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fb_assets")
ATOM = "/home/fluxx/Workspace/adtc-llm-native-sparse/probes/atom_factorization_one_layer/assets"
R = 256


def main():
    t0 = time.time()
    split = json.load(open(os.path.join(ATOM, "split.json")))
    n_calib = split["n_calib"]
    r = np.load(os.path.join(ATOM, "routing_L20.npz"))
    top8tr, top8 = r["top8"][:n_calib], r["top8"][n_calib:]
    w8 = r["weights"][n_calib:].astype(np.float64)
    R8 = np.load(os.path.join(ATOM, "teacher_L20.npz"))["R8"][
        n_calib:].astype(np.float64)
    H = top8.shape[0]
    nR = np.linalg.norm(R8, axis=1)

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
    print(f"sizes={sizes}", flush=True)

    MM = np.load(os.path.join(OUT, "M_stack_proxy.npy"), mmap_mode="r")
    Nc = MM.shape[0] // 256
    # per-expert effective rank (own calib outputs, 320x2048 each)
    eranks = []
    for e in range(256):
        Ye = MM[e * Nc:(e + 1) * Nc].astype(np.float32)
        s = np.linalg.svd(Ye, compute_uv=False).astype(np.float64)
        ev = s ** 2
        ev /= ev.sum()
        eranks.append(float(np.exp(-(ev * np.log(np.maximum(ev, 1e-300))).sum())))
        if (e + 1) % 64 == 0:
            print(f"  erank {e+1}/256", flush=True)
    eranks = np.array(eranks)
    print(f"per-expert erank: mean={eranks.mean():.1f} med="
          f"{np.median(eranks):.1f} min={eranks.min():.1f} "
          f"max={eranks.max():.1f}", flush=True)

    # community subspaces at total R
    rc = [max(1, round(R * sizes[c] / 256)) for c in range(4)]
    rc[int(np.argmax(sizes))] += R - sum(rc)
    Vc = {}
    for c in range(4):
        rows = np.concatenate([np.arange(e * Nc, (e + 1) * Nc)
                               for e in range(256) if clab[e] == c])
        svd = TruncatedSVD(n_components=rc[c], random_state=0)
        svd.fit(MM[rows])
        Vc[c] = svd.components_.T.astype(np.float32)
        print(f"  comm {c}: nexperts={sizes[c]} r={rc[c]}", flush=True)
    del MM

    # held E on the fly per expert (low RAM): accumulate Rhat directly
    Rh = np.zeros((H, 2048), np.float64)
    Xh = np.load(os.path.join(ATOM, "inputs_L20.npy"))[n_calib:].astype(
        np.float32)
    DEQ = "/tmp/agent1_f32"
    from collections import defaultdict
    use = defaultdict(list)
    for t in range(H):
        for s in range(8):
            use[int(top8[t, s])].append((t, s))
    for i, e in enumerate(sorted(use)):
        G = np.fromfile(f"{DEQ}/L20_E{e:03d}_gate.f32",
                        dtype=np.float32).reshape(512, 2048)
        U = np.fromfile(f"{DEQ}/L20_E{e:03d}_up.f32",
                        dtype=np.float32).reshape(512, 2048)
        D = np.fromfile(f"{DEQ}/L20_E{e:03d}_down.f32",
                        dtype=np.float32).reshape(2048, 512)
        gv = Xh @ G.T
        uv = Xh @ U.T
        Y = ((((gv / (1.0 + np.exp(-gv))) * uv) @ D.T)).astype(np.float64)
        P = Vc[clab[e]].astype(np.float64)
        P = P @ P.T
        for (t, s) in use[e]:
            Rh[t] += w8[t, s] * (P @ Y[t])
        del G, U, D, gv, uv, Y
    per = np.linalg.norm(Rh - R8, axis=1) / nR
    out = {"C": 4, "sizes": sizes, "rc": rc, "R": R,
           "rel_median": float(np.median(per)),
           "rel_mean": float(per.mean()), "rel_p95": float(np.quantile(per, 0.95)),
           "rel_max": float(per.max()),
           "per_expert_erank": {"mean": float(eranks.mean()),
                                "median": float(np.median(eranks)),
                                "min": float(eranks.min()),
                                "max": float(eranks.max())}}
    print(f"COMM r={R}: med={out['rel_median']:.4f} mean={out['rel_mean']:.4f} "
          f"p95={out['rel_p95']:.4f} max={out['rel_max']:.4f} "
          f"(global r=256 med=0.8662)", flush=True)
    json.dump(out, open(os.path.join(OUT, "fb_comm_proxy.json"), "w"), indent=1)
    print(f"OK fb02b ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
