"""06: SHARED + PRIVATE -- keep outlier atoms exact.

On top of a fitted dictionary config (global K* and best community cfg):
  outlier score per atom = z(log cmax) + z(clip(kurt,0,50)) + z(1 - R2)
  (cmax/kurt from calib contributions; R2 = cluster 1D-fit quality)
Private set P (atom ids, fixed offline): top |P| scorers, |P| in grid.
Private atoms are EXCLUDED from cluster codes and evaluated EXACTLY when
their expert routes:
  Rhat = sum_{top-M beta} beta_k s_k(x) d_k + sum_{private in routed} a s d
Cost counted: M + avg # private atoms among the routed 4096.

KILL relevance: if tails need such a large P that compute reduction <1.5x,
or tails still explode -> KILL.

Outputs: dict_private_summary.json
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from lib import OUT, eval_beta, load_exp_rows

HERE = os.path.dirname(os.path.abspath(__file__))
P_GRID = [512, 2048, 8192]
M_GRID = [256, 512, 1024]


def z(v):
    v = np.asarray(v, float)
    mu, sd = v.mean(), v.std()
    return (v - mu) / (sd if sd > 0 else 1.0)


def run_config(tag, dpath, top8h, w8h, Xh, R8h, H, summary):
    d = np.load(dpath)
    assign, coef, r2 = d["assign"], d["coef"].astype(np.float64), d["r2"]
    med_e, med_j, codes = d["med_e"], d["med_j"], d["codes"].astype(np.float64)
    K = codes.shape[1]
    Gm = np.zeros((K, 2048), np.float32)
    Um = np.zeros((K, 2048), np.float32)
    Dm = np.zeros((K, 2048), np.float32)
    by_exp = defaultdict(list)
    for k in range(K):
        by_exp[int(med_e[k])].append(k)
    for e, ks in by_exp.items():
        G, U, Dw = load_exp_rows(e)
        for k in ks:
            j = int(med_j[k])
            Gm[k], Um[k], Dm[k] = G[j], U[j], Dw[:, j]
        del G, U, Dw

    st = np.load(os.path.join(OUT, "atom_stats.npz"))
    score = (z(np.log(st["cmax"].astype(np.float64) + 1e-12))
             + z(np.clip(st["kurt"].astype(np.float64), 0, 50))
             + z(1.0 - r2.astype(np.float64)))
    order = np.argsort(-score, kind="stable")

    # private atom weights cached per P (largest superset covers smaller)
    pmax = max(P_GRID)
    pset_max = order[:pmax]
    Gp = np.zeros((pmax, 2048), np.float32)
    Up = np.zeros((pmax, 2048), np.float32)
    Dp = np.zeros((pmax, 2048), np.float32)
    pe = pset_max // 512
    pj = pset_max % 512
    by_exp2 = defaultdict(list)
    for i, e in enumerate(pe):
        by_exp2[int(e)].append(i)
    for e, ii in by_exp2.items():
        G, U, Dw = load_exp_rows(e)
        for i in ii:
            j = int(pj[i])
            Gp[i], Up[i], Dp[i] = G[j], U[j], Dw[:, j]
        del G, U, Dw
    # per-expert private atom index lists (positions in pset_max)
    priv_of_exp = defaultdict(list)
    for i, e in enumerate(pe):
        priv_of_exp[int(e)].append(i)

    gh = Xh.astype(np.float64) @ Gm.T.astype(np.float64)
    uh = Xh.astype(np.float64) @ Um.T.astype(np.float64)
    Smh = ((gh / (1.0 + np.exp(-gh))) * uh)
    Dm64 = Dm.astype(np.float64)
    nR = np.linalg.norm(R8h, axis=1)

    summary[tag] = {}
    for P in P_GRID:
        pset = set(pset_max[:P].tolist())
        # codes excluding private atoms
        codes_p = codes.copy()
        for a in pset_max[:P]:
            e = int(a // 512)
            codes_p[e, assign[a]] -= coef[a]
        errs = {M: [] for M in M_GRID}
        coss = {M: [] for M in M_GRID}
        pcnt = []
        for t in range(H):
            beta = (w8h[t][:, None] * codes_p[top8h[t]]).sum(axis=0)
            rk = np.argsort(-np.abs(beta), kind="stable")
            # exact private part for routed experts
            Rp = np.zeros(2048, np.float64)
            npriv = 0
            x = Xh[t].astype(np.float64)
            for s in range(8):
                e = int(top8h[t, s])
                ii = [i for i in priv_of_exp.get(e, []) if i < P]
                if not ii:
                    continue
                gv = Gp[ii].astype(np.float64) @ x
                uv = Up[ii].astype(np.float64) @ x
                sv = (gv / (1.0 + np.exp(-gv))) * uv
                Rp += ((w8h[t, s] * sv)[:, None]
                       * Dp[ii].astype(np.float64)).sum(axis=0)
                npriv += len(ii)
            pcnt.append(npriv)
            for M in M_GRID:
                sel = rk[:M]
                Rh = (((beta[sel] * Smh[t][sel])[:, None] * Dm64[sel])
                      .sum(axis=0) + Rp)
                errs[M].append(float(np.linalg.norm(Rh - R8h[t]) / nR[t]))
                den_ = np.linalg.norm(Rh) * nR[t]
                coss[M].append(float((Rh @ R8h[t]) / max(den_, 1e-30)))
        summary[tag][P] = {
            "priv_routed_mean": float(np.mean(pcnt)),
            "priv_routed_max": int(np.max(pcnt)),
            "M": {},
        }
        for M in M_GRID:
            v = np.array(errs[M])
            summary[tag][P]["M"][M] = {
                "rel_mean": float(v.mean()),
                "rel_median": float(np.median(v)),
                "rel_p95": float(np.quantile(v, 0.95)),
                "rel_p99": float(np.quantile(v, 0.99)),
                "rel_max": float(v.max()),
                "cos_mean": float(np.mean(coss[M])),
            }
        g = summary[tag][P]["M"][512]
        print(f"{tag} P={P:5d} routed_priv={np.mean(pcnt):.1f} "
              f"@512: mean={g['rel_mean']:.4f} med={g['rel_median']:.4f} "
              f"p99={g['rel_p99']:.4f} max={g['rel_max']:.4f}", flush=True)


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

    gs = json.load(open(os.path.join(OUT, "dict_global_summary.json")))
    # pick best global K by beta median@512, tie-break smaller K
    Ks = sorted(int(k) for k in gs["K"])
    bestK = min(Ks, key=lambda k: (gs["K"][str(k)]["beta"]["512"]
                                  ["rel_median"], k))
    print(f"best global K={bestK}", flush=True)
    summary = {"M_grid": M_GRID, "P_grid": P_GRID,
               "base_global_K": bestK, "cfgs": {}}
    run_config(f"global_K{bestK}",
               os.path.join(OUT, f"dict_global_K{bestK}.npz"),
               top8h, w8h, Xh, R8h, H, summary["cfgs"])
    cs = json.load(open(os.path.join(OUT, "dict_comm_summary.json")))
    bestc = min(cs["configs"],
                key=lambda k: (cs["configs"][k]["beta"]["512"]
                               ["rel_median"], k))
    print(f"best community cfg={bestc}", flush=True)
    summary["base_comm"] = bestc
    C = cs["configs"][bestc]["C"]
    K = cs["configs"][bestc]["K"]
    run_config(f"comm_{bestc}",
               os.path.join(OUT, f"dict_comm_C{C}_K{K}.npz"),
               top8h, w8h, Xh, R8h, H, summary["cfgs"])
    json.dump(summary, open(os.path.join(OUT, "dict_private_summary.json"),
                            "w"), indent=1)
    print(f"OK 06 ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
