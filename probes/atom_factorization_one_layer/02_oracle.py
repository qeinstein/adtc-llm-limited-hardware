"""02: ORACLE ceiling test -- the cheap KILL gate.

For each HELD-OUT token, gather the exact per-atom contributions of its 8
routed experts (4096 atoms):

    C[e,j] = alpha_e * s_{e,j}(x) * d_{e,j},  s = SiLU(g.x)*(u.x)

Rank atoms by ||C|| (magnitude oracle, TRUE coefficients, no refit) and
measure reconstruction error vs active atoms M at atom resolution:

    Rhat_M = cumsum of top-M contributions; err_M = ||R8 - Rhat_M|| / ||R8||

This is the ceiling for the hypothesis family: best per-token subset with
true coefficients. No routing-composed dictionary (fixed expert codes, no
per-token solve) can beat it. If the oracle itself needs M ~ 4096 for low
error, or stays >5% near M=2048 -> KILL the idea immediately.

Also measures a refit-headroom control on a token subset: top-M support +
least-squares coefficient refit (NOT part of the hypothesis mechanism; shows
whether failure is subset-selection vs coefficient quality).

Memory-frugal: tokens sorted by top-1 expert + small expert LRU cache.

Outputs (assets/):
  - oracle_curves.npz: err (H,4097) rel err at M=0..4096, cos (H,4097)
  - oracle_summary.json: grid stats + KILL-gate evaluation
  - oracle_refit.json: refit headroom on subset
"""
import json
import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
DEQ = "/tmp/agent1_f32"
L = 20
M_GRID = [128, 256, 512, 1024, 2048]
CACHE_CAP = 20  # experts (x12 MB)


class ExpCache:
    def __init__(self, cap):
        self.cap = cap
        self.d = OrderedDict()

    def get(self, e):
        if e in self.d:
            self.d.move_to_end(e)
            return self.d[e]
        G = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32",
                        dtype=np.float32).reshape(512, 2048)
        U = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32",
                        dtype=np.float32).reshape(512, 2048)
        D = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_down.f32",
                        dtype=np.float32).reshape(2048, 512)
        self.d[e] = (G, U, D)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return self.d[e]


def atom_contribs(x, e, alpha, cache):
    """(4096? no: 512,2048) exact atom contributions for one expert/token."""
    G, U, D = cache.get(e)
    gv = G @ x
    uv = U @ x
    s = (gv / (1.0 + np.exp(-gv))) * uv  # (512,)
    return (alpha * s)[:, None] * D.T  # (512,2048)


def main():
    split = json.load(open(os.path.join(OUT, "split.json")))
    n_calib, N = split["n_calib"], split["N"]
    X = np.load(os.path.join(OUT, "inputs_L20.npy")).astype(np.float32)
    r = np.load(os.path.join(OUT, "routing_L20.npz"))
    top8, w8 = r["top8"], r["weights"].astype(np.float64)
    R8 = np.load(os.path.join(OUT, "teacher_L20.npz"))["R8"].astype(np.float64)

    H = N - n_calib
    order_tok = sorted(range(n_calib, N), key=lambda t: top8[t, 0])
    cache = ExpCache(CACHE_CAP)
    ERR = np.zeros((H, 4097), dtype=np.float32)
    COS = np.zeros((H, 4097), dtype=np.float32)
    for hi, t in enumerate(order_tok):
        C = np.zeros((4096, 2048), np.float32)
        for s in range(8):
            e = int(top8[t, s])
            C[s * 512:(s + 1) * 512] = atom_contribs(
                X[t].astype(np.float64), e, w8[t, s], cache)
        # sanity: atom sum == teacher R8
        assert np.allclose(C.sum(axis=0).astype(np.float64), R8[t],
                           rtol=1e-4, atol=1e-3), f"atom sum != R8 at t={t}"
        ranking = np.argsort(-np.linalg.norm(C.astype(np.float64), axis=1))
        Cs = C[ranking].astype(np.float64)
        S = np.cumsum(Cs, axis=0)  # (4096,2048) f64 = 64 MB, transient
        nR = np.linalg.norm(R8[t])
        res = np.linalg.norm(S - R8[t][None, :], axis=1) / nR
        ERR[hi, 1:] = res
        ERR[hi, 0] = 1.0
        num = S @ R8[t]
        den = np.linalg.norm(S, axis=1) * nR
        COS[hi, 1:] = num / np.maximum(den, 1e-30)
        COS[hi, 0] = 0.0
        del C, Cs, S
        if (hi + 1) % 64 == 0:
            print(f"  {hi+1}/{H}", flush=True)

    np.savez_compressed(os.path.join(OUT, "oracle_curves.npz"),
                        err=ERR, cos=COS)
    q = lambda v, p: float(np.quantile(v, p))
    summ = {"H": H, "grid": {}}
    for M in M_GRID:
        v = ERR[:, M].astype(np.float64)
        c = COS[:, M].astype(np.float64)
        summ["grid"][M] = {
            "rel_mean": float(v.mean()), "rel_median": q(v, 0.5),
            "rel_p95": q(v, 0.95), "rel_p99": q(v, 0.99),
            "rel_max": float(v.max()),
            "cos_mean": float(c.mean()), "cos_min": float(c.min()),
        }
        g = summ["grid"][M]
        print(f"oracle M={M:5d}: rel mean={g['rel_mean']:.4f} "
              f"med={g['rel_median']:.4f} p95={g['rel_p95']:.4f} "
              f"p99={g['rel_p99']:.4f} max={g['rel_max']:.4f} "
              f"cos={g['cos_mean']:.4f}", flush=True)

    # KILL-gate evaluation (hypothesis criteria: >5% near M=2048 -> KILL)
    med2048 = summ["grid"][2048]["rel_median"]
    mean2048 = summ["grid"][2048]["rel_mean"]
    p992048 = summ["grid"][2048]["rel_p99"]
    gate = ("KILL" if (med2048 > 0.05 or mean2048 > 0.05) else
            "PASS_TO_DICTIONARY")
    summ["kill_gate"] = {
        "median_M2048": med2048, "mean_M2048": mean2048,
        "p99_M2048": p992048, "verdict": gate}
    print(f"ORACLE KILL GATE: median@2048={med2048:.4f} "
          f"mean@2048={mean2048:.4f} -> {gate}", flush=True)

    # refit-headroom control: top-M support + LS refit, subset of tokens
    sub = order_tok[:64]
    refit = {}
    for M in [128, 256, 512]:
        errs = []
        for t in sub:
            C = np.zeros((4096, 2048), np.float32)
            for s in range(8):
                e = int(top8[t, s])
                C[s * 512:(s + 1) * 512] = atom_contribs(
                    X[t].astype(np.float64), e, w8[t, s], cache)
            ranking = np.argsort(-np.linalg.norm(C.astype(np.float64),
                                                 axis=1))[:M]
            A = C[ranking].astype(np.float64).T  # (2048,M)
            coef, *_ = np.linalg.lstsq(A, R8[t], rcond=None)
            Rh = A @ coef
            errs.append(float(np.linalg.norm(Rh - R8[t])
                              / np.linalg.norm(R8[t])))
            del C, A
        refit[M] = {"rel_mean": float(np.mean(errs)),
                    "rel_median": float(np.median(errs)),
                    "rel_max": float(np.max(errs))}
        print(f"refit M={M}: mean={refit[M]['rel_mean']:.4f} "
              f"max={refit[M]['rel_max']:.4f}", flush=True)
    summ["refit_subset64"] = refit
    json.dump(summ, open(os.path.join(OUT, "oracle_summary.json"), "w"),
              indent=1)
    print("OK 02", flush=True)


if __name__ == "__main__":
    sys.exit(main())
