"""Agent3 probe 05: cheap linear probes for D4/D2 predictability.

METHOD (stated): ridge regression, closed form w=(A'A+lI)^-1 A'D on
  train tokens (n=224, fixed seed-5 shuffle), features standardized with
  TRAIN mean/std; lambda grid {1e-3,1e-2,...,1e4}x(n_feat) picked by a
  32-token val split; reported metrics on held-out 64 test tokens:
  test rel err ||D-Dhat||_F/||R8||_F and R^2 vs residual mean.
Feature sets: X | router-logits(256) | router-probs(256) | Rk (k=4/2).
Plus low-rank-constrained deployable correction: Dk ~= Rk @ Vr @ B with Vr
  = top-r right-singular vectors of TRAIN Rk (r=16,64,256), B fit by ridge;
  cost model 2*2048*r MACs/layer.
Output: results_probes.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights")
LAM_GRID = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4]


def ridge_fit(A, D, lam):
    n, f = A.shape
    G = A.T @ A + (lam * f) * np.eye(f)
    return np.linalg.solve(G, A.T @ D)


def ev(Dhat, D, R8):
    num = float(np.linalg.norm(Dhat - D) ** 2)
    den = float(np.linalg.norm(R8) ** 2)
    Dc = D - D.mean(axis=0, keepdims=True)
    r2 = 1.0 - num / float((Dc ** 2).sum())
    return {"rel_err_R8": float(np.sqrt(num / den)), "R2": float(r2)}


def run_probe(A, D, R8, idx_tr, idx_va, idx_te):
    mu, sd = A[idx_tr].mean(0), A[idx_tr].std(0) + 1e-12
    Z = (A - mu) / sd
    Zb = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
    best, bw = 1e18, None
    for lam in LAM_GRID:
        w = ridge_fit(Zb[idx_tr], D[idx_tr], lam)
        v = float(np.linalg.norm(Zb[idx_va] @ w - D[idx_va]) ** 2)
        if v < best:
            best, bw = v, (w, lam)
    w, lam = bw
    return ev(Zb[idx_te] @ w, D[idx_te], R8[idx_te]), lam


def main():
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 10, 20, 30]
    res = {}
    for L in layers:
        d = np.load(os.path.join(OUT, f"outputs_L{L:02d}.npz"))
        r = np.load(os.path.join(OUT, f"routing_L{L:02d}.npz"))
        N = d["R8"].shape[0]
        rng = np.random.default_rng(5)
        idx = rng.permutation(N)
        idx_tr, idx_va, idx_te = idx[:224], idx[224:256], idx[256:]
        X = d["X"].astype(np.float64)
        logits = r["logits"].astype(np.float64)
        P = np.exp(logits - logits.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        out = {}
        for Dnm, Rnm in (("D4", "R4"), ("D2", "R2")):
            D, Rk, R8 = (d[Dnm].astype(np.float64), d[Rnm].astype(np.float64),
                         d["R8"].astype(np.float64))
            feats = {"X": X, "logits": logits, "probs": P, Rnm: Rk}
            for fnm, A in feats.items():
                m, lam = run_probe(A, D, R8, idx_tr, idx_va, idx_te)
                out[f"{Dnm}_from_{fnm}"] = {**m, "lambda": lam}
                print(f"L{L:02d} {Dnm}<-{fnm}: rel={m['rel_err_R8']:.4f} "
                      f"R2={m['R2']:+.3f} lam={lam:g}", flush=True)
            # low-rank correction from Rk
            U, s, Vt = np.linalg.svd(Rk[idx_tr] - Rk[idx_tr].mean(0), full_matrices=False)
            for rr in (16, 64, 256):
                Vr = Vt[:rr].T
                A = Rk @ Vr
                m, lam = run_probe(A, D, R8, idx_tr, idx_va, idx_te)
                out[f"{Dnm}_from_{Rnm}_rank{rr}"] = {**m, "lambda": lam}
                print(f"L{L:02d} {Dnm}<-{Rnm}r{rr}: rel={m['rel_err_R8']:.4f} "
                      f"R2={m['R2']:+.3f}", flush=True)
        res[L] = out
    json.dump(res, open(os.path.join(HERE, "results_probes.json"), "w"), indent=1)
    print("OK probe05", flush=True)


if __name__ == "__main__":
    main()
