"""Agent3 probe 06: ONE cheap one-layer reconstruction per width.

Target: full MoE block output B(x) = shared(x) + R8(x) (ONE layer only,
  chosen by argv, default 10). Model: dense SwiGLU MLP width w in
  {512,1024,2048,3072}: Bhat = Down(silu(Gate x) * Up x).
FIT METHOD (stated, cheap -- NOT full distillation): full-batch Adam on
  train tokens (n=224, MSE), 1500 steps, lr in {1e-3, 3e-4} picked by val
  (n=32); small-normal init scaled 1/sqrt(fan_in); NO lr schedule, NO
  augmentation, single seed; test (n=64) rel err ||B-Bhat||/||B||.
  Baseline: linear ridge (probe-05 method) X -> B.
CPU cost is trivial (N=320); this stays a probe, not a training run.
Output: results_mlp.json
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights")


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    widths = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [512, 1024, 2048, 3072]
    d = np.load(os.path.join(OUT, f"outputs_L{L:02d}.npz"))
    X = d["X"].astype(np.float32)
    B = (d["R8"].astype(np.float64) + d["S"].astype(np.float64)).astype(np.float32)
    N = X.shape[0]
    rng = np.random.default_rng(5)
    idx = rng.permutation(N)
    tr, va, te = idx[:224], idx[224:256], idx[256:]
    Xt = torch.from_numpy(X)
    Bt = torch.from_numpy(B)
    # standardize target for stable training; invert at eval
    mu, sd = Bt[tr].mean(0), Bt[tr].std(0) + 1e-6
    Bn = (Bt - mu) / sd
    res = {"layer": L}
    torch.manual_seed(0)
    for w in widths:
        best = None
        for lr in (1e-3, 3e-4):
            torch.manual_seed(0)
            G = torch.nn.Linear(2048, w, bias=False)
            U = torch.nn.Linear(2048, w, bias=False)
            D = torch.nn.Linear(w, 2048, bias=False)
            for m in (G, U, D):
                torch.nn.init.normal_(m.weight, std=1.0 / np.sqrt(m.in_features))
            opt = torch.optim.Adam(list(G.parameters()) + list(U.parameters()) +
                                   list(D.parameters()), lr=lr)
            for step in range(1500):
                opt.zero_grad()
                h = torch.nn.functional.silu(G(Xt[tr])) * U(Xt[tr])
                loss = ((D(h) - Bn[tr]) ** 2).mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                h = torch.nn.functional.silu(G(Xt[va])) * U(Xt[va])
                vl = float((((D(h) - Bn[va])) ** 2).mean())
            if best is None or vl < best[0]:
                best = (vl, lr, G, U, D)
        vl, lr, G, U, D = best
        with torch.no_grad():
            h = torch.nn.functional.silu(G(Xt[te])) * U(Xt[te])
            pred = D(h) * sd[te] + mu[te] if False else D(h) * sd + mu
            num = float(((pred - Bt[te]) ** 2).sum())
            den = float((Bt[te] ** 2).sum())
        rel = float(np.sqrt(num / den))
        res[w] = {"test_rel": rel, "lr": lr, "val_mse": vl}
        print(f"L{L} w={w}: test_rel={rel:.4f} lr={lr:g}", flush=True)
    # linear ridge baseline X -> B (probe-05 method)
    A = X.astype(np.float64)
    Bb = B.astype(np.float64)
    muA, sdA = A[tr].mean(0), A[tr].std(0) + 1e-12
    Z = np.concatenate([(A - muA) / sdA, np.ones((N, 1))], axis=1)
    best, bw = 1e18, None
    for lam in (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4):
        G = Z[tr].T @ Z[tr] + (lam * Z.shape[1]) * np.eye(Z.shape[1])
        w = np.linalg.solve(G, Z[tr].T @ Bb[tr])
        v = float(np.linalg.norm(Z[va] @ w - Bb[va]) ** 2)
        if v < best:
            best, bw = v, w
    pred = Z[te] @ bw
    rel = float(np.linalg.norm(pred - Bb[te]) / np.linalg.norm(Bb[te]))
    res["linear"] = {"test_rel": rel}
    print(f"L{L} linear: test_rel={rel:.4f}", flush=True)
    json.dump(res, open(os.path.join(HERE, "results_mlp.json"), "w"), indent=1)
    print("OK probe06", flush=True)


if __name__ == "__main__":
    main()
