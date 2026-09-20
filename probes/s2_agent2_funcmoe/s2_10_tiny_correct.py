"""S2-Agent2 10: TINY nonlinear correction models C: D ~= C(Rk) / C(X).

Cheap falsification of top-k+correction AFTER linear probes failed (R2~0):
can a small 2-layer SiLU MLP (hidden h in {64,128,256}) predict the dropped
residual D4=D(R8-R4) from kept R4, or D2 from R2 (and from X)?
Protocol mirrors s2_06 (SAME seed-5 224/32/64 split, target standardized
with train stats, full-batch Adam 1500 steps, lr in {1e-3,3e-4} by val,
single seed). Metric matches s2_05: test ||D-Dhat||_F/||R8||_F + R2.
Deployable cost of C(Rk): 2*2048*h MACs/token/layer.
Output: results_correct.json (merged by layer, no clobber).
Usage: python3 s2_10_tiny_correct.py 0 | python3 s2_10_tiny_correct.py 5,20,30
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights")


def fit_mlp(A, D, tr, va, te, h, device="cpu"):
    At = torch.from_numpy(A.astype(np.float32))
    Dt = torch.from_numpy(D.astype(np.float32))
    mu, sd = Dt[tr].mean(0), Dt[tr].std(0) + 1e-6
    Dn = (Dt - mu) / sd
    Amu, Asd = At[tr].mean(0), At[tr].std(0) + 1e-6
    An = (At - Amu) / Asd
    best = None
    for lr in (1e-3, 3e-4):
        torch.manual_seed(0)
        F1 = torch.nn.Linear(A.shape[1], h, bias=True)
        F2 = torch.nn.Linear(h, D.shape[1], bias=True)
        torch.nn.init.normal_(F1.weight, std=1.0 / np.sqrt(F1.in_features))
        torch.nn.init.normal_(F2.weight, std=1.0 / np.sqrt(F2.in_features))
        torch.nn.init.zeros_(F1.bias)
        torch.nn.init.zeros_(F2.bias)
        opt = torch.optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=lr)
        for step in range(1500):
            opt.zero_grad()
            loss = ((F2(torch.nn.functional.silu(F1(An[tr]))) - Dn[tr]) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float(((F2(torch.nn.functional.silu(F1(An[va]))) - Dn[va]) ** 2).mean())
        if best is None or vl < best[0]:
            best = (vl, lr, F1, F2)
    vl, lr, F1, F2 = best
    with torch.no_grad():
        pred = F2(torch.nn.functional.silu(F1(An[te]))) * sd + mu
        pred = pred.numpy().astype(np.float64)
    return pred, lr, vl


def main():
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0]
    ext_path = os.path.join(HERE, "results_correct.json")
    res = json.load(open(ext_path)) if os.path.exists(ext_path) else {}
    for L in layers:
        d = np.load(os.path.join(OUT, f"outputs_L{L:02d}.npz"))
        N = d["R8"].shape[0]
        rng = np.random.default_rng(5)
        idx = rng.permutation(N)
        tr, va, te = idx[:224], idx[224:256], idx[256:]
        X = d["X"].astype(np.float64)
        R8 = d["R8"].astype(np.float64)
        out = {}
        for Dnm, Rnm in (("D4", "R4"), ("D2", "R2")):
            D = d[Dnm].astype(np.float64)
            Rk = d[Rnm].astype(np.float64)
            den = float((R8[te] ** 2).sum())
            Dc = D[te] - D[te].mean(axis=0, keepdims=True)
            r2den = float((Dc ** 2).sum())
            null = D[tr].mean(axis=0, keepdims=True)
            null_pred = np.broadcast_to(null, D[te].shape).copy()
            out[f"{Dnm}_nullmean"] = {
                "rel_err_R8": float(np.linalg.norm(null_pred - D[te]) / np.sqrt(den)),
                "R2": float(1.0 - float(((null_pred - D[te]) ** 2).sum()) / r2den)}
            for fnm, A in ((Rnm, Rk), ("X", X)):
                for h in (64, 128, 256):
                    pred, lr, vl = fit_mlp(A, D, tr, va, te, h)
                    num = float(((pred - D[te]) ** 2).sum())
                    m = {"rel_err_R8": float(np.sqrt(num / den)),
                         "R2": float(1.0 - num / r2den), "lr": lr, "val_mse": vl}
                    out[f"{Dnm}_from_{fnm}_h{h}"] = m
                    print(f"L{L:02d} {Dnm}<-{fnm}h{h}: rel={m['rel_err_R8']:.4f} "
                          f"R2={m['R2']:+.3f} lr={lr:g}", flush=True)
        res[str(L)] = out
    json.dump(res, open(ext_path, "w"), indent=1)
    print("OK s2_10", flush=True)


if __name__ == "__main__":
    main()
