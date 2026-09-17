"""09: dense SwiGLU student ceiling vs full MoE block B=R8+S (L20).

The LEARNED-BASIS ceiling: can newly synthesized nonlinear atoms
(a dense SwiGLU MLP of width h) span the routed block output?
  D_h(x) = Wd' [SiLU(Wg' x) * (Wu' x)],  h in {512,1024,2048,3072}

Protocol (stronger than s2_06: bigger N, true held-out test):
  - train 256 / val 64 from calib-320 (fixed split); TEST on held-384
    (disjoint token IDs -- real generalization, not a split of calib).
  - input + target standardized on train stats.
  - full-batch Adam, 1500 steps (s2 parity for comparability),
    lr in {1e-3, 3e-4} picked by val. Tensors pre-indexed (the minibatch
    version with per-step fancy indexing was ~5x too slow on this box).
  - metrics on HELD: per-token rel err (median/p95/p99/max/mean),
    cosine (mean/min), pooled rel; train rel (fit check).
  - baselines: linear ridge, mean predictor.

Compute: whole sweep ~2-4 h contended CPU. Run in background.

Outputs: student_summary.json, student_preds.npz (local only).
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
H_GRID = [512, 1024, 2048, 3072]
STEPS = 1500


def evaluate(G, U, D, Xt, Bt):
    with torch.no_grad():
        h = torch.nn.functional.silu(G(Xt)) * U(Xt)
        pred = D(h)
    return pred


def main():
    t0 = time.time()
    print(f"torch {torch.__version__} threads={torch.get_num_threads()}",
          flush=True)
    split = json.load(open(os.path.join(OUT, "split.json")))
    n_calib = split["n_calib"]
    X = np.load(os.path.join(OUT, "inputs_L20.npy")).astype(np.float32)
    B = np.load(os.path.join(OUT, "targets_L20.npz"))["B"].astype(np.float32)
    rng = np.random.default_rng(5)
    perm = rng.permutation(n_calib)
    tr, va = perm[:256], perm[256:]
    te = np.arange(n_calib, X.shape[0])

    mu_x, sd_x = X[tr].mean(0), X[tr].std(0) + 1e-6
    mu_b, sd_b = B[tr].mean(0), B[tr].std(0) + 1e-6
    Xn = torch.from_numpy((X - mu_x) / sd_x)
    Bn = torch.from_numpy((B - mu_b) / sd_b)
    Bt = torch.from_numpy(B)
    # pre-indexed splits (no per-step fancy indexing)
    Xtr, Xva, Xte = Xn[tr], Xn[va], Xn[te]
    Btr, Bva = Bn[tr], Bn[va]
    nrm_te = torch.linalg.norm(Bt[te], dim=1).numpy()

    res = {"n_train": 256, "n_val": 64, "n_test": int(len(te)),
           "steps": STEPS, "mode": "full-batch", "H": {}}
    for h in H_GRID:
        th = time.time()
        best = None
        for lr in (1e-3, 3e-4):
            tl = time.time()
            torch.manual_seed(0)
            G = torch.nn.Linear(2048, h, bias=False)
            U = torch.nn.Linear(2048, h, bias=False)
            D = torch.nn.Linear(h, 2048, bias=False)
            for m in (G, U, D):
                torch.nn.init.normal_(m.weight,
                                      std=1.0 / np.sqrt(m.in_features))
            opt = torch.optim.Adam(list(G.parameters())
                                   + list(U.parameters())
                                   + list(D.parameters()), lr=lr)
            for step in range(STEPS):
                opt.zero_grad()
                hh = torch.nn.functional.silu(G(Xtr)) * U(Xtr)
                loss = ((D(hh) - Btr) ** 2).mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                hv = torch.nn.functional.silu(G(Xva)) * U(Xva)
                vl = float(((D(hv) - Bva) ** 2).mean())
            print(f"  h={h} lr={lr:g} val_mse={vl:.4f} "
                  f"({(time.time()-tl)/60:.1f}m)", flush=True)
            if best is None or vl < best[0]:
                best = (vl, lr, G, U, D)
        vl, lr, G, U, D = best
        with torch.no_grad():
            hte = torch.nn.functional.silu(G(Xte)) * U(Xte)
            pred = (D(hte).numpy() * sd_b + mu_b).astype(np.float64)
            htr = torch.nn.functional.silu(G(Xtr)) * U(Xtr)
            pred_tr = (D(htr).numpy() * sd_b + mu_b).astype(np.float64)
        Bt_np = B[te].astype(np.float64)
        per = np.linalg.norm(pred - Bt_np, axis=1) / nrm_te
        cos = ((pred * Bt_np).sum(1)
               / np.maximum(np.linalg.norm(pred, axis=1) * nrm_te, 1e-30))
        pooled = float(np.linalg.norm(pred - Bt_np)
                       / np.linalg.norm(Bt_np))
        tr_rel = float(np.linalg.norm(pred_tr - B[tr].astype(np.float64))
                       / np.linalg.norm(B[tr].astype(np.float64)))
        res["H"][h] = {
            "lr": lr, "val_mse": vl, "train_rel": tr_rel,
            "test_pooled_rel": pooled,
            "rel_mean": float(per.mean()),
            "rel_median": float(np.median(per)),
            "rel_p95": float(np.quantile(per, 0.95)),
            "rel_p99": float(np.quantile(per, 0.99)),
            "rel_max": float(per.max()),
            "cos_mean": float(cos.mean()),
            "cos_min": float(cos.min()),
            "minutes": round((time.time() - th) / 60, 1),
        }
        g = res["H"][h]
        print(f"h={h:5d} lr={lr:g} train={tr_rel:.4f} pooled={pooled:.4f} "
              f"med={g['rel_median']:.4f} p95={g['rel_p95']:.4f} "
              f"p99={g['rel_p99']:.4f} max={g['rel_max']:.4f} "
              f"cos={g['cos_mean']:.4f} ({g['minutes']:.0f}m)", flush=True)

    # baselines on same splits
    A = X.astype(np.float64)
    Bb = B.astype(np.float64)
    muA, sdA = A[tr].mean(0), A[tr].std(0) + 1e-12
    Z = np.concatenate([(A - muA) / sdA, np.ones((A.shape[0], 1))], axis=1)
    best, bw = 1e18, None
    for lam in (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4):
        M = Z[tr].T @ Z[tr] + (lam * Z.shape[1]) * np.eye(Z.shape[1])
        w = np.linalg.solve(M, Z[tr].T @ Bb[tr])
        v = float(np.linalg.norm(Z[va] @ w - Bb[va]) ** 2)
        if v < best:
            best, bw = v, w
    pred = Z[te] @ bw
    per = np.linalg.norm(pred - Bb[te], axis=1) / nrm_te
    res["linear"] = {"test_pooled_rel": float(
        np.linalg.norm(pred - Bb[te]) / np.linalg.norm(Bb[te])),
        "rel_median": float(np.median(per)),
        "rel_max": float(per.max())}
    mp = np.broadcast_to(Bb[tr].mean(0), Bb[te].shape).copy()
    res["mean_pred"] = {"test_pooled_rel": float(
        np.linalg.norm(mp - Bb[te]) / np.linalg.norm(Bb[te]))}
    print(f"linear: pooled={res['linear']['test_pooled_rel']:.4f} "
          f"mean-pred: {res['mean_pred']['test_pooled_rel']:.4f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "student_summary.json"), "w"),
              indent=1)
    print(f"OK 09 ({(time.time()-t0)/60:.0f}m total)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
