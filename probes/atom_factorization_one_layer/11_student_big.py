"""11: big-N dense SwiGLU student ceiling vs B=R8+S (L20).

Fair-capacity rerun of 09: train 2624 / val 256 (fresh pool) / test held-384
(fixed). Full-batch is infeasible at N=2624 (h=3072 would take ~15 h), so:
minibatch-256 Adam, 400 epochs, lr=3e-4 (stable default; small-N run picks
the lr separately), val-checkpoint every 20 epochs (keep best), input+target
standardized on train. Tensors preloaded; per-step slices only.

Same held-out metrics as 09: per-token rel (median/p95/p99/max/mean),
cosine (mean/min), pooled rel; train rel (fit check).

Compute: ~2 h contended for the full h grid. Run in background.

Outputs: student_big_summary.json
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
EPOCHS = 400
BATCH = 256
LR = 3e-4
CKPT_EVERY = 20


def swiglu(G, U, D, x):
    return D(torch.nn.functional.silu(G(x)) * U(x))


def main():
    t0 = time.time()
    print(f"torch {torch.__version__} threads={torch.get_num_threads()}",
          flush=True)
    sb = json.load(open(os.path.join(OUT, "split_big.json")))
    Xo = np.load(os.path.join(OUT, "inputs_L20.npy")).astype(np.float32)
    Bo = np.load(os.path.join(OUT, "targets_L20.npz"))["B"].astype(np.float32)
    Xn = np.load(os.path.join(OUT, "inputs_L20_X.npy")).astype(np.float32)
    Bn = np.load(os.path.join(OUT, "targets_L20_X.npz"))["B"].astype(np.float32)
    n_calib = sb["n_calib"]
    te = np.arange(n_calib, Xo.shape[0])
    Xtr = np.concatenate([Xo[sb["train_calib"]], Xn[sb["train_new"]]], axis=0)
    Btr = np.concatenate([Bo[sb["train_calib"]], Bn[sb["train_new"]]], axis=0)
    Xva, Bva = Xn[sb["val_new"]], Bn[sb["val_new"]]
    Xte, Bte = Xo[te], Bo[te]
    print(f"train={Xtr.shape[0]} val={Xva.shape[0]} test={Xte.shape[0]}",
          flush=True)

    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-6
    mu_b, sd_b = Btr.mean(0), Btr.std(0) + 1e-6
    Xtrt = torch.from_numpy((Xtr - mu_x) / sd_x)
    Btrt = torch.from_numpy((Btr - mu_b) / sd_b)
    Xvat = torch.from_numpy((Xva - mu_x) / sd_x)
    Bvat = torch.from_numpy((Bva - mu_b) / sd_b)
    Xtet = torch.from_numpy((Xte - mu_x) / sd_x)
    Bte64 = Bte.astype(np.float64)
    nrm_te = np.linalg.norm(Bte64, axis=1)

    res = {"n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0]),
           "n_test": int(Xte.shape[0]), "epochs": EPOCHS, "batch": BATCH,
           "lr": LR, "H": {}}
    gen = torch.Generator().manual_seed(0)
    for h in H_GRID:
        th = time.time()
        torch.manual_seed(0)
        G = torch.nn.Linear(2048, h, bias=False)
        U = torch.nn.Linear(2048, h, bias=False)
        D = torch.nn.Linear(h, 2048, bias=False)
        for m in (G, U, D):
            torch.nn.init.normal_(m.weight, std=1.0 / np.sqrt(m.in_features))
        opt = torch.optim.Adam(list(G.parameters()) + list(U.parameters())
                               + list(D.parameters()), lr=LR)
        n = Xtrt.shape[0]
        nb = int(np.ceil(n / BATCH))
        best_vl, best_ep = 1e18, 0
        best_g, best_u, best_d = None, None, None
        for ep in range(EPOCHS):
            idx = torch.randperm(n, generator=gen)
            for b in range(nb):
                bb = idx[b * BATCH:(b + 1) * BATCH]
                opt.zero_grad()
                loss = ((swiglu(G, U, D, Xtrt[bb]) - Btrt[bb]) ** 2).mean()
                loss.backward()
                opt.step()
            if (ep + 1) % CKPT_EVERY == 0 or ep == EPOCHS - 1:
                with torch.no_grad():
                    vl = float(((swiglu(G, U, D, Xvat) - Bvat) ** 2).mean())
                if vl < best_vl:
                    best_vl = vl
                    best_ep = ep + 1
                    best_g = G.weight.detach().cpu().clone()
                    best_u = U.weight.detach().cpu().clone()
                    best_d = D.weight.detach().cpu().clone()
        with torch.no_grad():
            G.weight.copy_(best_g)
            U.weight.copy_(best_u)
            D.weight.copy_(best_d)
            pred = (swiglu(G, U, D, Xtet).numpy() * sd_b + mu_b).astype(
                np.float64)
            pred_tr = (swiglu(G, U, D, Xtrt).numpy() * sd_b + mu_b).astype(
                np.float64)
        per = np.linalg.norm(pred - Bte64, axis=1) / nrm_te
        cos = ((pred * Bte64).sum(1)
               / np.maximum(np.linalg.norm(pred, axis=1) * nrm_te, 1e-30))
        pooled = float(np.linalg.norm(pred - Bte64)
                       / np.linalg.norm(Bte64))
        tr_rel = float(np.linalg.norm(pred_tr - Btr.astype(np.float64))
                       / np.linalg.norm(Btr.astype(np.float64)))
        res["H"][h] = {
            "best_ep": best_ep, "val_mse": best_vl, "train_rel": tr_rel,
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
        print(f"h={h:5d} ep={best_ep} train={tr_rel:.4f} pooled={pooled:.4f} "
              f"med={g['rel_median']:.4f} p95={g['rel_p95']:.4f} "
              f"p99={g['rel_p99']:.4f} max={g['rel_max']:.4f} "
              f"cos={g['cos_mean']:.4f} ({g['minutes']:.0f}m)", flush=True)
    json.dump(res, open(os.path.join(OUT, "student_big_summary.json"), "w"),
              indent=1)
    print(f"OK 11 ({(time.time()-t0)/60:.0f}m total)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
