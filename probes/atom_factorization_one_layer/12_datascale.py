"""12: data-scaling diagnostic for the student failure (h=512 only).

If test error improves steeply with N, the sweep failure is a DATA problem
(and the ceiling is inconclusive). If test is flat vs N, it is a
REPRESENTATION problem (student cannot fit routed block -- strong).

Protocol mirrors 11 exactly (same pool/splits restricted to fractions,
same seed family, val-checkpointing). Fractions of the 2463-point train
pool: {0.25, 0.5, 1.0}. Val/test fixed (256/held-384).

Compute: ~3x15 min contended. Run after 11 finishes.
Outputs: student_datascale.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
H = 512
EPOCHS = 400
BATCH = 256
LR = 3e-4
CKPT_EVERY = 20
FRACS = [0.25, 0.5, 1.0]


def swiglu(G, U, D, x):
    return D(torch.nn.functional.silu(G(x)) * U(x))


def main():
    t0 = time.time()
    sb = json.load(open(os.path.join(OUT, "split_big.json")))
    Xo = np.load(os.path.join(OUT, "inputs_L20.npy")).astype(np.float32)
    Bo = np.load(os.path.join(OUT, "targets_L20.npz"))["B"].astype(np.float32)
    Xn = np.load(os.path.join(OUT, "inputs_L20_X.npy")).astype(np.float32)
    Bn = np.load(os.path.join(OUT, "targets_L20_X.npz"))["B"].astype(np.float32)
    n_calib = sb["n_calib"]
    te = np.arange(n_calib, Xo.shape[0])
    Xpool = np.concatenate([Xo[sb["train_calib"]], Xn[sb["train_new"]]], axis=0)
    Bpool = np.concatenate([Bo[sb["train_calib"]], Bn[sb["train_new"]]], axis=0)
    Xva, Bva = Xn[sb["val_new"]], Bn[sb["val_new"]]
    Xte, Bte = Xo[te], Bo[te]
    Bte64 = Bte.astype(np.float64)
    nrm_te = np.linalg.norm(Bte64, axis=1)
    rng = np.random.default_rng(23)
    perm = rng.permutation(Xpool.shape[0])

    res = {"H": H, "fracs": {}}
    for f in FRACS:
        th = time.time()
        idx = perm[:int(f * Xpool.shape[0])]
        Xtr, Btr = Xpool[idx], Bpool[idx]
        mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-6
        mu_b, sd_b = Btr.mean(0), Btr.std(0) + 1e-6
        Xtrt = torch.from_numpy((Xtr - mu_x) / sd_x)
        Btrt = torch.from_numpy((Btr - mu_b) / sd_b)
        Xvat = torch.from_numpy((Xva - mu_x) / sd_x)
        Bvat = torch.from_numpy((Bva - mu_b) / sd_b)
        Xtet = torch.from_numpy((Xte - mu_x) / sd_x)
        torch.manual_seed(0)
        G = torch.nn.Linear(2048, H, bias=False)
        U = torch.nn.Linear(2048, H, bias=False)
        D = torch.nn.Linear(H, 2048, bias=False)
        for m in (G, U, D):
            torch.nn.init.normal_(m.weight, std=1.0 / np.sqrt(m.in_features))
        opt = torch.optim.Adam(list(G.parameters()) + list(U.parameters())
                               + list(D.parameters()), lr=LR)
        gen = torch.Generator().manual_seed(0)
        n = Xtrt.shape[0]
        nb = int(np.ceil(n / BATCH))
        best_vl, best_ep = 1e18, 0
        best_g, best_u, best_d = None, None, None
        for ep in range(EPOCHS):
            idxe = torch.randperm(n, generator=gen)
            for b in range(nb):
                bb = idxe[b * BATCH:(b + 1) * BATCH]
                opt.zero_grad()
                loss = ((swiglu(G, U, D, Xtrt[bb]) - Btrt[bb]) ** 2).mean()
                loss.backward()
                opt.step()
            if (ep + 1) % CKPT_EVERY == 0 or ep == EPOCHS - 1:
                with torch.no_grad():
                    vl = float(((swiglu(G, U, D, Xvat) - Bvat) ** 2).mean())
                if vl < best_vl:
                    best_vl, best_ep = vl, ep + 1
                    best_g = G.weight.detach().cpu().clone()
                    best_u = U.weight.detach().cpu().clone()
                    best_d = D.weight.detach().cpu().clone()
        with torch.no_grad():
            G.weight.copy_(best_g)
            U.weight.copy_(best_u)
            D.weight.copy_(best_d)
            pred = (swiglu(G, U, D, Xtet).numpy() * sd_b + mu_b).astype(
                np.float64)
        per = np.linalg.norm(pred - Bte64, axis=1) / nrm_te
        res["fracs"][f] = {
            "n": int(n), "best_ep": best_ep, "val_mse": best_vl,
            "pooled": float(np.linalg.norm(pred - Bte64)
                            / np.linalg.norm(Bte64)),
            "median": float(np.median(per)),
            "max": float(per.max()),
            "minutes": round((time.time() - th) / 60, 1),
        }
        g = res["fracs"][f]
        print(f"frac={f} n={n} ep={best_ep} pooled={g['pooled']:.4f} "
              f"med={g['median']:.4f} ({g['minutes']:.0f}m)", flush=True)
    json.dump(res, open(os.path.join(OUT, "student_datascale.json"), "w"),
              indent=1)
    print(f"OK 12 ({(time.time()-t0)/60:.0f}m total)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
