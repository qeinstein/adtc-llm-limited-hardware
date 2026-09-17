"""Shared helpers for the one-layer atom factorization experiment."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
DEQ = "/tmp/agent1_f32"
L = 20


def load_exp_rows(e):
    G = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32",
                    dtype=np.float32).reshape(512, 2048)
    U = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32",
                    dtype=np.float32).reshape(512, 2048)
    D = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_down.f32",
                    dtype=np.float32).reshape(2048, 512)
    return G, U, D


def build_features():
    """F = [s_hat/sqrt(Nc), P(d_hat)/sqrt(64)], (A, Nc+64) f32 dense."""
    import json
    meta = json.load(open(os.path.join(OUT, "atom_meta.json")))
    A, Nc = meta["A"], meta["n_calib"]
    S = np.load(os.path.join(OUT, "S_all.npy"), mmap_mode="r")
    D = np.load(os.path.join(OUT, "D_all.npy"), mmap_mode="r")
    st = np.load(os.path.join(OUT, "atom_stats.npz"))
    snorm, dnorm = st["snorm"], st["dnorm"]
    rng = np.random.default_rng(7)
    Pi = (rng.standard_normal((2048, 64)) / np.sqrt(64)).astype(np.float32)
    F = np.zeros((A, Nc + 64), np.float32)
    CH = 4096
    for a in range(0, A, CH):
        b = min(a + CH, A)
        s = S[a:b].astype(np.float32)
        n = snorm[a:b].copy()
        n[n == 0] = 1.0
        F[a:b, :Nc] = (s / n[:, None]) / np.sqrt(Nc)
        d = D[a:b].astype(np.float32)
        m = dnorm[a:b].copy()
        m[m == 0] = 1.0
        p = (d / m[:, None]) @ Pi
        pn = np.linalg.norm(p, axis=1, keepdims=True)
        pn[pn == 0] = 1.0
        F[a:b, Nc:] = (p / pn) / np.sqrt(64)
    del S, D
    return F, Nc, A


def medoids_for(F, assign, K):
    med = np.zeros(K, np.int64)
    for k in range(K):
        m = np.where(assign == k)[0]
        if len(m) == 0:
            med[k] = -1
            continue
        cen = F[m].mean(axis=0).astype(np.float64)
        d2 = ((F[m].astype(np.float64) - cen) ** 2).sum(axis=1)
        med[k] = m[int(np.argmin(d2))]
    return med


def medoid_weights(med, K):
    """Reload (g,u,d) rows for medoid atoms. Returns Gm,Um,Dm,med_e,med_j."""
    med_e = (med.clip(min=0) // 512).astype(np.int32)
    med_j = (med.clip(min=0) % 512).astype(np.int32)
    Gm = np.zeros((K, 2048), np.float32)
    Um = np.zeros((K, 2048), np.float32)
    Dm = np.zeros((K, 2048), np.float32)
    by_exp = {}
    for k in range(K):
        if med[k] >= 0:
            by_exp.setdefault(int(med_e[k]), []).append(k)
    for e, ks in by_exp.items():
        G, U, Dw = load_exp_rows(e)
        for k in ks:
            j = int(med_j[k])
            Gm[k], Um[k], Dm[k] = G[j], U[j], Dw[:, j]
        del G, U, Dw
    return Gm, Um, Dm, med_e, med_j


def rescale_and_codes(assign, med, K, A):
    """1D function-space rescale per atom + expert codes.

    c_j = <phi_j,phi_k>/<phi_k,phi_k> on calib, where
    <phi_j,phi_k> = (s_j.s_k)(d_j.d_k).
    codes[e,k] = sum of c_j over expert e's atoms in cluster k.
    Returns coef (A,), r2 (A,), codes (256,K).
    """
    S = np.load(os.path.join(OUT, "S_all.npy"), mmap_mode="r")
    Dmm = np.load(os.path.join(OUT, "D_all.npy"), mmap_mode="r")
    st = np.load(os.path.join(OUT, "atom_stats.npz"))
    sn2 = st["snorm"].astype(np.float64) ** 2
    dn2 = st["dnorm"].astype(np.float64) ** 2
    _, _, Dm, _, _ = medoid_weights(med, K)
    sk = S[med.clip(min=0)].astype(np.float64)
    sk[med == -1] = 0.0
    dk = Dm.astype(np.float64)
    den = (sk ** 2).sum(axis=1) * (dk ** 2).sum(axis=1)
    den[den == 0] = 1.0
    coef = np.zeros(A, np.float32)
    r2 = np.zeros(A, np.float32)
    codes = np.zeros((256, K), np.float32)
    for k in range(K):
        m = np.where(assign == k)[0]
        if len(m) == 0 or med[k] < 0:
            continue
        Sm = S[m].astype(np.float64)
        Dm_ = Dmm[m].astype(np.float64)
        num = (Sm @ sk[k]) * (Dm_ @ dk[k])
        c = num / den[k]
        coef[m] = c
        base = sn2[m] * dn2[m]
        r2[m] = np.where(base > 0, (num ** 2) / (den[k] * base), 0.0)
        ee = m // 512
        for e in np.unique(ee):
            codes[e, k] = c[ee == e].sum()
    del S, Dmm
    return coef, r2, codes


def eval_beta(top8h, w8h, Xh, R8h, codes, Gm, Um, Dm, M_GRID,
              rules=("beta", "betas")):
    """Held-out eval with routing-composed beta. Returns res dict + Smh."""
    H = Xh.shape[0]
    nR = np.linalg.norm(R8h, axis=1)
    gh = Xh.astype(np.float64) @ Gm.T.astype(np.float64)
    uh = Xh.astype(np.float64) @ Um.T.astype(np.float64)
    Smh = ((gh / (1.0 + np.exp(-gh))) * uh)
    dmn = np.linalg.norm(Dm.astype(np.float64), axis=1)
    Dm64 = Dm.astype(np.float64)
    res = {}
    for rule in rules:
        errs = {M: [] for M in M_GRID}
        coss = {M: [] for M in M_GRID}
        nnz_b = []
        for t in range(H):
            beta = (w8h[t][:, None] * codes[top8h[t]]).sum(axis=0)
            nnz_b.append(int((beta != 0).sum()))
            if rule == "beta":
                score = np.abs(beta)
            else:
                score = np.abs(beta * Smh[t]) * dmn
            rk = np.argsort(-score, kind="stable")
            for M in M_GRID:
                sel = rk[:M]
                Rh = ((beta[sel] * Smh[t][sel])[:, None]
                      * Dm64[sel]).sum(axis=0)
                errs[M].append(float(np.linalg.norm(Rh - R8h[t]) / nR[t]))
                den_ = np.linalg.norm(Rh) * nR[t]
                coss[M].append(float((Rh @ R8h[t]) / max(den_, 1e-30)))
        res[rule] = {}
        for M in M_GRID:
            v = np.array(errs[M])
            res[rule][M] = {
                "rel_mean": float(v.mean()),
                "rel_median": float(np.median(v)),
                "rel_p95": float(np.quantile(v, 0.95)),
                "rel_p99": float(np.quantile(v, 0.99)),
                "rel_max": float(v.max()),
                "cos_mean": float(np.mean(coss[M])),
                "cos_min": float(np.min(coss[M])),
            }
        res[rule]["nnz_beta_mean"] = float(np.mean(nnz_b))
    return res, Smh
