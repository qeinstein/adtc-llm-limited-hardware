"""fb_04: CORRECTED expert-axis functional-basis oracle.

M[e,(n,d)] = E_e(x_n), shape [256, N*d]. Factorize on the EXPERT axis:
  M ~= C_r F_r,  C_r = top-r left singular vectors (256,r), static.
Held-out (oracular basis values from TRUE all-expert outputs):
  Phi(x) = C_r^T E_all(x);  Ehat = C_r Phi(x);  Rhat = alpha^T Ehat
Exact teacher routing throughout. r=256 MUST be exact (sanity).

Memory-frugal: Gram G = M M^T (256x256) accumulated from the M_stack
memmap in expert-pair blocks (no N*d transpose); E_all(held) streamed
once into an f32 memmap, then blocked rank eval.

Ranks: 16,32,64,96,128,160,192,224,240,248,252,256.
TAG=context|proxy via FB_DATA (default context).

Outputs: gram_<TAG>.npy, U_<TAG>.npy, S_<TAG>.npy, Eall_<TAG>.npy (local),
  expert_axis_<TAG>.json (ranks, spectrum, route-freq control, beta stats)
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fb_assets")
ATOM = "/home/fluxx/Workspace/adtc-llm-native-sparse/probes/atom_factorization_one_layer/assets"
DEQ = "/tmp/agent1_f32"
L = 20
RANKS = [16, 32, 64, 96, 128, 160, 192, 224, 240, 248, 252, 256]
TAG = os.environ.get("FB_DATA", "context")


def load_split():
    if TAG == "proxy":
        split = json.load(open(os.path.join(ATOM, "split.json")))
        n_calib = split["n_calib"]
        X = np.load(os.path.join(ATOM, "inputs_L20.npy")).astype(np.float32)
        r = np.load(os.path.join(ATOM, "routing_L20.npz"))
        R8 = np.load(os.path.join(ATOM, "teacher_L20.npz"))["R8"].astype(np.float64)
        return {"Nc_tag": "proxy", "Xh": X[n_calib:],
                "top8h": r["top8"][n_calib:],
                "w8h": r["weights"][n_calib:].astype(np.float64),
                "R8h": R8[n_calib:],
                "top8tr": r["top8"][:n_calib], "w8tr": r["weights"][:n_calib]}
    d = np.load(os.path.join(OUT, "context_inputs.npz"))
    t = np.load(os.path.join(OUT, "context_teacher.npz"))
    keep = np.load(os.path.join(OUT, "context_te_keep.npy"))
    return {"Nc_tag": "context", "Xh": d["Xte"][keep].astype(np.float32),
            "top8h": d["top8te"][keep],
            "w8h": d["w8te"][keep].astype(np.float64),
            "R8h": t["R8te"][keep].astype(np.float64),
            "top8tr": d["top8tr"], "w8tr": d["w8tr"]}


def main():
    t0 = time.time()
    sp = load_split()
    Xh, top8h, w8h, R8h = sp["Xh"], sp["top8h"], sp["w8h"], sp["R8h"]
    H = Xh.shape[0]
    nR = np.linalg.norm(R8h, axis=1)
    print(f"TAG={TAG} H={H}", flush=True)

    # 1. Gram G[e,f] = sum_n E_e(x_n).E_f(x_n) from M_stack memmap
    # (checkpointed: reuse U/S/gram if present)
    MM = np.load(os.path.join(OUT, f"M_stack_{TAG}.npy"), mmap_mode="r")
    Ne = MM.shape[0] // 256
    print(f"M_stack rows/expert Nc={Ne}", flush=True)
    if os.path.exists(os.path.join(OUT, f"U_{TAG}.npy")):
        G = np.load(os.path.join(OUT, f"gram_{TAG}.npy"))
        U = np.load(os.path.join(OUT, f"U_{TAG}.npy"))
        S = np.load(os.path.join(OUT, f"S_{TAG}.npy"))
        S2 = S ** 2
        print("  gram/U/S reused from checkpoint", flush=True)
    else:
        G = np.zeros((256, 256), np.float64)
        CH = 8  # expert block (32 OOMs: 2x1GB f64 blocks)
        for a in range(0, 256, CH):
            A = MM[a * Ne:(a + CH) * Ne].astype(np.float64).reshape(CH, -1)
            for b in range(a, 256, CH):  # upper triangle only (symmetric)
                B = MM[b * Ne:(b + CH) * Ne].astype(np.float64).reshape(CH, -1)
                G[a:a + CH, b:b + CH] = A @ B.T  # BLAS GEMM, not einsum
                if b != a:
                    G[b:b + CH, a:a + CH] = G[a:a + CH, b:b + CH].T
            del A
            print(f"  gram {a + CH}/256 ({time.time()-t0:.0f}s)", flush=True)
        G = (G + G.T) / 2
        np.save(os.path.join(OUT, f"gram_{TAG}.npy"), G)
        w, V = np.linalg.eigh(G)  # ascending
        w = w[::-1]
        U = V[:, ::-1].astype(np.float64)  # (256,256) static expert codes
        S2 = np.maximum(w, 0)
        S = np.sqrt(S2)
        np.save(os.path.join(OUT, f"U_{TAG}.npy"), U)
        np.save(os.path.join(OUT, f"S_{TAG}.npy"), S)
    ev = S2 / S2.sum()
    cum = np.cumsum(ev)
    erank = float(np.exp(-(ev * np.log(np.maximum(ev, 1e-300))).sum()))
    print(f"erank={erank:.1f} cumE64={cum[63]:.4f} cumE128={cum[127]:.4f} "
          f"cumE224={cum[223]:.4f} cumE256={cum[255]:.4f}", flush=True)

    # 2. route-frequency control (train + held)
    top8all = np.concatenate([sp["top8tr"], top8h], axis=0)
    w8all = np.concatenate(
        [np.asarray(sp["w8tr"], dtype=np.float64), w8h], axis=0)
    cnt = Counter(top8all.ravel().tolist())
    touched = len(cnt)
    mass = np.zeros(256)
    for t in range(top8all.shape[0]):
        for s in range(8):
            mass[int(top8all[t, s])] += float(w8all[t, s])
    mass /= mass.sum()
    mo = np.sort(mass)[::-1]
    print(f"experts touched: {touched}/256; top8/32/128 mass="
          f"{mo[:8].sum():.3f}/{mo[:32].sum():.3f}/{mo[:128].sum():.3f}",
          flush=True)
    # energy per expert on calib (diag of Gram) vs route mass
    diag = np.diag(G) / np.diag(G).sum()
    cc = float(np.corrcoef(np.log(diag + 1e-12), np.log(mass + 1e-12))[0, 1])

    # 3. stream ALL experts over held -> E_all memmap (256,H,2048) f32
    # (expert-major: sequential writes; token-major strided writes stall)
    epath = os.path.join(OUT, f"Eall_{TAG}.npy")
    try:
        E0 = np.load(epath, mmap_mode="r")
        have_E = E0.shape == (256, H, 2048)
        del E0
    except (OSError, ValueError):
        have_E = False
    E = np.lib.format.open_memmap(
        epath, mode="r+" if have_E else "w+", dtype=np.float32,
        shape=(256, H, 2048)) if have_E else np.lib.format.open_memmap(
        epath, mode="w+", dtype=np.float32, shape=(256, H, 2048))
    if have_E:
        print("  E_all reused from checkpoint", flush=True)
    for e in (range(0) if have_E else range(256)):
        Gt = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32",
                         dtype=np.float32).reshape(512, 2048)
        Ut = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32",
                         dtype=np.float32).reshape(512, 2048)
        D = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_down.f32",
                        dtype=np.float32).reshape(2048, 512)
        gv = Xh @ Gt.T
        uv = Xh @ Ut.T
        E[e] = ((((gv / (1.0 + np.exp(-gv))) * uv) @ D.T)).astype(np.float32)
        del Gt, Ut, D, gv, uv
        if (e + 1) % 64 == 0:
            print(f"  stream {e+1}/256 ({time.time()-t0:.0f}s)", flush=True)
    E.flush()
    # sanity: composed E == R8
    chk = np.zeros((H, 2048), np.float64)
    for t in range(H):
        chk[t] = (w8h[t][:, None] * E[:, t][top8h[t]].astype(np.float64)).sum(0)
    assert np.allclose(chk, R8h, rtol=1e-4, atol=1e-3), "E compose != R8"

    # 4. rank eval, token-blocked (B=16; routed-only accumulation).
    # Full Eh (B,256,2048) f64 OOMs; instead: Rhat from routed slots only,
    # expert errors accumulated per-expert in chunks. Peak ~200 MB.
    import gc
    res = {"H": H, "TAG": TAG, "Nc": Ne,
           "spectrum": {"erank": erank,
                        "cumE": {r: float(cum[r - 1]) for r in RANKS}},
           "routefreq": {"touched": touched,
                         "top_mass": [float(mo[:8].sum()), float(mo[:32].sum()),
                                       float(mo[:128].sum())],
                         "energy_mass_logcorr": cc},
           "ranks": {}}
    per_r = {r: [] for r in RANKS}
    cos_r = {r: [] for r in RANKS}
    # expert-error accumulators (subset of ranks: full 12x is 3x the flops
    # for a diagnostic; routed error is decisive and computed at all ranks)
    REXP = [16, 64, 128, 192, 256]
    expnum = {r: np.zeros(256) for r in REXP}
    expden = np.zeros(256)
    BB = 16
    for a in range(0, H, BB):
        b = min(a + BB, H)
        Eb = E[:, a:b].transpose(1, 0, 2).astype(np.float64)  # (B,256,2048)
        expden += (Eb ** 2).sum(axis=(0, 2))
        Phi256 = np.einsum("ei,bed->bid", U, Eb)  # once; slice views per rank
        for r in RANKS:
            C = U[:, :r]  # (256,r): C[e,i]
            Phi = Phi256[:, :r]
            C8 = C[top8h[a:b]]  # (B,8,r)
            Eh8 = np.einsum("bsi,bid->bsd", C8, Phi)  # routed only
            Rh = (w8h[a:b][:, :, None] * Eh8).sum(axis=1)
            per_r[r].append(np.linalg.norm(Rh - R8h[a:b], axis=1) / nR[a:b])
            cos_r[r].append(((Rh * R8h[a:b]).sum(1)) / np.maximum(
                np.linalg.norm(Rh, axis=1) * nR[a:b], 1e-30))
            if r in expnum:
                for e0 in range(0, 256, 32):
                    Ehc = np.einsum("ei,bid->bed", C[e0:e0 + 32], Phi)
                    expnum[r][e0:e0 + 32] += (
                        (Ehc - Eb[:, e0:e0 + 32]) ** 2).sum(axis=(0, 2))
            del Eh8, Rh
        del Eb, Phi256
        gc.collect()
        if (a // BB) % 8 == 0:
            print(f"  eval {b}/{H} ({time.time()-t0:.0f}s)", flush=True)
    for r in RANKS:
        v = np.concatenate(per_r[r])
        c = np.concatenate(cos_r[r])
        if r in expnum:
            ee = np.sqrt(expnum[r] / np.maximum(expden, 1e-300))  # per expert
            eem, eex = float(ee.mean()), float(ee.max())
        else:
            eem, eex = None, None
        res["ranks"][r] = {
            "rel_mean": float(v.mean()), "rel_median": float(np.median(v)),
            "rel_p95": float(np.quantile(v, 0.95)),
            "rel_p99": float(np.quantile(v, 0.99)),
            "rel_max": float(v.max()),
            "cos_mean": float(c.mean()), "cos_min": float(c.min()),
            "expert_rel_mean": eem, "expert_rel_max": eex}
        g = res["ranks"][r]
        ex = f"{g['expert_rel_mean']:.4f}" if g["expert_rel_mean"] is not None else "n/a"
        print(f"r={r:4d}: med={g['rel_median']:.4f} mean={g['rel_mean']:.4f} "
              f"p95={g['rel_p95']:.4f} p99={g['rel_p99']:.4f} "
              f"max={g['rel_max']:.4f} cos={g['cos_mean']:.4f} "
              f"exprt={ex}", flush=True)
    # r=256 SANITY: must be numerical precision
    sanity = res["ranks"][256]["rel_max"]
    print(f"SANITY r=256 max rel err: {sanity:.3e} "
          f"({'PASS' if sanity < 1e-4 else 'FAIL'})", flush=True)
    assert sanity < 1e-4, "r=256 not exact -- STOP AND DEBUG"
    # beta stats: beta_r(x) = sum_e a_e c[e,r] over top8 (no expert evals)
    beta = np.zeros((H, 256), np.float64)
    for t in range(H):
        beta[t] = (w8h[t][:, None] * U[top8h[t]]).sum(0)
    ab = np.abs(beta)
    pr = ((ab[:, :256] ** 2).sum(1) ** 2) / np.maximum((ab[:, :256] ** 4).sum(1), 1e-30)
    res["beta"] = {"partratio_mean_r256": float(pr.mean()),
                   "nnz_frac": float((ab > 1e-9).mean())}
    json.dump(res, open(os.path.join(OUT, f"expert_axis_{TAG}.json"), "w"),
              indent=1)
    print(f"OK fb04 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
