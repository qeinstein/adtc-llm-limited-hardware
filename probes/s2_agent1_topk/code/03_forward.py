"""S2 Agent1 (top-k) script 03: EXACT R8/Rk forward + SwiGLU z-stats + width energy.

Per (layer, token): route with REAL F32 router, forward REAL IQ2-dequant experts
  E_e(x) = Down(silu(Gate x) * Up x), f32 matmuls, f64 accumulation.
  R8 = sum_r w_r E_r (renormed, deployable); Rk = renormed top-k, k=1..7.
  Diagnostics: raw-scale (no-renorm) Rk vs R8; leave-one-rank-out errors.

Input models (weights/router/scales all REAL; token DIRECTIONS modeled):
  proxy: X = RMSNorm(real calib embeds) * g_L   (best at L00, weak deep)
  gauss: X ~ N(0, diag(s^2)), s^2 = REAL imatrix router E[x^2] (scales exact)

z-stats pooled over ROUTED (token,expert) pairs only: P(|z|<t), block sparsity,
  per-dim routed energy -> width 512->384/256 verdict input.

Usage: python3 03_forward.py 0 proxy[,gauss] [N_gauss]
Output: results/forward_L{L:02d}_{model}.json
"""
import json
import os
import struct
import sys
from collections import defaultdict

import numpy as np
import torch

WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
HERE = os.path.join(WS, "probes/s2_agent1_topk")
RES = os.path.join(HERE, "results")
A3W = os.path.join(WS, "probes/agent3_funcmoe/weights")
RAW = "/tmp/agent1_raw"
DEQ = "/tmp/agent1_f32"
IMA = "/tmp/imatrix.gguf"


def load_router_scales(L):
    d = open(IMA, "rb").read()
    off = 4 + 4
    nt = struct.unpack_from("<q", d, off)[0]
    off += 8
    nkv = struct.unpack_from("<q", d, off)[0]
    off += 8

    def rs(o):
        (n,) = struct.unpack_from("<q", d, o)
        o += 8
        return d[o:o + n].decode(), o + n

    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    for _ in range(nkv):
        _, off = rs(off)
        t = struct.unpack_from("<i", d, off)[0]
        off += 4
        if t == 8:
            _, off = rs(off)
        elif t == 9:
            at = struct.unpack_from("<i", d, off)[0]
            off += 4
            (n,) = struct.unpack_from("<q", d, off)
            off += 8
            if at == 8:
                for _ in range(n):
                    _, off = rs(off)
            else:
                off += SZ[at] * n
        else:
            off += SZ[t]
    want = {f"blk.{L}.ffn_gate_inp.weight.in_sum2",
            f"blk.{L}.ffn_gate_inp.weight.counts"}
    infos = {}
    for _ in range(nt):
        nm, off = rs(off)
        (nd,) = struct.unpack_from("<i", d, off)
        off += 4
        sh = [struct.unpack_from("<q", d, off + 8 * i)[0] for i in range(nd)]
        off += 8 * nd
        (ty,) = struct.unpack_from("<i", d, off)
        off += 4
        (to,) = struct.unpack_from("<q", d, off)
        off += 8
        if nm in want:
            infos[nm] = (sh, to)
    ds = (off + 31) // 32 * 32
    out = {}
    for nm, (sh, to) in infos.items():
        n = 1
        for v in sh:
            n *= v
        a = np.frombuffer(d, dtype=np.float32, count=n,
                          offset=ds + to).copy()
        out[nm] = a.reshape(tuple(sh[::-1]) if len(sh) > 1 else sh)
    s2 = out[f"blk.{L}.ffn_gate_inp.weight.in_sum2"].astype(np.float64)
    c = float(out[f"blk.{L}.ffn_gate_inp.weight.counts"].reshape(-1)[0])
    return np.maximum(s2 / c, 0)


def load_exp(L, e, kind):
    p = f"{DEQ}/L{L:02d}_E{e:03d}_{kind}.f32"
    W = np.fromfile(p, dtype=np.float32)
    if kind == "down":
        assert W.size == 2048 * 512, (p, W.size)
        return W.reshape(2048, 512)
    assert W.size == 512 * 2048, (p, W.size)
    return W.reshape(512, 2048)


def run_layer_model(L, model, N_gauss=320, seed=11):
    emb = torch.load(os.path.join(A3W, "calib_embed.pt"), map_location="cpu",
                     weights_only=True).to(torch.float32).numpy()
    g = np.fromfile(f"{RAW}/L{L:02d}_post_attention_norm.f32",
                    dtype=np.float32)
    W = np.fromfile(f"{RAW}/router_L{L:02d}.f32",
                    dtype=np.float32).reshape(256, 2048)
    if model == "proxy":
        rms = np.sqrt((emb.astype(np.float64) ** 2).mean(axis=1,
                                                         keepdims=True) + 1e-6)
        X = (emb / rms * g[None, :]).astype(np.float32)
        N = X.shape[0]
    elif model == "gauss":
        rng = np.random.default_rng(seed)
        s2 = load_router_scales(L)
        X = (rng.standard_normal((N_gauss, 2048)) *
             np.sqrt(s2)[None, :]).astype(np.float32)
        N = N_gauss
    else:
        raise ValueError(model)
    Xd = X.astype(np.float64)
    logits = Xd @ W.T.astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    top8 = np.argsort(P, axis=1)[:, ::-1][:, :8]
    p8 = np.take_along_axis(P, top8, axis=1)
    m8 = p8.sum(axis=1)
    w8 = p8 / m8[:, None]
    # coverage: every routed expert must have f32 weights
    union = sorted(set(top8.ravel().tolist()))
    missing = [e for e in union if not all(
        os.path.exists(f"{DEQ}/L{L:02d}_E{e:03d}_{k}.f32")
        for k in ("gate", "up", "down"))]
    if missing:
        raise SystemExit(f"L{L:02d}/{model}: {len(missing)} routed experts "
                         f"missing f32: {missing[:12]}")
    print(f"L{L:02d}/{model}: N={N} union={len(union)} "
          f"mass8={m8.mean():.4f}", flush=True)

    use = defaultdict(list)
    for t in range(N):
        for s in range(8):
            use[int(top8[t, s])].append((t, s))
    # per-slot expert outputs (N,8,2048) f32; accumulators f64
    Yslot = np.zeros((N, 8, 2048), np.float32)
    # z pooled over routed pairs; LIVE dims only (structurally dead dims --
    # gate+up row both exactly 0, esp. L00 -- reported separately, else they
    # masquerade as activation sparsity). N*8*512*4B = 5.2MB: store all z.
    Zall = np.zeros((N, 8, 512), np.float32)
    Dead = np.zeros((N, 8, 512), bool)
    for e, slots in sorted(use.items()):
        G, U, D = load_exp(L, e, "gate"), load_exp(L, e, "up"), \
            load_exp(L, e, "down")
        assert np.isfinite(G).all() and np.isfinite(U).all() \
            and np.isfinite(D).all()
        dead_e = (np.linalg.norm(G, axis=1) == 0) & \
            (np.linalg.norm(U, axis=1) == 0)
        toks = np.array([t for t, s in slots])
        Xe = X[toks]  # f32, routed tokens only
        gv = Xe @ G.T
        uv = Xe @ U.T
        z = (gv / (1.0 + np.exp(-gv))) * uv
        Y = z @ D.T
        for i, (t, s) in enumerate(slots):
            Yslot[t, s] = Y[i]
            Zall[t, s] = z[i]
            Dead[t, s] = dead_e
        del G, U, D, Xe, gv, uv, z, Y
    dead_frac = float(Dead.mean())
    print(f"  structural dead-dim frac (routed pairs): {dead_frac:.4f}",
          flush=True)
    Yd = Yslot.astype(np.float64)
    R8 = (Yd * w8[:, :, None]).sum(axis=1)
    nR8 = np.linalg.norm(R8, axis=1)
    assert (nR8 > 0).all()
    out = {"layer": L, "model": model, "N": N, "union": len(union),
           "dead_frac": dead_frac,
           "mass8": [float(m8.mean()), float(m8.min()), float(m8.max())],
           "rank_mass": [float(np.take_along_axis(
               P, top8[:, [r]], 1).mean()) for r in range(8)]}
    # Rk errors (renormed = deployable) + raw-scale diagnostic
    print(f"  {'k':>2s} {'mean':>7s} {'p50':>7s} {'p99':>7s} {'max':>7s} "
          f"{'rawmean':>8s} (rel ||R8-Rk||/||R8||)", flush=True)
    errs = {}
    rawerrs = {}
    for k in range(1, 8):
        pk = p8[:, :k]
        wk = pk / pk.sum(axis=1, keepdims=True)
        Rk = (Yd[:, :k] * wk[:, :, None]).sum(axis=1)
        rel = np.linalg.norm(R8 - Rk, axis=1) / nR8
        errs[k] = rel
        Rkraw = (Yd[:, :k] * p8[:, :k][:, :, None]).sum(axis=1)
        raw = np.linalg.norm(R8 - Rkraw, axis=1) / nR8
        rawerrs[k] = raw
        print(f"  {k:2d} {rel.mean():7.4f} {np.median(rel):7.4f} "
              f"{np.quantile(rel, .99):7.4f} {rel.max():7.4f} "
              f"{raw.mean():8.4f}", flush=True)
    out["relerr"] = {str(k): [float(errs[k].mean()),
                              float(np.median(errs[k])),
                              float(np.quantile(errs[k], 0.99)),
                              float(errs[k].max())] for k in errs}
    out["relerr_raw"] = {str(k): float(rawerrs[k].mean()) for k in rawerrs}
    # marginal rank contribution: renormed weight, norm share, LOO error
    wmean = [float(w8[:, r].mean()) for r in range(8)]
    slotnorm = np.linalg.norm(Yd, axis=2)  # (N,8) ||E_r||
    wslot = w8 * slotnorm
    nshare = (wslot / wslot.sum(axis=1, keepdims=True)).mean(axis=0)
    loo = []
    for r in range(8):
        keep = [i for i in range(8) if i != r]
        pk = p8[:, keep]
        wk = pk / pk.sum(axis=1, keepdims=True)
        Rloo = (Yd[:, keep] * wk[:, :, None]).sum(axis=1)
        loo.append(float((np.linalg.norm(R8 - Rloo, axis=1) / nR8).mean()))
    out["rank_w"] = wmean
    out["rank_normshare"] = [float(v) for v in nshare]
    out["rank_loo"] = loo
    print("  rank: " + " ".join(f"{r + 1}:w{wmean[r]:.3f}/n{nshare[r]:.3f}/"
                                f"loo{loo[r]:.4f}" for r in range(8)),
          flush=True)
    # expert-output redundancy: cosine of routed outputs (sampled pairs)
    rng = np.random.default_rng(3)
    ti = rng.integers(0, N, 400)
    a = rng.integers(0, 8, 400)
    b = (a + rng.integers(1, 8, 400)) % 8
    A = Yd[ti, a]
    B = Yd[ti, b]
    cos = ((A * B).sum(1) / np.linalg.norm(A, axis=1) /
           np.linalg.norm(B, axis=1))
    out["routed_cos"] = [float(cos.mean()), float(np.median(np.abs(cos))),
                         float(np.abs(cos).max())]
    print(f"  routed-output cos: mean={cos.mean():+.4f} "
          f"med|.|={np.median(np.abs(cos)):.4f} max|.|={np.abs(cos).max():.4f}",
          flush=True)
    # z-stats EXACT, pooled routed pairs, LIVE dims only
    Z = Zall.reshape(-1, 512).astype(np.float64)
    M = ~Dead.reshape(-1, 512)
    Zl = Z[M]
    zrms = float(np.sqrt((Zl ** 2).mean()))
    az = np.abs(Zl)
    sp = {t: float((az < t * zrms).mean())
          for t in (0.01, 0.05, 0.10, 0.25, 0.50)}
    # block sparsity on live dims, packed (dead dims excluded from blocks)
    blk = {}
    for b in (16, 32, 64):
        nb = 0
        nzb = 0
        for row, m in zip(np.abs(Z), M):
            lv = row[m]
            nfull = (len(lv) // b) * b
            if nfull == 0:
                continue
            zb = lv[:nfull].reshape(-1, b).max(axis=1)
            nb += len(zb)
            nzb += int((zb < 0.05 * zrms).sum())
        blk[b] = float(nzb / nb) if nb else -1.0
    out["z_live"] = {"rms": zrms, "live_vals": int(Zl.size),
                     "nearzero": {str(k): v for k, v in sp.items()},
                     "block": {str(k): v for k, v in blk.items()}}
    print(f"  z(live): rms={zrms:.4f} P<1%={sp[0.01]:.4f} "
          f"P<5%={sp[0.05]:.4f} P<10%={sp[0.10]:.4f} "
          f"P<25%={sp[0.25]:.4f} P<50%={sp[0.50]:.4f} "
          f"b16={blk[16]:.4f} b32={blk[32]:.4f} b64={blk[64]:.4f}",
          flush=True)
    # width energy from ROUTED per-dim energy, PER EXPERT (dims are not
    # comparable across experts), then averaged over routed experts
    ew = {w: [] for w in (448, 384, 256, 128)}
    live_counts = []
    for e, slots in sorted(use.items()):
        rows = np.array([Zall[t, s] for t, s in slots], dtype=np.float64)
        ez2 = (rows ** 2).mean(axis=0)
        tot = ez2.sum()
        if tot <= 0:
            continue
        cs = np.sort(ez2 / tot)[::-1].cumsum()
        for w in ew:
            ew[w].append(float(cs[w - 1]))
        live_counts.append(int((ez2 > 0).sum()))
    ewidth = {w: [float(np.mean(ew[w])), float(np.min(ew[w])),
                  float(np.max(ew[w]))] for w in ew}
    out["width_routed"] = {
        "energy_mean_min_max": {str(k): v for k, v in ewidth.items()},
        "n_experts": len(ew[256]),
        "live_dims_mean": float(np.mean(live_counts))}
    print("  width(per-expert routed E[z^2], mean/min/max): " +
          " ".join(f"e{w}={ewidth[w][0]:.4f}" for w in (448, 384, 256, 128)) +
          f" livedims={np.mean(live_counts):.1f}", flush=True)
    json.dump(out, open(os.path.join(
        RES, f"forward_L{L:02d}_{model}.json"), "w"), indent=1)
    return out


def main():
    L = int(sys.argv[1])
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else ["proxy"]
    Ng = int(sys.argv[3]) if len(sys.argv) > 3 else 320
    for m in models:
        run_layer_model(L, m, Ng)


if __name__ == "__main__":
    main()
