"""Agent1 probe 06: channel-independent model with REAL scales + REAL weights.

REAL inputs: imatrix E[x^2] per channel (router+gate inputs), imatrix E[z^2]
  per dim per expert (ALL 256 experts x 40 layers), exact dequant of 64 experts,
  all 40 F32 routers.
MODEL (labeled): channels independent Gaussian, x_i ~ N(0, s_i^2), s from imatrix.

Deliverables:
 A. predicted-vs-measured E[z^2] per dim (validates Gaussian+independence;
    mismatch => correlations matter)
 B. refined single-token SwiGLU sparsity w/ REAL scales (fetched experts):
    P(|z|<t), magnitude histogram, block sparsity 16/32/64
 C. per-expert output energy E[||E_e||^2] diagonal estimate (ALL experts via
    measured E[z^2] x mean down-col energy) + MoE(x)/x ratio estimate
 D. marginal expert contribution + removal error K=8..2 (diagonal approx:
    experts uncorrelated, alpha-E uncorrelated) => PRIOR for K-reduction
"""
import glob
import os
import struct
import sys

import numpy as np

IMA = "/tmp/imatrix.gguf"
RAW = "/tmp/agent1_raw"
DEQ = "/tmp/agent1_f32"
LAYERS = [0, 10, 20, 30]
rng = np.random.default_rng(11)


def load_imatrix(names):
    d = open(IMA, "rb").read()
    off = 4; off += 4
    nt = struct.unpack_from("<q", d, off)[0]; off += 8
    nkv = struct.unpack_from("<q", d, off)[0]; off += 8

    def rs(o):
        (n,) = struct.unpack_from("<q", d, o); o += 8
        return d[o:o + n].decode(), o + n

    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    for _ in range(nkv):
        _, off = rs(off)
        t = struct.unpack_from("<i", d, off)[0]; off += 4
        if t == 8:
            _, off = rs(off)
        elif t == 9:
            at = struct.unpack_from("<i", d, off)[0]; off += 4
            (n,) = struct.unpack_from("<q", d, off); off += 8
            if at == 8:
                for _ in range(n):
                    _, off = rs(off)
            else:
                off += SZ[at] * n
        else:
            off += SZ[t]
    infos = {}
    for _ in range(nt):
        nm, off = rs(off)
        (nd,) = struct.unpack_from("<i", d, off); off += 4
        sh = [struct.unpack_from("<q", d, off + 8 * i)[0] for i in range(nd)]
        off += 8 * nd
        (ty,) = struct.unpack_from("<i", d, off); off += 4
        (to,) = struct.unpack_from("<q", d, off); off += 8
        if nm in names:
            infos[nm] = (sh, to)
    ds = (off + 31) // 32 * 32
    out = {}
    for nm, (sh, to) in infos.items():
        n = 1
        for v in sh:
            n *= v
        a = np.frombuffer(d, dtype=np.float32, count=n, offset=ds + to).copy()
        out[nm] = a.reshape(tuple(sh[::-1]) if len(sh) > 1 else sh)
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    names = set()
    for L in LAYERS:
        names.add(f"blk.{L}.ffn_gate_exps.weight.in_sum2")
        names.add(f"blk.{L}.ffn_gate_exps.weight.counts")
        names.add(f"blk.{L}.ffn_down_exps.weight.in_sum2")
        names.add(f"blk.{L}.ffn_down_exps.weight.counts")
        names.add(f"blk.{L}.ffn_gate_inp.weight.in_sum2")
        names.add(f"blk.{L}.ffn_gate_inp.weight.counts")
    M = load_imatrix(names)

    S = 256  # MC samples
    print("== [A] predicted vs measured E[z^2] (per-dim, fetched experts) ==")
    print(f"  {'L':>3s} {'E':>3s} {'corr':>6s} {'slope':>6s} {'pred/mean':>9s}")
    for L in LAYERS:
        experts = sorted({int(os.path.basename(p).split("_")[1][1:])
                          for p in glob.glob(f"{DEQ}/L{L:02d}_E*_gate.f32")})
        gc = M[f"blk.{L}.ffn_gate_exps.weight.counts"].reshape(256)
        gs2 = M[f"blk.{L}.ffn_gate_exps.weight.in_sum2"]
        zc = M[f"blk.{L}.ffn_down_exps.weight.counts"].reshape(256)
        zs2 = M[f"blk.{L}.ffn_down_exps.weight.in_sum2"]
        for e in experts:
            ex2 = np.maximum(gs2[e] / max(gc[e], 1), 0)  # (2048,)
            G = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32", dtype=np.float32).reshape(512, 2048).astype(np.float64)
            U = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32", dtype=np.float32).reshape(512, 2048).astype(np.float64)
            vg = (G ** 2) @ ex2  # (512,) predicted Var(g_j)
            vu = (U ** 2) @ ex2
            # per-dim Gaussian MC for E[(silu(g)u)^2], g,u indep
            gg = rng.standard_normal((4096, 1)) * np.sqrt(np.maximum(vg, 0))[None, :]
            uu = rng.standard_normal((4096, 1)) * np.sqrt(np.maximum(vu, 0))[None, :]
            pred = ((silu(gg) * uu) ** 2).mean(0)
            meas = np.maximum(zs2[e] / max(zc[e], 1), 0)
            ok = (meas > 0) & (pred > 0)
            if ok.sum() > 32:
                corr = float(np.corrcoef(np.log(pred[ok]), np.log(meas[ok]))[0, 1])
                slope = float(np.polyfit(np.log(pred[ok]), np.log(meas[ok]), 1)[0])
            else:
                corr, slope = float("nan"), float("nan")
            print(f"  {L:3d} {e:3d} {corr:6.3f} {slope:6.3f} {float(pred.mean()/max(meas.mean(),1e-30)):9.3f}")

    print("\n== [B] refined sparsity w/ REAL scales (fetched experts) ==")
    print(f"  {'L':>3s} {'sp1%':>6s} {'sp5%':>6s} {'sp10%':>7s} {'sp25%':>7s} {'b16':>6s} {'b64':>6s} (mean over 16 experts)")
    for L in LAYERS:
        experts = sorted({int(os.path.basename(p).split("_")[1][1:])
                          for p in glob.glob(f"{DEQ}/L{L:02d}_E*_gate.f32")})
        gc = M[f"blk.{L}.ffn_gate_exps.weight.counts"].reshape(256)
        gs2 = M[f"blk.{L}.ffn_gate_exps.weight.in_sum2"]
        sps = []
        for e in experts:
            s = np.sqrt(np.maximum(gs2[e] / max(gc[e], 1), 0))
            G = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32", dtype=np.float32).reshape(512, 2048).astype(np.float64)
            U = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32", dtype=np.float32).reshape(512, 2048).astype(np.float64)
            X = rng.standard_normal((S, 2048)) * s[None, :]
            z = silu(X @ G.T) * (X @ U.T)
            zrms = float(np.sqrt((z ** 2).mean()))
            az = np.abs(z)
            sp = [float((az < t * zrms).mean()) for t in (0.01, 0.05, 0.10, 0.25)]
            b16 = float((az.reshape(S, 32, 16).max(2) < 0.05 * zrms).mean())
            b64 = float((az.reshape(S, 8, 64).max(2) < 0.05 * zrms).mean())
            sps.append(sp + [b16, b64])
        a = np.mean(sps, 0)
        print(f"  {L:3d} {a[0]:6.3f} {a[1]:6.3f} {a[2]:7.3f} {a[3]:7.3f} {a[4]:6.4f} {a[5]:6.4f}")

    print("\n== [C/D] diagonal MoE/x ratio + removal error (channel-indep, REAL router+scales) ==")
    print("  experts assumed mutually uncorrelated; alpha-E uncorrelated (OPTIMISTIC for K-cut).")
    for L in LAYERS:
        r_s2 = M[f"blk.{L}.ffn_gate_inp.weight.in_sum2"]
        r_c = float(M[f"blk.{L}.ffn_gate_inp.weight.counts"].reshape(-1)[0])
        s = np.sqrt(np.maximum(r_s2 / r_c, 0)).astype(np.float64)
        W = np.fromfile(f"{RAW}/router_L{L:02d}.f32", dtype=np.float32).reshape(256, 2048).astype(np.float64)
        X = rng.standard_normal((S, 2048)) * s[None, :]
        logits = X @ W.T
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits); P /= P.sum(1, keepdims=True)
        top8 = np.argsort(P, 1)[:, ::-1][:, :8]
        # per-expert output energy: E[||y||^2] ~= sum_j E[z_j^2] * mean_col_energy
        # mean down-col energy from fetched experts of THIS layer
        experts = sorted({int(os.path.basename(p).split("_")[1][1:])
                          for p in glob.glob(f"{DEQ}/L{L:02d}_E*_down.f32")})
        col_e = []
        for e in experts:
            D = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_down.f32", dtype=np.float32).reshape(2048, 512).astype(np.float64)
            col_e.append((D ** 2).sum(0))
        colmean = np.mean(col_e, 0)  # (512,)
        zc = M[f"blk.{L}.ffn_down_exps.weight.counts"].reshape(256)
        zs2 = M[f"blk.{L}.ffn_down_exps.weight.in_sum2"].astype(np.float64)
        ez2 = np.maximum(zs2 / np.maximum(zc, 1)[:, None], 0)
        Ee = (ez2 * colmean[None, :]).sum(1)  # (256,) per-expert output energy
        Ex = float((X ** 2).sum(1).mean())
        # renorm'd alphas
        A = np.take_along_axis(P, top8, 1)
        A = A / A.sum(1, keepdims=True)
        Eesel = Ee[top8]
        moe_e = (A ** 2 * Eesel).sum(1).mean()
        print(f"  L{L:02d}: E||x||^2={Ex:.1f} E||MoE||^2~{moe_e:.3f} ratio={float(np.sqrt(moe_e/Ex)):.4f} "
              f"(diagonal; |MoE|/|x|)")
        # removal error dropping ranks k+1..8 (renorm remaining)
        errs = {}
        for k in (6, 4, 2, 1):
            Ak = A[:, :k] / A[:, :k].sum(1, keepdims=True)
            # err^2 = E||full-trunc||^2 / E||full||^2 under uncorr experts
            full = (A ** 2 * Eesel).sum(1)
            wfull = A * np.sqrt(Eesel)
            wtr = np.zeros_like(wfull)
            wtr[:, :k] = Ak * np.sqrt(Eesel[:, :k])
            err2 = (((wfull - wtr) ** 2).sum(1) / np.maximum(full, 1e-30)).mean()
            errs[k] = float(np.sqrt(max(err2, 0)))
        print(f"       removal rel-err: " + "  ".join(f"K={k}:{errs[k]:.3f}" for k in (6, 4, 2, 1)))


if __name__ == "__main__":
    main()
