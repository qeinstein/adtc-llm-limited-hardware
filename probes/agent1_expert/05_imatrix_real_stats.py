"""Agent1 probe 05: REAL activation statistics from Unsloth imatrix calibration data.

File: imatrix_unsloth.gguf_file (52 MB, this model's own quant calibration:
  510 entries x 76 chunks of unsloth_calibration_Qwen3.5-35B-A3B.txt).
Each weight tensor has `.in_sum2` (sum of input x^2 per input channel, F32)
and `.counts` (tokens observed). So E[x_i^2] = in_sum2/counts is the REAL
per-channel input second moment -- diagonal only (no correlations).

REAL deliverables (no inference needed):
 A. router input E[x^2] per channel per layer -> concentration under the
    channel-independent model (strictly better than isotropic; correlations
    still missing -> labeled prior, not measurement)
 B. down_exps E[z^2] per intermediate dim per expert -> REAL width importance,
    dead-dim census, per-layer/per-expert variation  => WIDTH VERDICT
 C. consistency: count-weighted gate-input scales vs router-input scales
    (same physical vector; must match if format is understood)
 D. per-expert counts -> REAL selection frequency; cross-check vs route corpus
 E. predicted gate pre-activation Var per dim (weights x real scales) for
    fetched experts; dead-at-gate census
 F. predicted vs measured E[z^2]: validates Gaussian+independence model,
    reveals silu suppression structure

Caveats: calibration text != clinical EN/SW; diagonal only; count-weighted.
"""
import json
import os
import struct
from collections import Counter

import numpy as np

IMA = "/tmp/imatrix.gguf"
RAW = "/tmp/agent1_raw"
DEQ = "/tmp/agent1_f32"
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = ("/home/fluxx/Workspace/adtc-llm-native-sparse/research/native_sparse_experiments"
          "/results/phase5e_route_corpus_v1/route_corpus.jsonl")


def load_imatrix():
    d = open(IMA, "rb").read()
    off = 4
    off += 4
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
    infos = []
    for _ in range(nt):
        nm, off = rs(off)
        (nd,) = struct.unpack_from("<i", d, off); off += 4
        sh = [struct.unpack_from("<q", d, off + 8 * i)[0] for i in range(nd)]
        off += 8 * nd
        (ty,) = struct.unpack_from("<i", d, off); off += 4
        (to,) = struct.unpack_from("<q", d, off); off += 8
        assert ty == 0, (nm, ty)
        infos.append((nm, sh, to))
    ds = (off + 31) // 32 * 32
    out = {}
    for nm, sh, to in infos:
        n = 1
        for v in sh:
            n *= v
        a = np.frombuffer(d, dtype=np.float32, count=n, offset=ds + to).copy()
        # GGUF ne0 fastest: shape reversed for numpy (row-major last-dim-fastest)
        out[nm] = a.reshape(tuple(sh[::-1]) if len(sh) > 1 else sh)
    return out


def main():
    M = load_imatrix()
    print(f"loaded {len(M)} imatrix entries")

    # ---- D. per-expert counts + cross-check vs route corpus ----
    recs = [json.loads(l) for l in open(CORPUS)]
    print("\n[D] per-expert selection: imatrix counts vs route-corpus frequency")
    corrs = []
    for L in [0, 10, 20, 30, 39]:
        cnt = M[f"blk.{L}.ffn_gate_exps.weight.counts"].reshape(256)
        c = Counter()
        for r in recs:
            c.update(r["layers"][L])
        fq = np.array([c.get(e, 0) for e in range(256)], dtype=float)
        fq /= fq.sum(); p = cnt / cnt.sum()
        corr = float(np.corrcoef(p, fq)[0, 1])
        corrs.append(corr)
        print(f"  L{L:02d}: imatrix_tokens_total={int(cnt.sum())} corr_with_corpus={corr:.3f} "
              f"top1_share={p.max():.4f} frac_never={float((cnt==0).mean()):.4f}")
    print(f"  mean corr={np.mean(corrs):.3f} (different prompts; >0.3 = same popularity structure)")

    # ---- C. consistency: gate-input vs router-input scales ----
    print("\n[C] consistency: count-weighted gate-input E[x^2] vs router-input E[x^2]")
    for L in [0, 10, 20, 30]:
        r_s2 = M[f"blk.{L}.ffn_gate_inp.weight.in_sum2"]
        r_c = float(M[f"blk.{L}.ffn_gate_inp.weight.counts"].reshape(-1)[0])
        g_s2 = M[f"blk.{L}.ffn_gate_exps.weight.in_sum2"]  # (256, 2048)
        g_c = M[f"blk.{L}.ffn_gate_exps.weight.counts"].reshape(256)
        r = r_s2 / r_c
        gw = g_s2.sum(0) / g_c.sum()  # in_sum2 already sums; divide by total
        rel = float(np.linalg.norm(gw - r) / np.linalg.norm(r))
        print(f"  L{L:02d}: rel-diff={rel:.4f} router_rms={float(np.sqrt(r.mean())):.3f} "
              f"(0.00x = same vector, format understood)")

    # ---- A. router input scales + channel-independent concentration ----
    print("\n[A] router input scales (REAL E[x^2]) + channel-indep top-K mass prior")
    rng = np.random.default_rng(7)
    S = 256
    print(f"  {'L':>3s} {'rms':>6s} {'max/med':>8s} {'outl>10x':>9s} {'t1':>6s} {'t2':>6s} {'t4':>6s} {'t8':>6s}")
    for L in range(40):
        r_s2 = M[f"blk.{L}.ffn_gate_inp.weight.in_sum2"]
        r_c = float(M[f"blk.{L}.ffn_gate_inp.weight.counts"].reshape(-1)[0])
        ex2 = (r_s2 / r_c).astype(np.float64)
        s = np.sqrt(np.maximum(ex2, 0))
        W = np.fromfile(f"{RAW}/router_L{L:02d}.f32", dtype=np.float32).reshape(256, 2048).astype(np.float64)
        Z = rng.standard_normal((S, 2048))
        X = Z * s[None, :]
        logits = X @ W.T
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits); P /= P.sum(1, keepdims=True)
        cum = np.sort(P, 1)[:, ::-1].cumsum(1).mean(0)
        med = float(np.median(s))
        outl = float((s > 10 * med).mean())
        print(f"  {L:3d} {float(np.sqrt(ex2.mean())):6.3f} {float(s.max()/med):8.1f} {outl:10.4f} "
              f"{cum[0]:6.3f} {cum[1]:6.3f} {cum[3]:6.3f} {cum[7]:6.3f}")

    # ---- B. down E[z^2]: REAL width importance ----
    print("\n[B] REAL intermediate-dim energy from down_exps E[z^2] (per expert, 256 experts/layer)")
    print(f"  {'L':>3s} {'e448':>6s} {'e384':>6s} {'e256':>6s} {'e128':>6s} "
          f"{'dead1e-4':>8s} {'dead1e-3':>8s} (fractions of 512 dims, mean over experts)")
    for L in range(40):
        z_s2 = M[f"blk.{L}.ffn_down_exps.weight.in_sum2"]  # (256, 512)
        z_c = M[f"blk.{L}.ffn_down_exps.weight.counts"].reshape(256)
        ez2 = z_s2 / np.maximum(z_c, 1)[:, None]
        ez2 = np.maximum(ez2, 0)
        # normalize per expert, sort desc, cumulative energy
        tot = ez2.sum(1, keepdims=True)
        tot[tot == 0] = 1
        pn = ez2 / tot
        cs = np.sort(pn, 1)[:, ::-1].cumsum(1)
        e448 = float(cs[:, 447].mean()); e384 = float(cs[:, 383].mean())
        e256 = float(cs[:, 255].mean()); e128 = float(cs[:, 127].mean())
        mean_dim = ez2.mean(1, keepdims=True)
        d4 = float((ez2 < 1e-4 * mean_dim).mean())
        d3 = float((ez2 < 1e-3 * mean_dim).mean())
        print(f"  {L:3d} {e448:6.4f} {e384:6.4f} {e256:6.4f} {e128:6.4f} {d4:8.4f} {d3:8.4f}")


if __name__ == "__main__":
    main()
