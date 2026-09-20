"""S2 Agent1 (top-k) script 01: proxy-REAL routing + mass concentration, all 40 layers.

Inputs (all local, zero network):
  - REAL bf16 calib embeds (320 clinical EN/SW tokens, agent3 01 output, in-repo)
  - REAL F32 routers (40x 256x2048, /tmp/agent1_raw)
  - REAL post-attention RMSNorm scales (40x 2048, /tmp/agent1_raw)
  - REAL route corpus (2016 tokens x 40 layers, in-repo) -- validation only
  - REAL imatrix router in_sum2/counts (/tmp/imatrix.gguf) -- validation only

Proxy inputs: X_L = RMSNorm(emb; eps=1e-6) * g_L  (agent2/agent3 methodology).
  For L00 the residual stream IS the embedding, so L00 is the most faithful
  layer; deeper layers miss attention/SSM sublayer outputs -- labeled PROXY.
Routing: probs = softmax(X @ W.T) over 256; top-8 by prob; w renormed
  (Qwen3-MoE / llama.cpp qwen3_next semantics; rowsum==1 asserted).

Outputs: results/routing_all40.json, results/union_curve.json
"""
import json
import os
import struct
import sys
from collections import Counter

import numpy as np
import torch

WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
HERE = os.path.join(WS, "probes/s2_agent1_topk")
RES = os.path.join(HERE, "results")
A3W = os.path.join(WS, "probes/agent3_funcmoe/weights")
RAW = "/tmp/agent1_raw"
IMA = "/tmp/imatrix.gguf"
CORPUS = os.path.join(WS, "research/native_sparse_experiments/results/"
                           "phase5e_route_corpus_v1/route_corpus.jsonl")


def load_imatrix_router():
    """Return {L: ex2 (2048,) float64} router input E[x^2] + counts."""
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
    want = {}
    for L in range(40):
        want[f"blk.{L}.ffn_gate_inp.weight.in_sum2"] = L
        want[f"blk.{L}.ffn_gate_inp.weight.counts"] = L
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
        assert ty == 0, (nm, ty)
        if nm in want:
            infos[nm] = (sh, to)
    assert len(infos) == 80, len(infos)
    ds = (off + 31) // 32 * 32
    out = {}
    for nm, (sh, to) in infos.items():
        n = 1
        for v in sh:
            n *= v
        a = np.frombuffer(d, dtype=np.float32, count=n,
                          offset=ds + to).copy()
        out[nm] = a.reshape(tuple(sh[::-1]) if len(sh) > 1 else sh)
    ex2 = {}
    for L in range(40):
        s2 = out[f"blk.{L}.ffn_gate_inp.weight.in_sum2"].astype(np.float64)
        c = float(out[f"blk.{L}.ffn_gate_inp.weight.counts"].reshape(-1)[0])
        ex2[L] = (s2 / c, c)
    return ex2


def main():
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 \
        else list(range(40))
    os.makedirs(RES, exist_ok=True)
    emb = torch.load(os.path.join(A3W, "calib_embed.pt"), map_location="cpu",
                     weights_only=True).to(torch.float32).numpy()
    N = emb.shape[0]
    assert emb.shape[1] == 2048, emb.shape
    print(f"N={N} calib tokens", flush=True)
    rms = np.sqrt((emb.astype(np.float64) ** 2).mean(axis=1, keepdims=True)
                  + 1e-6)
    emb_n = (emb / rms).astype(np.float64)

    recs = [json.loads(l) for l in open(CORPUS)]
    selfreq = []
    for L in range(40):
        c = Counter()
        for r in recs:
            c.update(r["layers"][L])
        tot = sum(c.values())
        selfreq.append(np.array([c.get(e, 0) / tot for e in range(256)]))
    print(f"corpus: {len(recs)} real tokens", flush=True)

    imex2 = load_imatrix_router()
    print("imatrix router scales loaded", flush=True)

    res = {}
    print(f"{'L':>3s} {'rmsX':>6s} {'m1':>6s} {'m2':>6s} {'m4':>6s} {'m8':>6s} "
          f"{'r58in8':>7s} {'union':>5s} {'logcorr':>7s} {'imrms':>6s} "
          f"{'imcorr':>7s} {'imrel':>6s}", flush=True)
    for L in layers:
        W = np.fromfile(f"{RAW}/router_L{L:02d}.f32",
                        dtype=np.float32).reshape(256, 2048)
        assert np.isfinite(W).all()
        g = np.fromfile(f"{RAW}/L{L:02d}_post_attention_norm.f32",
                        dtype=np.float32)
        assert np.isfinite(g).all() and (g != 0).all()
        X = emb_n * g[None, :]
        logits = X @ W.T.astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        assert np.isfinite(P).all()
        top8 = np.argsort(P, axis=1)[:, ::-1][:, :8]
        pw = np.take_along_axis(P, top8, axis=1)
        pw = pw / pw.sum(axis=1, keepdims=True)
        assert np.allclose(pw.sum(1), 1.0, atol=1e-6)
        Ps = np.sort(P, axis=1)[:, ::-1]
        cum = Ps.cumsum(axis=1).mean(axis=0)
        m8tok = Ps[:, :8].sum(axis=1)
        r58 = (Ps[:, 4:8].sum(axis=1) / m8tok).mean()  # ranks5-8 in-top8 share
        # rank masses (mean prob of rank r over 256) + within-8 shares
        rankmass = [float(Ps[:, r].mean()) for r in range(8)]
        w8share = [float((Ps[:, r] / m8tok).mean()) for r in range(8)]
        pf = np.zeros(256)
        for row in top8:
            for e in row:
                pf[e] += 1
        pf /= pf.sum()
        m = (pf > 0) | (selfreq[L] > 0)
        logcorr = float(np.corrcoef(np.log(pf[m] + 1e-9),
                                    np.log(selfreq[L][m] + 1e-9))[0, 1])
        # proxy 2nd moment vs REAL imatrix 2nd moment (same physical vector)
        pex2 = (X ** 2).mean(axis=0)
        iex2, cnt = imex2[L]
        imcorr = float(np.corrcoef(pex2, iex2)[0, 1])
        imrel = float(np.linalg.norm(pex2 - iex2) / np.linalg.norm(iex2))
        nunion = len(set(top8.ravel().tolist()))
        res[L] = {"N": N, "rmsX": float(np.sqrt((X ** 2).mean())),
                  "mass_top148": [float(cum[0]), float(cum[1]),
                                  float(cum[3]), float(cum[7])],
                  "mass_cum8": [float(cum[k]) for k in range(8)],
                  "rank_mass": rankmass, "rank_within8": w8share,
                  "mass8_mean": float(m8tok.mean()),
                  "mass8_min": float(m8tok.min()),
                  "mass8_max": float(m8tok.max()),
                  "ranks58_in8": float(r58),
                  "union320": nunion,
                  "corpus_logcorr": logcorr,
                  "imatrix_rms": float(np.sqrt(iex2.mean())),
                  "imatrix_corr": imcorr, "imatrix_reldiff": imrel,
                  "imatrix_tokens": cnt,
                  "top8": top8.tolist()}
        print(f"{L:3d} {res[L]['rmsX']:6.3f} {cum[0]:6.3f} {cum[1]:6.3f} "
              f"{cum[3]:6.3f} {cum[7]:6.3f} {r58:7.3f} {nunion:5d} "
              f"{logcorr:7.3f} {np.sqrt(iex2.mean()):6.3f} {imcorr:7.3f} "
              f"{imrel:6.3f}", flush=True)
    # union-vs-N curves for fetch planning (candidate R-layers)
    curves = {}
    for L in [10, 20, 30, 39]:
        t8 = np.array(res[L]["top8"])
        curves[L] = {n: len(set(t8[:n].ravel().tolist()))
                     for n in (32, 64, 96, 128, 160, 224, 320)}
        print(f"L{L:02d} union-vs-N: " +
              " ".join(f"{n}:{v}" for n, v in curves[L].items()), flush=True)
    json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != "top8"}
               for k, v in res.items()},
              open(os.path.join(RES, "routing_all40.json"), "w"), indent=1)
    # full top8 index kept separately for fetch planning
    np.savez_compressed(os.path.join(RES, "top8_all40.npz"),
                        **{f"L{L:02d}": np.array(res[L]["top8"]) for L in res})
    json.dump({str(k): v for k, v in curves.items()},
              open(os.path.join(RES, "union_curve.json"), "w"), indent=1)
    print("OK 01", flush=True)


if __name__ == "__main__":
    main()
