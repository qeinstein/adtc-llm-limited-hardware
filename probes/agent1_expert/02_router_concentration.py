"""Agent1 probe 02: router concentration from REAL router weights + calibrated inputs.

REAL inputs: all 40 F32 routers (256x2048), all 40 post-attention RMSNorm scales,
  2016-token REAL route corpus (selection frequencies).
ASSUMPTION (labeled): router input direction isotropic; second moment calibrated
  by the real RMSNorm scales (x_i = g_i * z_i, z~N(0,1) -> RMS(x)=rms(g)~1).
  Under this model, measured concentration is a LOWER bound on real concentration
  (real hidden states are anisotropic; trained routers discriminate within the
  occupied subspace, killing off-subspace logit noise). So:
    high isotropic concentration => real is even HIGHER (KEEP K-reduction)
    low isotropic concentration  => INCONCLUSIVE (cannot KILL; need real acts)

Also tests: row-norm/selection-frequency correlation (REAL x REAL), router
  geometry (row norms, cosines, effective rank).
"""
import json
import math
import os
from collections import Counter

import numpy as np

RAW = "/tmp/agent1_raw"
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = ("/home/fluxx/Workspace/adtc-llm-native-sparse/research/native_sparse_experiments"
          "/results/phase5e_route_corpus_v1/route_corpus.jsonl")

rng = np.random.default_rng(0)
S = 256  # samples per layer

recs = [json.loads(l) for l in open(CORPUS)]
selfreq = []
for L in range(40):
    c = Counter()
    for r in recs:
        c.update(r["layers"][L])
    tot = sum(c.values())
    selfreq.append(np.array([c.get(e, 0) / tot for e in range(256)]))

print(f"{'L':>3s} {'rms(g)':>7s} {'|row|med':>8s} {'|row|max/min':>11s} {'cosmed':>7s} "
      f"{'erank':>6s} {'t1':>6s} {'t2':>6s} {'t4':>6s} {'t6':>6s} {'t8':>6s} {'corr':>6s}")
cum_top = {1: [], 2: [], 4: [], 6: [], 8: []}
for L in range(40):
    W = np.fromfile(f"{RAW}/router_L{L:02d}.f32", dtype=np.float32).reshape(256, 2048)
    g = np.fromfile(f"{RAW}/L{L:02d}_post_attention_norm.f32", dtype=np.float32)
    assert np.isfinite(W).all() and np.isfinite(g).all()
    assert W.shape == (256, 2048) and g.shape == (2048,)
    rn = np.linalg.norm(W, axis=1)
    # pairwise cosine on a 64-expert subsample (exact enough, fast)
    idx = rng.choice(256, 64, replace=False)
    Wn = W[idx] / rn[idx][:, None]
    C = Wn @ Wn.T
    cosmed = float(np.median(np.abs(C[np.triu_indices(64, 1)])))
    # effective rank of router
    s = np.linalg.svd(W, compute_uv=False)
    p = (s ** 2) / (s ** 2).sum()
    erank = float(np.exp(-(p * np.log(p + 1e-30)).sum()))
    # isotropic-calibrated mass
    Z = rng.standard_normal((S, 2048)).astype(np.float64)
    X = Z * g[None, :]
    logits = X @ W.T  # (S,256)
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    Ps = np.sort(P, axis=1)[:, ::-1]
    cum = Ps.cumsum(axis=1).mean(axis=0)
    for k in cum_top:
        cum_top[k].append(float(cum[k - 1]))
    # row-norm vs REAL selection frequency correlation
    corr = float(np.corrcoef(rn, selfreq[L])[0, 1])
    print(f"{L:3d} {float(np.sqrt((g**2).mean())):7.3f} {float(np.median(rn)):8.3f} "
          f"{float(rn.max()/rn.min()):11.2f} {cosmed:7.4f} {erank:6.1f} {cum[0]:6.3f} "
          f"{cum[1]:6.3f} {cum[3]:6.3f} {cum[5]:6.3f} {cum[7]:6.3f} {corr:6.3f}")

print("\nmean cumulative top-K mass (isotropic-calibrated, LOWER-bound prior):")
for k in (1, 2, 4, 6, 8):
    v = np.array(cum_top[k])
    print(f"  top-{k}: mean={v.mean():.4f} min={v.min():.4f} (L{int(v.argmin())}) max={v.max():.4f} (L{int(v.argmax())})")
print("\nomitted-mass if K cut (mean over layers):")
for k in (6, 4, 2):
    omit = 1 - np.array(cum_top[k])
    print(f"  K={k}: omitted={omit.mean():.4f} (min {omit.min():.4f} / max {omit.max():.4f})")
