"""S2-Agent2 01: routing mass concentration across ALL 40 layers (local-only).

Proxy inputs x = RMSNorm(emb)*g_L from REAL bf16 calib embeds (N=320) +
REAL F32 routers, exact softmax/top-8 semantics (salvaged 02 methodology).
No network. Decides whether L00 is representative for the deep dive.
Output: results_routing40.json
"""
import json
import os

import numpy as np
import torch

WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
A3 = os.path.join(WS, "probes/agent3_funcmoe/weights")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "/tmp/agent1_raw"

emb = torch.load(os.path.join(A3, "calib_embed.pt"), map_location="cpu",
                 weights_only=True).to(torch.float32).numpy()
N = emb.shape[0]
rms = np.sqrt((emb.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6)
emb_n = (emb / rms).astype(np.float64)

res = {}
for L in range(40):
    W = np.fromfile(f"{RAW}/router_L{L:02d}.f32", dtype=np.float32).reshape(256, 2048)
    g = np.fromfile(f"{RAW}/L{L:02d}_post_attention_norm.f32", dtype=np.float32)
    X = emb_n * g[None, :].astype(np.float64)
    logits = X @ W.T.astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    Ps = np.sort(P, axis=1)[:, ::-1]
    cum = Ps.cumsum(axis=1).mean(axis=0)
    top8 = np.argsort(P, axis=1)[:, ::-1][:, :8]
    m8 = P[np.arange(N)[:, None], top8].sum(axis=1)
    m4 = P[np.arange(N)[:, None], top8[:, :4]].sum(axis=1)
    m2 = P[np.arange(N)[:, None], top8[:, :2]].sum(axis=1)
    res[L] = {
        "mass1": float(cum[0]), "mass2": float(cum[1]),
        "mass4": float(cum[3]), "mass8": float(cum[7]),
        "omit4_within8": float((1 - m4 / m8).mean()),
        "omit2_within8": float((1 - m2 / m8).mean()),
        "union8": int(len(set(top8.ravel().tolist()))),
    }
    print(f"L{L:02d}: mass1={cum[0]:.3f} mass2={cum[1]:.3f} mass4={cum[3]:.3f} "
          f"mass8={cum[7]:.3f} omit4={1-(m4/m8).mean():.3f} omit2={1-(m2/m8).mean():.3f} "
          f"union={res[L]['union8']}", flush=True)
json.dump(res, open(os.path.join(HERE, "results_routing40.json"), "w"), indent=1)
print("OK s2_01", flush=True)
