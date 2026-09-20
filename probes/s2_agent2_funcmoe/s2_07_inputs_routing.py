"""S2-Agent2 07: proxy inputs + routing for extension layers (local-only).

Same math as s2_01 (RMSNorm bf16 calib embeds * REAL post-attn norm g_L,
REAL F32 routers, exact softmax/top-8, f64). Saves A3-02-compatible
inputs_Lxx.npy + routing_Lxx.npz into THIS dir's weights/ (never A3 dir).
Usage: python3 s2_07_inputs_routing.py 5,20,30
"""
import json
import os
import sys

import numpy as np
import torch

WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
A3 = os.path.join(WS, "probes/agent3_funcmoe/weights")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights")
os.makedirs(OUT, exist_ok=True)
RAW = "/tmp/agent1_raw"


def main():
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [5, 20, 30]
    emb = torch.load(os.path.join(A3, "calib_embed.pt"), map_location="cpu",
                     weights_only=True).to(torch.float32).numpy()
    N = emb.shape[0]
    rms = np.sqrt((emb.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6)
    emb_n = (emb / rms).astype(np.float64)
    for L in layers:
        W = np.fromfile(f"{RAW}/router_L{L:02d}.f32", dtype=np.float32).reshape(256, 2048)
        g = np.fromfile(f"{RAW}/L{L:02d}_post_attention_norm.f32", dtype=np.float32)
        X = emb_n * g[None, :].astype(np.float64)
        np.save(os.path.join(OUT, f"inputs_L{L:02d}.npy"), X.astype(np.float32))
        logits = X @ W.T.astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        top8 = np.argsort(P, axis=1)[:, ::-1][:, :8]
        pw = np.take_along_axis(P, top8, axis=1)
        pw = pw / pw.sum(axis=1, keepdims=True)
        np.savez_compressed(os.path.join(OUT, f"routing_L{L:02d}.npz"),
                            top8=top8, weights=pw.astype(np.float32),
                            logits=logits.astype(np.float32))
        Ps = np.sort(P, axis=1)[:, ::-1]
        cum = Ps.cumsum(axis=1).mean(axis=0)
        print(f"L{L:02d}: N={N} top1/2/4/8 mass={cum[0]:.3f}/{cum[1]:.3f}/{cum[3]:.3f}/{cum[7]:.3f} "
              f"union={len(set(top8.ravel().tolist()))}", flush=True)
    print("OK s2_07", flush=True)


if __name__ == "__main__":
    main()
