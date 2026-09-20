"""S2-Agent2 03: R8/R4/R2 + D4/D2 on proxy-REAL L00 activations.

Same math as salvaged 03 (exact forward, f32 math, f64 accum, renormed
deployable truncation), shared expert from s2 weights (bf16 HF).
Adds: kept-weight rescale factors, corr(|D|, omitted mass), cos(D,Rk).
Output: weights/outputs_L00.npz, results_topk.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
A3 = os.path.join(WS, "probes/agent3_funcmoe/weights")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights")
DEQ = "/tmp/agent1_f32"


def load_exp(L, e, kind):
    W = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_{kind}.f32", dtype=np.float32)
    return W.reshape(512, 2048) if kind != "down" else W.reshape(2048, 512)


def expert_out(X, G, U, D):
    gv = X @ G.T
    uv = X @ U.T
    z = (gv / (1.0 + np.exp(-gv))) * uv
    return z @ D.T


def main():
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0]
    res = {}
    for L in layers:
        X = np.load(os.path.join(A3, f"inputs_L{L:02d}.npy")).astype(np.float32)
        N = X.shape[0]
        r = np.load(os.path.join(A3, f"routing_L{L:02d}.npz"))
        top8, w8, logits = r["top8"], r["weights"].astype(np.float64), r["logits"]
        P = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        m4 = P[np.arange(N)[:, None], top8[:, :4]].sum(axis=1, keepdims=True)
        m2 = P[np.arange(N)[:, None], top8[:, :2]].sum(axis=1, keepdims=True)
        m8 = P[np.arange(N)[:, None], top8].sum(axis=1, keepdims=True)
        w4 = P[np.arange(N)[:, None], top8[:, :4]] / m4
        w2 = P[np.arange(N)[:, None], top8[:, :2]] / m2

        R8 = np.zeros((N, 2048), np.float64)
        R4 = np.zeros((N, 2048), np.float64)
        R2 = np.zeros((N, 2048), np.float64)
        use = defaultdict(list)
        for t in range(N):
            for s in range(8):
                use[int(top8[t, s])].append((t, s))
        enorms = {}
        Emean = {}
        for e, slots in sorted(use.items()):
            G, U, D = load_exp(L, e, "gate"), load_exp(L, e, "up"), load_exp(L, e, "down")
            Y = expert_out(X, G, U, D).astype(np.float64)
            enorms[e] = float(np.linalg.norm(Y, axis=1).mean())
            Emean[e] = Y.mean(axis=0)
            for (t, s) in slots:
                R8[t] += w8[t, s] * Y[t]
                if s < 4:
                    R4[t] += w4[t, s] * Y[t]
                if s < 2:
                    R2[t] += w2[t, s] * Y[t]
            del G, U, D, Y
        Gs = torch.load(os.path.join(OUT, f"L{L:02d}_shexp_gate.pt"),
                        map_location="cpu", weights_only=True).to(torch.float32).numpy()
        Us = torch.load(os.path.join(OUT, f"L{L:02d}_shexp_up.pt"),
                        map_location="cpu", weights_only=True).to(torch.float32).numpy()
        Ds = torch.load(os.path.join(OUT, f"L{L:02d}_shexp_down.pt"),
                        map_location="cpu", weights_only=True).to(torch.float32).numpy()
        ws = torch.load(os.path.join(OUT, f"L{L:02d}_shexp_inp.pt"),
                        map_location="cpu", weights_only=True).to(torch.float32).numpy().reshape(-1)
        S = expert_out(X, Gs, Us, Ds).astype(np.float64)
        sg = 1.0 / (1.0 + np.exp(-(X.astype(np.float64) @ ws.astype(np.float64))))
        S = S * sg[:, None]
        B = R8 + S
        D4, D2 = R8 - R4, R8 - R2
        nR8 = np.linalg.norm(R8, axis=1)
        nB = np.linalg.norm(B, axis=1)
        rD4 = np.linalg.norm(D4, axis=1) / nR8
        rD2 = np.linalg.norm(D2, axis=1) / nR8
        bD4 = np.linalg.norm(D4, axis=1) / nB
        bD2 = np.linalg.norm(D2, axis=1) / nB
        E = np.stack([Emean[e] for e in sorted(Emean)])
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        C = En @ En.T
        off = C[np.triu_indices(len(E), 1)]
        np.savez_compressed(os.path.join(OUT, f"outputs_L{L:02d}.npz"),
                            R8=R8.astype(np.float32), R4=R4.astype(np.float32),
                            R2=R2.astype(np.float32), D4=D4.astype(np.float32),
                            D2=D2.astype(np.float32), S=S.astype(np.float32), X=X)
        om4 = 1 - (m4 / m8).ravel()
        om2 = 1 - (m2 / m8).ravel()
        cos4 = float(np.mean([(D4[t] @ R4[t]) / (np.linalg.norm(D4[t]) * np.linalg.norm(R4[t]) + 1e-30) for t in range(N)]))
        cos2 = float(np.mean([(D2[t] @ R2[t]) / (np.linalg.norm(D2[t]) * np.linalg.norm(R2[t]) + 1e-30) for t in range(N)]))
        res[L] = {
            "N": N, "n_experts": len(use),
            "relD4_R8": [float(rD4.mean()), float(rD4.min()), float(rD4.max())],
            "relD2_R8": [float(rD2.mean()), float(rD2.min()), float(rD2.max())],
            "relD4_block": [float(bD4.mean()), float(bD4.min()), float(bD4.max())],
            "relD2_block": [float(bD2.mean()), float(bD2.min()), float(bD2.max())],
            "omit4_within8": float(om4.mean()), "omit2_within8": float(om2.mean()),
            "rescale4_m8overm4": float((m8 / m4).mean()),
            "rescale2_m8overm2": float((m8 / m2).mean()),
            "corr_absD4_omit4": float(np.corrcoef(np.linalg.norm(D4, axis=1), om4)[0, 1]),
            "corr_absD2_omit2": float(np.corrcoef(np.linalg.norm(D2, axis=1), om2)[0, 1]),
            "cos_D4_R4": cos4, "cos_D2_R2": cos2,
            "mass8": float(m8.mean()),
            "norm_R8": float(nR8.mean()), "norm_R4": float(np.linalg.norm(R4, axis=1).mean()),
            "norm_R2": float(np.linalg.norm(R2, axis=1).mean()),
            "norm_S": float(np.linalg.norm(S, axis=1).mean()),
            "norm_B": float(nB.mean()), "norm_X": float(np.linalg.norm(X, axis=1).mean()),
            "sig_gate": [float(sg.mean()), float(sg.min()), float(sg.max())],
            "expert_cos": [float(off.mean()), float(np.median(np.abs(off))), float(np.abs(off).max())],
            "expert_norm_ratio_maxmin": float(max(enorms.values()) / min(enorms.values())),
        }
        print(f"L{L:02d}: |D4|/|R8|={rD4.mean():.4f} |D2|/|R8|={rD2.mean():.4f} "
              f"|D4|/|B|={bD4.mean():.4f} |D2|/|B|={bD2.mean():.4f} "
              f"omit4={om4.mean():.4f} omit2={om2.mean():.4f} "
              f"Ecos={off.mean():+.3f}/|{np.median(np.abs(off)):.3f}", flush=True)
    json.dump(res, open(os.path.join(HERE, "results_topk.json"), "w"), indent=1)
    print("OK s2_03", flush=True)


if __name__ == "__main__":
    main()
