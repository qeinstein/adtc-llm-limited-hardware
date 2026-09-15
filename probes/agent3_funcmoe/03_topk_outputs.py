"""Agent3 probe 03: R8/R4/R2 + D4/D2 on proxy-REAL activations.

Per layer: stream REAL IQ2-dequantized experts, exact forward
  E_e(x) = Down(silu(Gate x) * Up x), float32 math, float64 accumulation.
  R8/R4/R2 with renormed weights (deployable truncation semantics);
  D4 = R8-R4, D2 = R8-R2; shared expert S(x) (bf16 weights).
Reports: rel residual norms (vs R8 and vs block R8+S), omitted-mass stats,
  per-expert output norms, pairwise expert-output cosines (redundancy check).
Opportunistic isotropic control: fraction of isotropic-top8 covered by the
  already-fetched proxy union; full isotropic R8 iff coverage allows.

Usage: python3 03_topk_outputs.py [0,10,20,30]
Output: weights/outputs_L{L}.npz, results_topk.json
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights")
DEQ = "/tmp/agent1_f32"
RAW = "/tmp/agent1_raw"


def load_exp(L, e, kind):
    p = f"{DEQ}/L{L:02d}_E{e:03d}_{kind}.f32"
    W = np.fromfile(p, dtype=np.float32)
    return W.reshape(512, 2048) if kind != "down" else W.reshape(2048, 512)


def expert_out(X, G, U, D):
    gv = X @ G.T
    uv = X @ U.T
    z = (gv / (1.0 + np.exp(-gv))) * uv
    return z @ D.T


def main():
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 10, 20, 30]
    res = {}
    for L in layers:
        X = np.load(os.path.join(OUT, f"inputs_L{L:02d}.npy")).astype(np.float32)
        N = X.shape[0]
        r = np.load(os.path.join(OUT, f"routing_L{L:02d}.npz"))
        top8, w8, logits = r["top8"], r["weights"].astype(np.float64), r["logits"]
        P = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        # renormed truncation weights from full probs (deployable)
        m4 = P[np.arange(N)[:, None], top8[:, :4]].sum(axis=1, keepdims=True)
        m2 = P[np.arange(N)[:, None], top8[:, :2]].sum(axis=1, keepdims=True)
        m8 = P[np.arange(N)[:, None], top8].sum(axis=1, keepdims=True)
        w4 = P[np.arange(N)[:, None], top8[:, :4]] / m4
        w2 = P[np.arange(N)[:, None], top8[:, :2]] / m2

        R8 = np.zeros((N, 2048), np.float64)
        R4 = np.zeros((N, 2048), np.float64)
        R2 = np.zeros((N, 2048), np.float64)
        # token->slots using expert e
        from collections import defaultdict
        use = defaultdict(list)
        for t in range(N):
            for s in range(8):
                use[int(top8[t, s])].append((t, s))
        enorms = {}
        Emean = {}  # mean output per expert (for cosine redundancy)
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
        # shared expert
        Gs = torch.load(os.path.join(OUT, f"L{L:02d}_shexp_gate.pt"),
                        map_location="cpu", weights_only=True).to(torch.float32).numpy()
        Us = torch.load(os.path.join(OUT, f"L{L:02d}_shexp_up.pt"),
                        map_location="cpu", weights_only=True).to(torch.float32).numpy()
        Ds = torch.load(os.path.join(OUT, f"L{L:02d}_shexp_down.pt"),
                        map_location="cpu", weights_only=True).to(torch.float32).numpy()
        ws = torch.load(os.path.join(OUT, f"L{L:02d}_shexp_inp.pt"),
                        map_location="cpu", weights_only=True).to(torch.float32).numpy()
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
        # expert redundancy: cosine of mean outputs
        E = np.stack([Emean[e] for e in sorted(Emean)])
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        C = En @ En.T
        off = C[np.triu_indices(len(E), 1)]
        np.savez_compressed(os.path.join(OUT, f"outputs_L{L:02d}.npz"),
                            R8=R8.astype(np.float32), R4=R4.astype(np.float32),
                            R2=R2.astype(np.float32), D4=D4.astype(np.float32),
                            D2=D2.astype(np.float32), S=S.astype(np.float32),
                            X=X)
        # isotropic control coverage
        rng = np.random.default_rng(7)
        g = np.fromfile(f"{RAW}/L{L:02d}_post_attention_norm.f32", dtype=np.float32)
        W = np.fromfile(f"{RAW}/router_L{L:02d}.f32", dtype=np.float32).reshape(256, 2048)
        Xi = (rng.standard_normal((N, 2048)) * g[None, :]).astype(np.float64)
        li = Xi @ W.T.astype(np.float64)
        li -= li.max(axis=1, keepdims=True)
        Pi = np.exp(li)
        Pi /= Pi.sum(axis=1, keepdims=True)
        ti8 = np.argsort(Pi, axis=1)[:, ::-1][:, :8]
        have = set(use.keys())
        cov_tok = float(np.mean([all(int(e) in have for e in row) for row in ti8]))
        cov_slot = float(np.mean([int(e) in have for row in ti8 for e in row]))
        Pis = np.sort(Pi, axis=1)[:, ::-1]
        cumi = Pis.cumsum(axis=1).mean(axis=0)
        res[L] = {
            "N": N, "n_experts": len(use),
            "relD4_R8": [float(rD4.mean()), float(rD4.min()), float(rD4.max())],
            "relD2_R8": [float(rD2.mean()), float(rD2.min()), float(rD2.max())],
            "relD4_block": [float(bD4.mean()), float(bD4.min()), float(bD4.max())],
            "relD2_block": [float(bD2.mean()), float(bD2.min()), float(bD2.max())],
            "omit4_within8": float(1 - (m4 / m8).mean()),
            "omit2_within8": float(1 - (m2 / m8).mean()),
            "mass8": float(m8.mean()),
            "norm_R8": float(nR8.mean()), "norm_S": float(np.linalg.norm(S, axis=1).mean()),
            "norm_B": float(nB.mean()), "norm_X": float(np.linalg.norm(X, axis=1).mean()),
            "sig_gate": [float(sg.mean()), float(sg.min()), float(sg.max())],
            "expert_cos": [float(off.mean()), float(np.median(np.abs(off))),
                           float(np.abs(off).max())],
            "expert_norm_ratio_maxmin": float(max(enorms.values()) / min(enorms.values())),
            "iso_cover_tok": cov_tok, "iso_cover_slot": cov_slot,
            "iso_mass148": [float(cumi[0]), float(cumi[3]), float(cumi[7])],
        }
        print(f"L{L:02d}: |D4|/|R8|={rD4.mean():.4f} |D2|/|R8|={rD2.mean():.4f} "
              f"|D4|/|B|={bD4.mean():.4f} |D2|/|B|={bD2.mean():.4f} "
              f"omit4={1-(m4/m8).mean():.4f} omit2={1-(m2/m8).mean():.4f} "
              f"Ecos={off.mean():+.3f}/|{np.median(np.abs(off)):.3f} "
              f"iso_cov={cov_tok:.3f}/{cov_slot:.3f}", flush=True)
    json.dump(res, open(os.path.join(HERE, "results_topk.json"), "w"), indent=1)
    print("OK probe03", flush=True)


if __name__ == "__main__":
    main()
