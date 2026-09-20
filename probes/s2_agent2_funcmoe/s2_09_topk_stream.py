"""S2-Agent2 09: streaming R8/R4/R2 + D4/D2 for extension layers.

Same math as s2_03 (exact forward, f32 matmuls, f64 accum, renormed
deployable truncation, shared expert bf16->f32, sigmoid gate) but:
 - evaluates each expert ONLY on tokens routing to it (~11 vs 320 rows,
   ~30x cheaper; per-row matmuls so rows match full-N up to BLAS blocking);
 - dequants each expert's 3 slices on demand from /tmp/agent1_raw and
   deletes the f32 it created (bounds /tmp; pre-existing f32 kept).
Inputs/routing loaded from THIS dir's weights/ first, A3 fallback.
--verify L: recompute and diff vs cached outputs_Lxx.npz (no writes).
Output: weights/outputs_Lxx.npz, results_topk_ext.json (merged, no clobber).
Usage: python3 s2_09_topk_stream.py 5,20,30 | python3 s2_09_topk_stream.py --verify 0
"""
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch

WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
A3 = os.path.join(WS, "probes/agent3_funcmoe/weights")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights")
RAW = "/tmp/agent1_raw"
DEQ = "/tmp/agent1_f32"


def load_exp_stream(L, e, kind):
    """Return f32 weight + flag whether we created the f32 (caller deletes)."""
    ext = "iq2xxs" if kind != "down" else "iq2s"
    src = f"{RAW}/L{L:02d}_E{e:03d}_{kind}.{ext}"
    dst = f"{DEQ}/L{L:02d}_E{e:03d}_{kind}.f32"
    made = False
    if not os.path.exists(dst):
        ty = "iq2_xxs" if "xxs" in ext else "iq2_s"
        ne0, nr = (2048, 512) if kind != "down" else (512, 2048)
        r = subprocess.run(["/tmp/dequant", src, dst, ty, str(ne0), str(nr)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"dequant {src}: {r.stderr[:200]}")
        made = True
    W = np.fromfile(dst, dtype=np.float32)
    return (W.reshape(512, 2048) if kind != "down" else W.reshape(2048, 512)), dst, made


def expert_out(X, G, U, D):
    gv = X @ G.T
    uv = X @ U.T
    z = (gv / (1.0 + np.exp(-gv))) * uv
    return z @ D.T


def load_inputs(L):
    for base in (OUT, A3):
        xi = os.path.join(base, f"inputs_L{L:02d}.npy")
        ri = os.path.join(base, f"routing_L{L:02d}.npz")
        if os.path.exists(xi) and os.path.exists(ri):
            return np.load(xi).astype(np.float32), np.load(ri), base
    raise FileNotFoundError(f"inputs/routing L{L:02d}")


def compute_layer(L):
    X, r, base = load_inputs(L)
    N = X.shape[0]
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
        G, f1, m1 = load_exp_stream(L, e, "gate")
        U, f2, m2b = load_exp_stream(L, e, "up")
        D, f3, m3 = load_exp_stream(L, e, "down")
        idx = np.array([t for (t, s) in slots])
        Y = expert_out(X[idx], G, U, D).astype(np.float64)
        enorms[e] = float(np.linalg.norm(Y, axis=1).mean())
        Emean[e] = Y.mean(axis=0)
        for k, (t, s) in enumerate(slots):
            R8[t] += w8[t, s] * Y[k]
            if s < 4:
                R4[t] += w4[t, s] * Y[k]
            if s < 2:
                R2[t] += w2[t, s] * Y[k]
        del G, U, D, Y
        for f, mk in ((f1, m1), (f2, m2b), (f3, m3)):
            if mk:
                os.remove(f)
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
    return X, R8, R4, R2, S, sg, top8, m4, m2, m8, enorms, Emean


def summarize(L, X, R8, R4, R2, S, sg, top8, m4, m2, m8, enorms, Emean):
    N = X.shape[0]
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
    om4 = 1 - (m4 / m8).ravel()
    om2 = 1 - (m2 / m8).ravel()
    cos4 = float(np.mean([(D4[t] @ R4[t]) / (np.linalg.norm(D4[t]) * np.linalg.norm(R4[t]) + 1e-30) for t in range(N)]))
    cos2 = float(np.mean([(D2[t] @ R2[t]) / (np.linalg.norm(D2[t]) * np.linalg.norm(R2[t]) + 1e-30) for t in range(N)]))
    return {
        "N": N, "n_experts": len(enorms),
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


def main():
    if sys.argv[1] == "--verify":
        L = int(sys.argv[2])
        X, R8, R4, R2, S, sg, top8, m4, m2, m8, enorms, Emean = compute_layer(L)
        d = np.load(os.path.join(OUT, f"outputs_L{L:02d}.npz"))
        for nm, M in (("R8", R8), ("R4", R4), ("R2", R2), ("S", S)):
            ref = d[nm].astype(np.float64)
            ad = np.abs(M - ref).max()
            rel = ad / (np.abs(ref).max() + 1e-30)
            print(f"L{L:02d} {nm}: maxabs={ad:.3e} maxrel={rel:.3e}", flush=True)
        print("OK s2_09 verify", flush=True)
        return
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [5, 20, 30]
    ext_path = os.path.join(HERE, "results_topk_ext.json")
    res = json.load(open(ext_path)) if os.path.exists(ext_path) else {}
    for L in layers:
        X, R8, R4, R2, S, sg, top8, m4, m2, m8, enorms, Emean = compute_layer(L)
        D4, D2 = R8 - R4, R8 - R2
        np.savez_compressed(os.path.join(OUT, f"outputs_L{L:02d}.npz"),
                            R8=R8.astype(np.float32), R4=R4.astype(np.float32),
                            R2=R2.astype(np.float32), D4=D4.astype(np.float32),
                            D2=D2.astype(np.float32), S=S.astype(np.float32), X=X)
        row = summarize(L, X, R8, R4, R2, S, sg, top8, m4, m2, m8, enorms, Emean)
        res[str(L)] = row
        print(f"L{L:02d}: |D4|/|R8|={row['relD4_R8'][0]:.4f} |D2|/|R8|={row['relD2_R8'][0]:.4f} "
              f"|D4|/|B|={row['relD4_block'][0]:.4f} |D2|/|B|={row['relD2_block'][0]:.4f}", flush=True)
    json.dump(res, open(ext_path, "w"), indent=1)
    print("OK s2_09", flush=True)


if __name__ == "__main__":
    main()
