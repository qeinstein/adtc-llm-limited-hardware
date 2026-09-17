"""01: exact teacher R8 for L20 + per-expert stats.

Streams REAL dequantized f32 experts from /tmp/agent1_f32 (only the union
needed by our N tokens). Exact forward, float32 math, float64 accumulation:

    E_e(x) = Down(SiLU(Gate x) * Up x);  R8 = sum_top8 alpha_e E_e.

Memory-frugal: one expert (12 MB) at a time; 2.6 GB box.

Outputs (assets/):
  - teacher_L20.npz: R8 (N,2048 f32), Emean (U,2048 f32) mean expert output
    over tokens routed to it, enorm (U,) mean output norm, eids (U,)
  - teacher_stats.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
DEQ = "/tmp/agent1_f32"
L = 20


def load_exp(e, kind):
    p = f"{DEQ}/L{L:02d}_E{e:03d}_{kind}.f32"
    W = np.fromfile(p, dtype=np.float32)
    return W.reshape(512, 2048) if kind != "down" else W.reshape(2048, 512)


def expert_out(X, G, U, D):
    gv = X @ G.T
    uv = X @ U.T
    z = (gv / (1.0 + np.exp(-gv))) * uv
    return z @ D.T


def main():
    split = json.load(open(os.path.join(OUT, "split.json")))
    n_calib, N = split["n_calib"], split["N"]
    X = np.load(os.path.join(OUT, "inputs_L20.npy")).astype(np.float32)
    r = np.load(os.path.join(OUT, "routing_L20.npz"))
    top8, w8 = r["top8"], r["weights"].astype(np.float64)

    use = defaultdict(list)
    for t in range(N):
        for s in range(8):
            use[int(top8[t, s])].append((t, s))
    eids = sorted(use.keys())
    print(f"N={N} union experts={len(eids)}", flush=True)

    R8 = np.zeros((N, 2048), np.float64)
    Emean = np.zeros((len(eids), 2048), np.float64)
    Ecnt = np.zeros(len(eids), np.int64)
    enorm_acc = np.zeros(len(eids), np.float64)
    for i, e in enumerate(eids):
        G, U, D = load_exp(e, "gate"), load_exp(e, "up"), load_exp(e, "down")
        Y = expert_out(X, G, U, D).astype(np.float64)
        enorm_acc[i] = np.linalg.norm(Y, axis=1).mean()
        for (t, s) in use[e]:
            R8[t] += w8[t, s] * Y[t]
            Emean[i] += Y[t]
            Ecnt[i] += 1
        del G, U, D, Y
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(eids)}", flush=True)
    Emean /= Ecnt[:, None]

    nR8 = np.linalg.norm(R8, axis=1)
    np.savez_compressed(os.path.join(OUT, "teacher_L20.npz"),
                        R8=R8.astype(np.float32),
                        Emean=Emean.astype(np.float32),
                        enorm=enorm_acc.astype(np.float32),
                        eids=np.array(eids, dtype=np.int32))
    # expert redundancy: cosine of mean outputs
    En = Emean / np.linalg.norm(Emean, axis=1, keepdims=True)
    C = En @ En.T
    off = C[np.triu_indices(len(eids), 1)]
    stats = {
        "n_experts": len(eids),
        "norm_R8_mean": float(nR8.mean()),
        "norm_R8_calib": float(nR8[:n_calib].mean()),
        "norm_R8_held": float(nR8[n_calib:].mean()),
        "norm_X": float(np.linalg.norm(X, axis=1).mean()),
        "expert_cos_mean": float(off.mean()),
        "expert_cos_medabs": float(np.median(np.abs(off))),
        "expert_cos_maxabs": float(np.abs(off).max()),
        "expert_norm_ratio_maxmin": float(enorm_acc.max() / enorm_acc.min()),
    }
    json.dump(stats, open(os.path.join(OUT, "teacher_stats.json"), "w"),
              indent=1)
    print("teacher: |R8|={:.3f} Ecos={:+.3f}/|{:.3f}| norm_ratio={:.2f}"
          .format(stats["norm_R8_mean"], stats["expert_cos_mean"],
                  stats["expert_cos_medabs"],
                  stats["expert_norm_ratio_maxmin"]), flush=True)
    print("OK 01", flush=True)


if __name__ == "__main__":
    sys.exit(main())
