"""Agent1 probe 04: expert weight structure + SwiGLU sparsity + width pruning.

REAL inputs: exact IQ2_XXS/IQ2_S dequant (pinned ggml @3057bb66) of gate/up/down
  for 16 experts x layers {0,10,20,30} + real RMSNorm scales.
ASSUMPTION (labeled): isotropic-calibrated inputs x_i = g_i * z_i, z~N(0,1).
  Direction-sensitive quantities (sparsity fractions, output scales) are PRIORS,
  not measurements; weight-only quantities (dim importance energy, norm stats)
  are REAL structural facts.

Q_a: width 512->384/256: cumulative dim-importance energy (REAL, weight-only)
Q_b: SwiGLU sparsity: P(|z|<t), magnitude histogram, block sparsity 16/32/64
Q_c: expert output scale ||E(x)|| and cross-expert variation (prior for
     output-weighted pruning / dispensable branches)
Q_d: gate/up branch asymmetry (dispensable branches?)
"""
import glob
import os
import subprocess
import sys

import numpy as np

RAW = "/tmp/agent1_raw"
DEQ = "/tmp/agent1_f32"
os.makedirs(DEQ, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 10, 20, 30]

rng = np.random.default_rng(1)


def dequant_all():
    for L in LAYERS:
        for kind, ty, ne0, nr in (("gate", "iq2_xxs", 2048, 512),
                                  ("up", "iq2_xxs", 2048, 512),
                                  ("down", "iq2_s", 512, 2048)):
            ext = "iq2xxs" if "xxs" in ty else "iq2s"
            for src in sorted(glob.glob(f"{RAW}/L{L:02d}_E*_ {kind}.{ext}".replace(" ", ""))):
                base = os.path.basename(src).replace(f".{ext}", "")
                dst = f"{DEQ}/{base}.f32"
                if os.path.exists(dst):
                    continue
                r = subprocess.run(["/tmp/dequant", src, dst, ty, str(ne0), str(nr)],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"dequant {src}: {r.stderr[:200]}")
    print("dequant done")


def load(L, e, kind):
    p = f"{DEQ}/L{L:02d}_E{e:03d}_{kind}.f32"
    W = np.fromfile(p, dtype=np.float32)
    if kind == "down":
        W = W.reshape(2048, 512)
    else:
        W = W.reshape(512, 2048)
    assert np.isfinite(W).all(), p
    return W


def main():
    dequant_all()
    S = 512
    print(f"{'L':>3s} {'E':>3s} {'std_g':>7s} {'std_u':>7s} {'std_d':>7s} "
          f"{'e384':>6s} {'e256':>6s} {'e128':>6s} {'Pneg':>6s} "
          f"{'sp1%':>6s} {'sp5%':>6s} {'sp10%':>7s} {'b16':>6s} {'b32':>6s} {'b64':>6s} {'|E|/|x|':>8s}")
    erows = []
    for L in LAYERS:
        g = np.fromfile(f"{RAW}/L{L:02d}_post_attention_norm.f32", dtype=np.float32)
        X = (rng.standard_normal((S, 2048)) * g[None, :]).astype(np.float64)
        xnorm = np.linalg.norm(X, axis=1).mean()
        experts = sorted({int(os.path.basename(p).split("_")[1][1:])
                          for p in glob.glob(f"{DEQ}/L{L:02d}_E*_gate.f32")})
        for e in experts:
            G = load(L, e, "gate").astype(np.float64)
            U = load(L, e, "up").astype(np.float64)
            D = load(L, e, "down").astype(np.float64)
            # Q_a: dim importance (REAL weight-only)
            ng = np.linalg.norm(G, axis=1)
            nu = np.linalg.norm(U, axis=1)
            nd = np.linalg.norm(D, axis=0)
            s = ng * nu * nd
            s_sorted = np.sort(s)[::-1]
            ecum = np.cumsum(s_sorted ** 2)
            ecum = ecum / ecum[-1]
            e384, e256, e128 = ecum[383], ecum[255], ecum[127]
            # Q_b/c: forward under calibrated inputs
            gv = X @ G.T  # (S,512)
            uv = X @ U.T
            silu = gv / (1.0 + np.exp(-gv))
            z = silu * uv
            zrms = float(np.sqrt((z ** 2).mean()))
            az = np.abs(z)
            sp1 = float((az < 0.01 * zrms).mean())
            sp5 = float((az < 0.05 * zrms).mean())
            sp10 = float((az < 0.10 * zrms).mean())
            pneg = float((gv < 0).mean())
            # block sparsity: fraction of blocks with max < 5% rms
            def bsp(b):
                zb = az.reshape(S, 512 // b, b).max(axis=2)
                return float((zb < 0.05 * zrms).mean())
            b16, b32, b64 = bsp(16), bsp(32), bsp(64)
            Y = z @ D.T
            ynorm = np.linalg.norm(Y, axis=1).mean()
            ratio = ynorm / xnorm
            erows.append((L, e, e384, e256, e128, sp1, sp5, sp10, b16, b32, b64, ratio))
            print(f"{L:3d} {e:3d} {G.std():7.4f} {U.std():7.4f} {D.std():7.4f} "
                  f"{e384:6.3f} {e256:6.3f} {e128:6.3f} {pneg:6.3f} "
                  f"{sp1:6.3f} {sp5:6.3f} {sp10:7.3f} {b16:6.3f} {b32:6.3f} {b64:6.3f} {ratio:8.3f}")
    A = np.array([r[2:] for r in erows])
    names = ["e384", "e256", "e128", "sp1", "sp5", "sp10", "b16", "b32", "b64", "ratio"]
    print("\naggregate over %d experts (mean / min / max):" % len(erows))
    for j, n in enumerate(names):
        print(f"  {n:6s}: {A[:, j].mean():.4f} / {A[:, j].min():.4f} / {A[:, j].max():.4f}")


if __name__ == "__main__":
    main()
