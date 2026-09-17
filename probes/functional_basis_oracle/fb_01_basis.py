"""Functional-basis oracle 01: stacked expert-response matrix + PCA.

PROVISIONAL (proxy inputs) until the contextual dump lands; same code runs
on contextual states by switching DATA_TAG.

Fits the ORACLE shared functional basis: stack calib expert outputs
  M[e*Nc:(e+1)*Nc, :] = E_e(X_calib)   (256*Nc, 2048)
TruncatedSVD (randomized, exact enough at this scale) -> V (2048 x RMAX).
The oracle reconstructs held-out expert outputs by projection:
  Ehat_e(x) = V_r V_r^T E_e(x)   (uses TRUE E_e at test time)
then composes with TRUE routing: Rhat = sum_e a_e Ehat_e.
This tests whether a SHARED r-dim output subspace captures expert
functions. One-sided by design: fixed subspace is sufficient, not
necessary (x-adaptive could do better); the dense-student failure covers
the x-adaptive side. See REPORT.md.

Memory: M is a disk memmap (256*320 x 2048 f32 = 640 MB); streaming build.

Outputs (fb_assets/): M_stack.npy (memmap), Vt_R256.npy, S_top.npy,
  pca_meta.json, spectrum.json
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fb_assets")
os.makedirs(OUT, exist_ok=True)
ATOM = "/home/fluxx/Workspace/adtc-llm-native-sparse/probes/atom_factorization_one_layer/assets"
DEQ = "/tmp/agent1_f32"
L = 20
RMAX = 256
DATA_TAG = os.environ.get("FB_DATA", "proxy")  # or 'context'


def load_inputs():
    if DATA_TAG == "proxy":
        split = json.load(open(os.path.join(ATOM, "split.json")))
        n_calib = split["n_calib"]
        X = np.load(os.path.join(ATOM, "inputs_L20.npy")).astype(np.float32)
        return X[:n_calib], n_calib
    else:
        d = np.load(os.path.join(OUT, "context_inputs.npz"))
        return d["Xtr"].astype(np.float32), d["Xtr"].shape[0]


def main():
    t0 = time.time()
    Xc, Nc = load_inputs()
    A = 256 * Nc
    print(f"DATA={DATA_TAG} Nc={Nc} M=({A},2048)", flush=True)
    M = np.lib.format.open_memmap(os.path.join(OUT, f"M_stack_{DATA_TAG}.npy"),
                                  mode="w+", dtype=np.float32, shape=(A, 2048))
    for e in range(256):
        G = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32",
                        dtype=np.float32).reshape(512, 2048)
        U = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32",
                        dtype=np.float32).reshape(512, 2048)
        D = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_down.f32",
                        dtype=np.float32).reshape(2048, 512)
        gv = Xc @ G.T
        uv = Xc @ U.T
        Y = (((gv / (1.0 + np.exp(-gv))) * uv) @ D.T).astype(np.float32)
        M[e * Nc:(e + 1) * Nc] = Y
        del G, U, D, gv, uv, Y
        if (e + 1) % 64 == 0:
            print(f"  stream {e+1}/256 ({time.time()-t0:.0f}s)", flush=True)
    M.flush()
    print(f"M built ({time.time()-t0:.0f}s), SVD...", flush=True)
    t1 = time.time()
    svd = TruncatedSVD(n_components=RMAX, random_state=0)
    svd.fit(M)
    Vt = svd.components_.astype(np.float32)  # (256,2048)
    S = svd.singular_values_.astype(np.float64)
    np.save(os.path.join(OUT, f"Vt_R256_{DATA_TAG}.npy"), Vt)
    np.save(os.path.join(OUT, f"S_top_{DATA_TAG}.npy"), S)
    ev = (S ** 2)
    ev /= ev.sum()
    cum = np.cumsum(ev)
    erank = float(np.exp(-(ev * np.log(np.maximum(ev, 1e-300))).sum()))
    spec = {"RMAX": RMAX, "Nc": Nc, "erank_top256": erank,
            "cumvar": {r: float(cum[r - 1]) for r in
                       (16, 32, 64, 96, 128, 160, 192, 224, 256)},
            "top_sv_ratio": float(S[0] / S[-1])}
    json.dump(spec, open(os.path.join(OUT, f"spectrum_{DATA_TAG}.json"), "w"),
              indent=1)
    print(f"erank={erank:.1f} cumvar64={cum[63]:.4f} cumvar128={cum[127]:.4f} "
          f"cumvar256={cum[255]:.4f} (svd {time.time()-t1:.0f}s)", flush=True)
    print(f"OK fb01 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
