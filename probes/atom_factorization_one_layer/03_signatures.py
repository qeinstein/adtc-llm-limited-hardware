"""03: atom signatures for ALL 256 experts x 512 atoms (L20).

For each atom a=(e,j), on CALIB tokens only (held-out stays clean):
  - scalar activation row: s_a[t] = SiLU(g.x_t)*(u.x_t)   -> S_all (A,Nc)
  - output direction:      d_a (2048,)                    -> D_all (A,2048)
  - stats: ||d||, ||s||, energy, routed count, mean/max |contrib|,
    kurtosis of |contrib| over routed calib tokens        -> stats.npz
    where contrib[t] = alpha_e(t)*s_a(t)*||d_a||

Writes go straight to disk memmaps (1.2 GB total); RAM stays ~100 MB.

Outputs (assets/): S_all.npy, D_all.npy, atom_stats.npz, atom_meta.json
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


def main():
    split = json.load(open(os.path.join(OUT, "split.json")))
    n_calib = split["n_calib"]
    Xc = np.load(os.path.join(OUT, "inputs_L20.npy"))[:n_calib].astype(
        np.float32)
    r = np.load(os.path.join(OUT, "routing_L20.npz"))
    top8, w8 = r["top8"][:n_calib], r["weights"][:n_calib].astype(np.float64)

    slots = defaultdict(list)  # expert -> [(token, alpha)]
    for t in range(n_calib):
        for s in range(8):
            slots[int(top8[t, s])].append((t, float(w8[t, s])))

    A = 256 * 512
    S = np.lib.format.open_memmap(os.path.join(OUT, "S_all.npy"), mode="w+",
                                  dtype=np.float32, shape=(A, n_calib))
    D = np.lib.format.open_memmap(os.path.join(OUT, "D_all.npy"), mode="w+",
                                  dtype=np.float32, shape=(A, 2048))
    dnorm = np.zeros(A, np.float32)
    snorm = np.zeros(A, np.float32)
    energy = np.zeros(A, np.float32)
    nroute = np.zeros(A, np.int32)
    cmean = np.zeros(A, np.float32)
    cmax = np.zeros(A, np.float32)
    kurt = np.zeros(A, np.float32)

    for e in range(256):
        G = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_gate.f32",
                        dtype=np.float32).reshape(512, 2048)
        U = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_up.f32",
                        dtype=np.float32).reshape(512, 2048)
        Dw = np.fromfile(f"{DEQ}/L{L:02d}_E{e:03d}_down.f32",
                         dtype=np.float32).reshape(2048, 512)
        gv = Xc @ G.T
        uv = Xc @ U.T
        s = ((gv / (1.0 + np.exp(-gv))) * uv).astype(np.float64)  # (Nc,512)
        d = Dw.T.astype(np.float64)  # (512,2048)
        sl = slice(e * 512, (e + 1) * 512)
        S[sl] = s.T.astype(np.float32)
        D[sl] = d.astype(np.float32)
        dn = np.linalg.norm(d, axis=1)
        sn = np.linalg.norm(s, axis=0)
        dnorm[sl] = dn
        snorm[sl] = sn
        energy[sl] = (s ** 2).sum(axis=0)
        tok = slots.get(e, [])
        nroute[sl] = len(tok)
        if tok:
            ti = np.array([t for t, _ in tok])
            al = np.array([a for _, a in tok])
            cm = np.abs(s[ti] * al[:, None]) * dn[None, :]  # (n,512)
            cmean[sl] = cm.mean(axis=0)
            cmax[sl] = cm.max(axis=0)
            if len(tok) >= 8:
                mu = cm.mean(axis=0)
                va = ((cm - mu) ** 2).mean(axis=0)
                m4 = ((cm - mu) ** 4).mean(axis=0)
                kurt[sl] = m4 / np.maximum(va ** 2, 1e-30) - 3.0
        del G, U, Dw, gv, uv, s, d
        if (e + 1) % 32 == 0:
            print(f"  {e+1}/256", flush=True)
    S.flush()
    D.flush()
    np.savez_compressed(os.path.join(OUT, "atom_stats.npz"),
                        dnorm=dnorm, snorm=snorm, energy=energy,
                        nroute=nroute, cmean=cmean, cmax=cmax, kurt=kurt)
    json.dump({"A": A, "n_calib": n_calib, "layer": L},
              open(os.path.join(OUT, "atom_meta.json"), "w"), indent=1)
    print(f"A={A} S={S.shape} D={D.shape}", flush=True)
    print("OK 03", flush=True)


if __name__ == "__main__":
    sys.exit(main())
