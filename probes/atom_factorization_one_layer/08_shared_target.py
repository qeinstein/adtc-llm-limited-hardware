"""08: fetch L20 shared expert (bf16, ~6 MB) + build block targets.

B(x) = R8(x) + S(x),  S = Down(silu(Gate x)*Up x) * sigmoid(x @ w_gate).
Same forward as probes/agent3_funcmoe/03_topk_outputs.py.

Outputs (assets/): shexp_L20.npz (G,U,D,w f32), targets_L20.npz (B f32).
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
L = 20
URL14 = ("https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/main/"
         "model.safetensors-00014-of-00014.safetensors")
P = f"model.language_model.layers.{L}.mlp"


def fetch_range(start, end, timeout=120):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp = f.name
    try:
        r = subprocess.run(
            ["curl", "-sL", "--fail", "--retry", "5", "--retry-delay", "3",
             "--max-time", str(timeout), "-r", f"{start}-{end}",
             "-o", tmp, URL14], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"fetch failed: {r.stderr[:200]}")
        with open(tmp, "rb") as f:
            data = f.read()
        assert len(data) == end - start + 1
        return data
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def main():
    n = struct.unpack("<Q", fetch_range(0, 7))[0]
    hdr = json.loads(fetch_range(8, 8 + n - 1))
    hdr.pop("__metadata__", None)
    d0 = 8 + n

    def get(name):
        e = hdr[name]
        s, t = e["data_offsets"]
        raw = fetch_range(d0 + s, d0 + t - 1)
        assert e["dtype"] == "BF16"
        return torch.frombuffer(bytearray(raw),
                                dtype=torch.bfloat16).reshape(e["shape"])

    G = get(f"{P}.shared_expert.gate_proj.weight").to(torch.float32).numpy()
    U = get(f"{P}.shared_expert.up_proj.weight").to(torch.float32).numpy()
    D = get(f"{P}.shared_expert.down_proj.weight").to(torch.float32).numpy()
    w = get(f"{P}.shared_expert_gate.weight").to(torch.float32).numpy()
    print(f"shexp: G{G.shape} U{U.shape} D{D.shape} w{w.shape}", flush=True)
    np.savez_compressed(os.path.join(OUT, "shexp_L20.npz"), G=G, U=U, D=D,
                        w=w)

    X = np.load(os.path.join(OUT, "inputs_L20.npy")).astype(np.float64)
    gv = X @ G.T.astype(np.float64)
    uv = X @ U.T.astype(np.float64)
    S = (((gv / (1.0 + np.exp(-gv))) * uv)
         @ D.T.astype(np.float64))
    sg = 1.0 / (1.0 + np.exp(-(X @ w.T.astype(np.float64).ravel())))
    S = S * sg[:, None]
    R8 = np.load(os.path.join(OUT, "teacher_L20.npz"))["R8"].astype(np.float64)
    B = (R8 + S).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, "targets_L20.npz"), B=B,
                        S=S.astype(np.float32))
    nR = np.linalg.norm(R8, axis=1)
    nS = np.linalg.norm(S, axis=1)
    nB = np.linalg.norm(R8 + S, axis=1)
    print(f"|R8|={nR.mean():.3f} |S|={nS.mean():.3f} |B|={nB.mean():.3f} "
          f"sig=[{sg.min():.3f},{sg.mean():.3f},{sg.max():.3f}]", flush=True)
    print("OK 08", flush=True)


if __name__ == "__main__":
    sys.exit(main())
