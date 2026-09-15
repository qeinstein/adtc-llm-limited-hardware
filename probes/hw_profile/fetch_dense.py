"""hw_profile fetch: range-fetch packed dense tensors for exact-kernel replay.
Writes ONLY to OUTDIR (/tmp/hwprof_dense). Resumable/skip-if-complete.
Usage: python3 fetch_dense.py [job]   # job = index into JOBS for xargs -P
"""
import json
import os
import subprocess
import sys

BASE = "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = "/tmp/hwprof_dense"
os.makedirs(OUTDIR, exist_ok=True)

INV = json.load(open("/home/fluxx/Workspace/adtc-llm-native-sparse/probes/agent1_expert/gguf_inventory.json"))
T = {t["name"]: t for t in INV["tensors"]}
SIZES = {}
names = sorted(T, key=lambda n: T[n]["file_offset"])
for i, n in enumerate(names):
    if i + 1 < len(names):
        SIZES[n] = T[names[i + 1]]["file_offset"] - T[n]["file_offset"]

WANT = [
    "blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight",
    "blk.3.attn_output.weight", "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
    "blk.0.ssm_out.weight", "blk.0.ffn_gate_inp.weight",
    "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight",
    "blk.0.ffn_down_shexp.weight", "output.weight",
    "blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight",
]


def fetch(name):
    t = T[name]
    n = SIZES[name]
    out = os.path.join(OUTDIR, name.replace(".", "_") + ".bin")
    if os.path.exists(out) and os.path.getsize(out) == n:
        print(f"have {name} ({n} B)", flush=True)
        return
    a, b = t["file_offset"], t["file_offset"] + n - 1
    # split large tensors into chunks for parallelism from xargs side; here single stream
    r = subprocess.run(["curl", "-sL", "--fail", "--retry", "3", "-r", f"{a}-{b}",
                        "-o", out, BASE], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"fetch {name} failed: {r.stderr[:300]}")
    got = os.path.getsize(out)
    assert got == n, f"short: {got} vs {n}"
    print(f"got {name} ({got} B)", flush=True)


if __name__ == "__main__":
    jobs = [WANT[int(sys.argv[1])]] if len(sys.argv) > 1 else WANT
    for w in jobs:
        fetch(w)
