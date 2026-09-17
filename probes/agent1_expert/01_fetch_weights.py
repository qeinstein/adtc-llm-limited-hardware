"""Agent1 probe 01: range-fetch weight slices from the pinned UD-IQ2_XXS GGUF.

Fetches (all via HTTPS Range, no full download):
  - all 40 routers blk.L.ffn_gate_inp.weight (F32 2048x256, 2 MiB each)
  - all 40 post_attention_norm (F32 2048) + attn_norm (F32 2048)
  - expert slices for analysis layers: gate/up (IQ2_XXS) + down (IQ2_S)
    for experts listed in EXPERTS_PER_LAYER (default: 8 route-frequent +
    8 random per layer, resolved after router fetch; first pass: 0..15)

Layout notes (GGUF, validated against probe11 README + route logs):
  - gate/up _exps: expert e contiguous at e*270336 B (512 rows x 8 blk x 66 B)
  - down _exps:    expert e contiguous at e*335872 B (2048 rows x 2 blk x 82 B)
  - dequant row order: gate/up rows are the 512 intermediate dims (ne0=2048);
    down rows are 2048 output dims (ne0=512).

Usage:
  python3 01_fetch_weights.py routers            # F32 tensors only (~80 MB)
  python3 01_fetch_weights.py experts 0,10,20,30 # expert slices for layers
  python3 01_fetch_weights.py experts-all        # 8 experts x all 40 layers
"""
import json
import os
import subprocess
import sys

BASE = "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = "/tmp/agent1_raw"
os.makedirs(RAW, exist_ok=True)

INV = json.load(open(os.path.join(HERE, "gguf_inventory.json")))
T = {t["name"]: t for t in INV["tensors"]}

GATE_EXPERT_BYTES = 270336
DOWN_EXPERT_BYTES = 335872


def fetch_range(a, b, out):
    if os.path.exists(out) and os.path.getsize(out) == b - a + 1:
        print(f"  have {out} ({b-a+1} B)")
        return
    r = subprocess.run(["curl", "-sL", "--fail", "--retry", "3", "-r", f"{a}-{b}",
                        "-o", out, BASE], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"fetch {a}-{b} failed: {r.stderr[:300]}")
    got = os.path.getsize(out)
    assert got == b - a + 1, f"short: {got} vs {b-a+1}"
    print(f"  got {out} ({got} B)")


def fetch_routers():
    for L in range(40):
        t = T[f"blk.{L}.ffn_gate_inp.weight"]
        assert t["type"] == "F32" and t["shape"] == [2048, 256], t
        n = 2048 * 256 * 4
        fetch_range(t["file_offset"], t["file_offset"] + n - 1,
                    f"{RAW}/router_L{L:02d}.f32")


def fetch_norms():
    for L in range(40):
        for nm in ("attn_norm", "post_attention_norm"):
            t = T[f"blk.{L}.{nm}.weight"]
            assert t["type"] == "F32", t
            n = 2048 * 4
            fetch_range(t["file_offset"], t["file_offset"] + n - 1,
                        f"{RAW}/L{L:02d}_{nm}.f32")


def fetch_experts(layers, experts):
    for L in layers:
        tg = T[f"blk.{L}.ffn_gate_exps.weight"]
        tu = T[f"blk.{L}.ffn_up_exps.weight"]
        td = T[f"blk.{L}.ffn_down_exps.weight"]
        for e in experts:
            fetch_range(tg["file_offset"] + e * GATE_EXPERT_BYTES,
                        tg["file_offset"] + (e + 1) * GATE_EXPERT_BYTES - 1,
                        f"{RAW}/L{L:02d}_E{e:03d}_gate.iq2xxs")
            fetch_range(tu["file_offset"] + e * GATE_EXPERT_BYTES,
                        tu["file_offset"] + (e + 1) * GATE_EXPERT_BYTES - 1,
                        f"{RAW}/L{L:02d}_E{e:03d}_up.iq2xxs")
            fetch_range(td["file_offset"] + e * DOWN_EXPERT_BYTES,
                        td["file_offset"] + (e + 1) * DOWN_EXPERT_BYTES - 1,
                        f"{RAW}/L{L:02d}_E{e:03d}_down.iq2s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "routers":
        fetch_routers()
    elif cmd == "norms":
        fetch_norms()
    elif cmd == "experts":
        layers = [int(x) for x in sys.argv[2].split(",")]
        experts = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(16))
        fetch_experts(layers, experts)
    else:
        raise SystemExit(f"unknown cmd {cmd}")
