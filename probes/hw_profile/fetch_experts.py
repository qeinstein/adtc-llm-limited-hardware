"""hw_profile fetch: range-fetch packed expert slices for token60 replay.
Reads /tmp/hwprof_need.txt lines 'L E'. Writes to /tmp/agent1_raw (shared
convention) so slices are reusable. Skips complete files.
Usage: python3 fetch_experts.py <idx>  (one bundle per xargs job)
"""
import json
import os
import subprocess
import sys

BASE = "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
RAW = "/tmp/agent1_raw"
INV = json.load(open("/home/fluxx/Workspace/adtc-llm-native-sparse/probes/agent1_expert/gguf_inventory.json"))
T = {t["name"]: t for t in INV["tensors"]}
GB = 270336
DB = 335872


def fetch_range(a, b, out):
    n = b - a + 1
    if os.path.exists(out) and os.path.getsize(out) == n:
        return
    tmp = out + ".part"
    r = subprocess.run(["curl", "-sL", "--fail", "--retry", "3", "-r", f"{a}-{b}",
                        "-o", tmp, BASE], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"fetch {a}-{b} failed: {r.stderr[:200]}")
    os.rename(tmp, out)


if __name__ == "__main__":
    need = [l.split() for l in open("/tmp/hwprof_need.txt") if l.strip()]
    L, E = int(need[int(sys.argv[1])][0]), int(need[int(sys.argv[1])][1])
    tg = T[f"blk.{L}.ffn_gate_exps.weight"]["file_offset"]
    tu = T[f"blk.{L}.ffn_up_exps.weight"]["file_offset"]
    td = T[f"blk.{L}.ffn_down_exps.weight"]["file_offset"]
    fetch_range(tg + E * GB, tg + (E + 1) * GB - 1, f"{RAW}/L{L:02d}_E{E:03d}_gate.iq2xxs")
    fetch_range(tu + E * GB, tu + (E + 1) * GB - 1, f"{RAW}/L{L:02d}_E{E:03d}_up.iq2xxs")
    fetch_range(td + E * DB, td + (E + 1) * DB - 1, f"{RAW}/L{L:02d}_E{E:03d}_down.iq2s")
    print(f"got L{L:02d} E{E:03d}", flush=True)
