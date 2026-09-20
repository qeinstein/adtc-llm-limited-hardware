"""S2 Agent1 (top-k) script 02: fetch + dequant FULL expert layers.

Target: all 256 experts x layers (default 10,20,30,39) + top-up L00 to 256.
Slices from pinned UD-IQ2_XXS GGUF (agent1 layout, verified sizes):
  gate/up expert = 270336 B (IQ2_XXS), down expert = 335872 B (IQ2_S).
Dequant: /tmp/dequant (pinned ggml) -> f32 (gate/up 512x2048, down 2048x512).

Resume-safe: skips complete slices (exact size), deletes partials, refetches.
Per-layer sequential; curl parallel within layer (8 workers, --retry 5).
Checkpoint: prints READY Lxx when all 256x3 f32 verified.

Usage: python3 02_fetch.py 20,39
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

WS = "/home/fluxx/Workspace/adtc-llm-native-sparse"
RAW = "/tmp/agent1_raw"
DEQ = "/tmp/agent1_f32"
BASE = ("https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
        "bc014a17be43adabd7066b7a86075ff935c6a4e2/"
        "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf")
INV = json.load(open(os.path.join(WS, "probes/agent1_expert/gguf_inventory.json")))
T = {t["name"]: t for t in INV["tensors"]}
GB, DB = 270336, 335872


def fetch_one(job):
    a, b, out = job
    if os.path.exists(out) and os.path.getsize(out) == b - a + 1:
        return "have"
    tmp = out + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    r = subprocess.run(
        ["curl", "-sL", "--fail", "--retry", "5", "--retry-all-errors",
         "--max-time", "300", "-r", f"{a}-{b}", "-o", tmp, BASE],
        capture_output=True, text=True)
    if r.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return f"FAIL {a}-{b}: {r.stderr[:160]}"
    if os.path.getsize(tmp) != b - a + 1:
        os.remove(tmp)
        return f"SHORT {a}-{b}"
    os.rename(tmp, out)
    return "got"


def dequant(src, dst, ty, ne0, nr):
    if os.path.exists(dst) and os.path.getsize(dst) == ne0 * nr * 4:
        return "have"
    r = subprocess.run(["/tmp/dequant", src, dst, ty, str(ne0), str(nr)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dequant {src}: {r.stderr[:200]}")
    assert os.path.getsize(dst) == ne0 * nr * 4, dst
    return "got"


def main():
    layers = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 \
        else [10, 20, 30, 39]
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(DEQ, exist_ok=True)
    for L in layers:
        tg = T[f"blk.{L}.ffn_gate_exps.weight"]
        tu = T[f"blk.{L}.ffn_up_exps.weight"]
        td = T[f"blk.{L}.ffn_down_exps.weight"]
        jobs = []
        for e in range(256):
            for ti, nb, kind, ext in ((tg, GB, "gate", "iq2xxs"),
                                      (tu, GB, "up", "iq2xxs"),
                                      (td, DB, "down", "iq2s")):
                src = f"{RAW}/L{L:02d}_E{e:03d}_{kind}.{ext}"
                a = ti["file_offset"] + e * nb
                jobs.append((a, a + nb - 1, src))
        todo = [j for j in jobs if not (
            os.path.exists(j[2]) and os.path.getsize(j[2]) == j[1] - j[0] + 1)]
        print(f"L{L:02d}: {len(jobs) - len(todo)}/{len(jobs)} slices present, "
              f"fetching {len(todo)}", flush=True)
        for attempt in range(4):
            if not todo:
                break
            with ThreadPoolExecutor(max_workers=8) as ex:
                res = list(ex.map(fetch_one, todo))
            fails = [r for r in res if r not in ("have", "got")]
            got = len(res) - len(fails)
            todo = [j for j, r in zip(todo, res) if r not in ("have", "got")]
            print(f"  round {attempt}: got {got}, fails {len(fails)}",
                  flush=True)
            for f in fails[:4]:
                print(f"    {f}", flush=True)
        if todo:
            raise SystemExit(f"L{L:02d}: {len(todo)} slices FAILED after "
                             f"retries -- rerun to resume")
        # dequant (local, sequential)
        ndeq = 0
        for e in range(256):
            for kind, ext in (("gate", "iq2xxs"), ("up", "iq2xxs"),
                              ("down", "iq2s")):
                src = f"{RAW}/L{L:02d}_E{e:03d}_{kind}.{ext}"
                dst = f"{DEQ}/L{L:02d}_E{e:03d}_{kind}.f32"
                ty = "iq2_xxs" if "xxs" in ext else "iq2_s"
                ne0, nr = (2048, 512) if kind != "down" else (512, 2048)
                if dequant(src, dst, ty, ne0, nr) == "got":
                    ndeq += 1
            if e % 64 == 63:
                print(f"  dequant {e + 1}/256", flush=True)
        print(f"READY L{L:02d} (dequanted {ndeq} new slices)", flush=True)
    print("OK 02", flush=True)


if __name__ == "__main__":
    main()
