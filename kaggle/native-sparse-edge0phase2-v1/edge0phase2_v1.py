"""Phase-2 quality gates: Qwen3.5-control vs Qwen3.6-challenger (matched UD-IQ2_XXS).

Why this comparison: Edge0's released pipeline (MLX-only, Qwen3.6-int4 base +
K4+prerouter+LoRA) cannot execute on CPU/Ubuntu (see PHASE2 fidelity note).
This kernel answers the executable, decision-relevant questions:
  (1) control baselines on OUR gates (needed for Phase 4 regardless);
  (2) is the Qwen3.6 base itself viable on our gates (informs Phase-4 base)?

Matched quants (both UD-IQ2_XXS) isolate the pretrain difference.
Disk-safe: models downloaded/run/deleted SEQUENTIALLY (peak ~13 GB).

v4 harness fix (v3 post-mortem): v3's server generations came back EMPTY
(87/88 records, completion_tokens=300 = hit max) because Qwen3 thinking
consumed the whole 300-token budget (evidence: h14 'DENY' cost 279 tokens).
v4 appends the Qwen3-documented /no_think trigger (deployment-faithful:
production cannot afford think-tokens at 18 tok/s), captures
reasoning_content alongside content, and fail-fasts if >=2 of the first
8 prompts return empty content. MMLU bumped 100 -> 200 tasks for power.

v5 harness fix (v4 post-mortem): /no_think suffix did NOT disable thinking
(v4 log: rsn_chars ~900-1200, out_chars=0, ctok=300 on all 8 probes before
the fail-fast correctly tripped — no blind burn). v5 sends
"enable_thinking": false in the chat payload (ignored harmlessly if the
server predates it) AND raises the budget to 1200 tokens so think+answer
fits even on the slow path; the server already splits reasoning_content
from content, so grading on content stays clean either way. Fail-fast now
trips only when BOTH fields are empty (>=2/8); persistent thinking just
logs a loud warning and continues on the slow path (~100s/prompt).
Salvaged from v4: MMLU-A-200 = 40.5% (tasks 1-100 reproduce 37.0 exactly,
third replication of the control).

Gates (frozen, raw-pinned + sha-verified from the repo):
  - MMLU 200-task matched likelihood (deterministic)
  - data/swahili_eval_set.json: 18 prompts, gold-keyword scoring
  - docs/research/falcon_probe_heldout.json: 24 prompts, outputs captured
    for adjudicated grading (checks are natural-language; kernel does NOT
    auto-grade them beyond capture + timing)
  - metadata.json test_prompts tp_001/tp_002 (qualitative capture)
Generation: llama-server /chat/completions (repo template applied by the
server), temperature 0, fixed seed (deterministic comparison).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-edge0phase2-v1")
OUT = WORK / "native-sparse-edge0phase2-v1-results"
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
SERVER = BUILD / "bin" / "llama-server"
PERPLEXITY = BUILD / "bin" / "llama-perplexity"

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
N_THREADS = 4
N_TASKS = 200
NO_THINK = " /no_think"
GEN_MAX_TOKENS = 1200

# frozen gate data: (repo path, sha256) @ REPO_PIN
REPO_PIN = "405b114ed9951b596cbf1b1482c985f8e31dc948"
GATE_FILES = {
    "swahili_eval_set.json":
        "a927c9dd342eef13e196bd5cf2d8ad1d35ff0444b766c2ee9e375c12f868a8d6",
    "falcon_probe_heldout.json":
        "6747c0907e63938f81942219858e1b9cf9fa131c5d915c8664885b2dc851caa4",
    "metadata.json":
        "ff30829531186152802e22cc12a04418efbba65be29845096864422854964ee9",
}
GATE_PATHS = {"swahili_eval_set.json": "data/swahili_eval_set.json",
              "falcon_probe_heldout.json":
                  "docs/research/falcon_probe_heldout.json",
              "metadata.json": "metadata.json"}

# model A: control (pinned size+sha from the q2k kernel)
A_REPO = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
A_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
A_SIZE = 10_656_955_008
A_SHA = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
A_URL = ("https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
         f"{A_REPO}/{A_FILE}")
# model B: challenger (pinned by repo commit; size/sha RECORDED, first run)
B_REPO = "a483e9e6cbd595906af30beda3187c2663a1118c"
B_FILE = "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"
B_URL = ("https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/"
         f"{B_REPO}/{B_FILE}")

DATA_REVISION = "37884b81b4957f1950a53b6ff48d77c8dd5e430c"
DATA_URL = ("https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/"
            f"resolve/{DATA_REVISION}/mmlu-test.bin")

OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)


def run(cmd, cwd=None, log=None):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       check=False)
    print(p.stdout[-2000:], flush=True)
    if log:
        log.write_text(p.stdout, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"command exited {p.returncode}: {cmd}")
    return p.stdout


def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            d.update(b)
    return d.hexdigest()


def build():
    if not (LLAMA / ".git").exists():
        run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git",
             str(LLAMA)])
    run(["git", "-C", str(LLAMA), "fetch", "--depth", "1", "origin",
         LLAMA_COMMIT])
    run(["git", "-C", str(LLAMA), "checkout", LLAMA_COMMIT])
    run(["cmake", "-S", str(LLAMA), "-B", str(BUILD),
         "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DLLAMA_CURL=ON"])
    run(["cmake", "--build", str(BUILD), "--config", "Release",
         f"-j{N_THREADS}", "--target", "llama-server", "llama-perplexity"],
        log=OUT / "build.log")


def fetch_gates():
    got = {}
    for name, want in GATE_FILES.items():
        url = (f"https://raw.githubusercontent.com/qeinstein/"
               f"adtc-llm-limited-hardware/{REPO_PIN}/{GATE_PATHS[name]}")
        dest = SCRATCH / name
        urllib.request.urlretrieve(url, dest)
        d = sha256(dest)
        if d != want:
            raise RuntimeError(f"gate {name} sha mismatch: {d}")
        got[name] = {"sha256": d, "size": dest.stat().st_size}
    return got


def fetch_model(url, dest, size=None, sha=None):
    if not dest.exists() or (size and dest.stat().st_size != size):
        run(["curl", "-L", "--retry", "3", "-C", "-", "-o", str(dest), url])
    st = dest.stat().st_size
    d = sha256(dest)
    if size and st != size:
        raise RuntimeError(f"size mismatch {st} != {size}")
    if sha and d != sha:
        raise RuntimeError(f"sha mismatch {d[:16]}")
    return {"size_bytes": st, "sha256": d}


GGML_TYPE_NAMES = {0: "f32", 1: "f16", 2: "q4_0", 8: "q8_0", 10: "q2_k",
                   11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k",
                   15: "q8_k", 16: "iq2_xxs", 18: "iq3_xxs", 22: "iq2_s"}


def expert_recipe(path):
    import struct
    with open(path, "rb") as f:
        head = f.read(64 << 20)
    assert head[0:4] == b"GGUF"
    _ver, n_tensors, n_kv = struct.unpack_from("<IQQ", head, 4)
    off = 24

    def read_str(o):
        (n,) = struct.unpack_from("<Q", head, o)
        o += 8
        return head[o:o + n].decode("utf-8", "replace"), o + n

    def skip_value(o, typ):
        if typ in (4, 5, 6):
            return o + 4
        if typ in (10, 11, 12):
            return o + 8
        if typ in (0, 1, 7):
            return o + 1
        if typ in (2, 3):
            return o + 2
        if typ == 8:
            _, o = read_str(o)
            return o
        if typ == 9:
            (at,) = struct.unpack_from("<I", head, o)
            o += 4
            (n,) = struct.unpack_from("<Q", head, o)
            o += 8
            if at == 8:
                for _ in range(n):
                    _, o = read_str(o)
                return o
            return o + {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                        10: 8, 11: 8, 12: 8}[at] * n
        raise RuntimeError(f"kv type {typ}")

    for _ in range(n_kv):
        _, off = read_str(off)
        (typ,) = struct.unpack_from("<I", head, off)
        off = skip_value(off + 4, typ)
    rec = {}
    for _ in range(n_tensors):
        name, off = read_str(off)
        (nd,) = struct.unpack_from("<I", head, off)
        off += 4 + 8 * nd
        (typ,) = struct.unpack_from("<I", head, off)
        off += 12
        if "_exps" in name:
            rec[name] = GGML_TYPE_NAMES.get(typ, f"T{typ}")
    return rec


def run_mmlu(model, dataset, label):
    cmd = [str(PERPLEXITY), "-m", str(model), "-f", str(dataset), "-ngl",
           "0", "-t", str(N_THREADS), "-c", "512", "-b", "256", "-ub",
           "64", "-np", "2", "--multiple-choice", "--multiple-choice-tasks",
           str(N_TASKS), "--no-repack", "--no-warmup"]
    print("+", " ".join(cmd), flush=True)
    t0 = time.monotonic()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=False)
    el = time.monotonic() - t0
    (OUT / f"{label}.stdout.txt").write_text(p.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(p.stderr, encoding="utf-8")
    text = p.stdout + "\n" + p.stderr
    finals = re.findall(r"Final result:\s*([0-9.]+)\s*\+/-\s*([0-9.]+)",
                        text)
    if p.returncode or not finals:
        raise RuntimeError(f"{label} failed rc={p.returncode}: {text[-3000:]}")
    s, sg = finals[-1]
    return {"label": label, "elapsed_sec": el, "score_percent": float(s),
            "sigma_percent": float(sg)}


def wait_port(port, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
            return True
        except OSError:
            time.sleep(2)
    return False


def wait_ready(port, timeout=900):
    """llama-server listens BEFORE the model finishes loading (POSTs get
    503 until slots are ready). Poll a tiny completion until non-503."""
    import urllib.error
    t0 = time.time()
    attempt = 0
    while time.time() - t0 < timeout:
        attempt += 1
        try:
            chat_complete(
                {"messages": [{"role": "user", "content": "Hi"}],
                 "max_tokens": 2, "temperature": 0.0, "seed": 42}, port=port)
            print(f"  server ready after {attempt} probes, "
                  f"{time.time()-t0:.0f}s", flush=True)
            return True
        except urllib.error.HTTPError as e:
            if e.code != 503:
                raise
            time.sleep(20)
        except OSError:
            time.sleep(5)
    return False


def chat_complete(payload, port=8080):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def run_generation(model, prompts, label):
    """prompts: list of {id, text, max_tokens}. Returns list of records."""
    port = 8080
    logf = open(OUT / f"server_{label}.log", "w", encoding="utf-8")
    srv = subprocess.Popen(
        [str(SERVER), "-m", str(model), "--port", str(port), "-t",
         str(N_THREADS), "-c", "2048", "--no-warmup"],
        stdout=logf, stderr=subprocess.STDOUT, text=True)
    try:
        if not wait_port(port, 600):
            raise RuntimeError("server did not listen")
        if not wait_ready(port, 900):
            raise RuntimeError("server never became ready (503 loop?)")
        # warmup: one tiny completion (prefaults, discarded)
        chat_complete({"messages": [{"role": "user", "content": "Hi"}],
                       "max_tokens": 4, "temperature": 0.0, "seed": 42,
                       "enable_thinking": False},
                      port=port)
        import urllib.error
        recs = []
        for pr in prompts:
            t0 = time.monotonic()
            last = None
            for attempt in range(4):
                try:
                    r = chat_complete(
                        {"messages": [{"role": "user",
                                       "content": pr["text"]}],
                         "max_tokens": pr.get("max_tokens", GEN_MAX_TOKENS),
                         "temperature": 0.0, "seed": 42,
                         "enable_thinking": False}, port=port)
                    last = None
                    break
                except urllib.error.HTTPError as e:
                    last = e
                    if e.code != 503 or attempt == 3:
                        raise
                    print(f"  {pr['id']}: 503, retry {attempt+1}/3",
                          flush=True)
                    time.sleep(30)
            if last is not None:
                raise last
            el = time.monotonic() - t0
            msg = r["choices"][0]["message"]
            ch = msg.get("content") or ""
            rs = msg.get("reasoning_content") or ""
            usage = r.get("usage", {})
            recs.append({"id": pr["id"], "prompt": pr["text"],
                         "max_tokens": pr.get("max_tokens", GEN_MAX_TOKENS),
                         "output": ch, "output_no_think": strip_thinking(ch),
                         "reasoning_content": rs,
                         "reasoning_chars": len(rs),
                         "elapsed_sec": el,
                         "completion_tokens": usage.get("completion_tokens"),
                         "prompt_tokens": usage.get("prompt_tokens")})
            print(f"  {label} {pr['id']}: {el:.1f}s "
                  f"ctok={usage.get('completion_tokens')} "
                  f"out_chars={len(ch)} rsn_chars={len(rs)}", flush=True)
            if len(recs) == 8:
                dead = sum(1 for x in recs
                           if not x["output"].strip()
                           and not x["reasoning_content"].strip())
                thinky = sum(1 for x in recs if x["reasoning_chars"] > 500)
                if dead >= 2:
                    raise RuntimeError(
                        "fail-fast: >=2/8 first outputs fully empty")
                if thinky >= 5:
                    print("  WARNING: thinking persists despite "
                          "enable_thinking=false "
                          f"({thinky}/8 with >500 rsn chars); continuing on "
                          "slow path (1200-token budget, grade content only)",
                          flush=True)
        (OUT / f"generation_{label}.json").write_text(
            json.dumps(recs, indent=1), encoding="utf-8")
        return recs
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()
        logf.close()


def main():
    t_start = time.time()
    build()
    gates_info = fetch_gates()
    sw = json.loads((SCRATCH / "swahili_eval_set.json").read_text())
    held = json.loads(
        (SCRATCH / "falcon_probe_heldout.json").read_text())["prompts"]
    meta = json.loads((SCRATCH / "metadata.json").read_text())
    prompts = ([{"id": f"sw{i:02d}",
                 "text": p["query"] + NO_THINK,
                 "max_tokens": GEN_MAX_TOKENS} for i, p in enumerate(sw)] +
               [{"id": h["id"], "text": h["text"] + NO_THINK,
                 "max_tokens": max(h.get("max_tokens", 200), GEN_MAX_TOKENS)}
                for h in held] +
               [{"id": t["prompt_id"], "text": t["prompt"] + NO_THINK,
                 "max_tokens": GEN_MAX_TOKENS + 80}
                for t in meta["test_prompts"]])
    urllib.request.urlretrieve(DATA_URL, SCRATCH / "mmlu-test.bin")
    dataset = SCRATCH / "mmlu-test.bin"

    results = {"schema": "native-sparse-edge0phase2/v1",
               "harness": "v5-enable_thinking-false-1200tok",
               "gates": gates_info, "repo_pin": REPO_PIN,
               "hardware": {"platform": platform.platform(),
                            "cpu_count": os.cpu_count()},
               "models": {}}
    # ---- model A (control), then delete before B (disk-safe) ----
    ma = WORK / A_FILE
    info_a = fetch_model(A_URL, ma, A_SIZE, A_SHA)
    rec_a = expert_recipe(ma)
    n_a = sum(1 for t in rec_a.values() if t in ("iq2_xxs", "iq2_s"))
    mmlu_a = run_mmlu(ma, dataset, "mmlu_A_35")
    gen_a = run_generation(ma, prompts, "A_35")
    results["models"]["A_qwen35"] = {"file": A_FILE, "info": info_a,
                                     "expert_tensors": len(rec_a),
                                     "iq2_expert_tensors": n_a,
                                     "mmlu": mmlu_a,
                                     "n_generation": len(gen_a)}
    ma.unlink()
    # ---- model B (challenger) ----
    mb = WORK / B_FILE
    info_b = fetch_model(B_URL, mb)
    rec_b = expert_recipe(mb)
    n_b = sum(1 for t in rec_b.values() if t in ("iq2_xxs", "iq2_s"))
    same_recipe = (rec_a == rec_b)
    mmlu_b = run_mmlu(mb, dataset, "mmlu_B_36")
    gen_b = run_generation(mb, prompts, "B_36")
    results["models"]["B_qwen36"] = {"file": B_FILE, "info": info_b,
                                     "expert_tensors": len(rec_b),
                                     "iq2_expert_tensors": n_b,
                                     "same_expert_recipe_as_A": same_recipe,
                                     "mmlu": mmlu_b,
                                     "n_generation": len(gen_b)}
    mb.unlink()
    # ---- keyword scoring (swahili only; heldout adjudicated offline) ----
    for tag, gen in (("A_qwen35", gen_a), ("B_qwen36", gen_b)):
        hits, tot = 0, 0
        per = []
        by_id = {g["id"]: g for g in gen}
        for i, p in enumerate(sw):
            out = by_id[f"sw{i:02d}"]["output_no_think"].lower()
            kw = p.get("gold_keywords", [])
            h = sum(1 for k in kw if k.lower() in out)
            hits += h
            tot += len(kw)
            per.append({"id": f"sw{i:02d}", "hits": h, "of": len(kw)})
        results["models"][tag]["swahili_keyword"] = {
            "hits": hits, "of": tot,
            "rate": hits / tot if tot else 0.0, "per_prompt": per}
    results["mmlu_delta_pp_B_minus_A"] = (
        mmlu_b["score_percent"] - mmlu_a["score_percent"])
    results["wall_sec"] = time.time() - t_start
    (OUT / "result.json").write_text(json.dumps(results, indent=2),
                                     encoding="utf-8")
    shutil.copy2(__file__, OUT / Path(__file__).name)
    print(json.dumps({"mmlu_A": mmlu_a["score_percent"],
                      "mmlu_B": mmlu_b["score_percent"],
                      "sw_A": results["models"]["A_qwen35"]["swahili_keyword"]["rate"],
                      "sw_B": results["models"]["B_qwen36"]["swahili_keyword"]["rate"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
