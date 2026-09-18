"""Phase 3.5: training-free K4 via decoupled k1/k2 normalization (paper 2609.04575).

Qwen3.6-35B-A3B (our Phase-2-selected base). Patches llama.cpp pin 3057bb6:
  (1) build_moe_ffn: GGML_MOE_K1 (executed, dims) + GGML_MOE_K2 (norm mass
      over top-k2, paper Eq.2). Unset env = bit-identical native.
  (2) CPU node loop: post-barrier capture of routed mixture (edge0_moe_out)
      when GGML_MOE_OUT is set (layer-error probe).

Arms (MMLU-200 matched-likelihood, same bin/tasks as v5's 42.0):
  native(unset) | k8-explicit(8,8) | naive(4,4) | (4,8) (4,12) (4,16) (4,24)
  (4,32) | Q2K-experts (4,16). Priority-ordered so early stop keeps the
  decisive comparisons. (8,8)-explicit must equal unset (patch check).

Also: layer-error probe (moe_out relerr/cosine/norm vs K8, prefill-aligned,
2 prompts x 3 configs) + CLI tok/s per config (incl. k2-width cost) +
sanity generations ((4,16) x12 + native spot x3, inspected after pull).

Decision rule (applied after pull; user brief):
  STRONG KEEP: (4,16) within ~1pp of K8 or indistinguishable + no collapse.
  KEEP/RECOVER-LATER: modest 1-3pp loss, competent. RECOVER NOW: >3pp loss
  but big help vs naive (LoRA on THIS student path). NO-GO: catastrophic.

REMAINDER kernel (r-v1): v1 completed all 8 MMLU arms (transcribed in
probes/edge0_port/PHASE3_5.md) then died in transcode_q2k on a GGUF KV
parser bug (scalar metadata types unhandled + n_kv/n_tensors swapped;
both fixed here, regression test tests/test_gguf_kv_skip.py). This run
skips MMLU (REMAINDER=1) and executes transcode -> mmlu_q2k_k416 ->
layer probe -> sanity -> speed, with result.json dumped after EVERY
stage so a crash preserves the completed prefix.

S-KERNEL (s-v1): r-v2 landed transcode + mmlu_q2k_k416 (38.5) then was
SIGKILLed 66min into the FIRST layer-probe run (layer_p0_k8, hook
active) with zero output — mechanism UNKNOWN (hook code reviewed: no
deadlock/leak; post-mortem in PHASE3_5.md). Attach of r-v2 outputs was
REJECTED (ERROR-state kernels have no attachable outputs), so this run
re-transcodes (tested path, ~32min) and runs:
(1) layer SPEED probe, NO hook, 4 threads (tok/s per config);
(2) layer CAPTURE canary, hook, 1 thread (race-free: hook-after-barrier
races the next node on MT; 1T serializes), abort-on-first-hang;
(3) sanity (no hook) + speed_q2k. EVERY subprocess has a timeout,
progress prints, mem logging, and per-run dumps — no more silent death.

S-V2: root-caused the s-v1/r-v2 kills as (probably) a master-era
llama-cli regression: the phase35 scripts carried a TYPO pin
(3057bb6c... instead of 3057bb66...), the fetch/checkout failures were
unchecked, and all phase35 runs built master (~Sept 2026), not the pin
(MMLU deltas stand: same binary across arms). s-v2 re-pins to the TRUE
pin 3057bb66c86c46d5781e50e85462a760ba7d1feb, ASSERTS checkout +
rev-parse, and adds a 60s heartbeat to every cli_run.

S-V3: s-v2 STILL died on the true pin (heartbeat: linear 0.7GB/min
MemAvailable drain 31.9->15.3GB over 24min, then OOM SIGKILL) with our
patch inactive => stock-CLI/flags/file issue, not pin or patch. Diffed
vs the PROVEN edge0trace invocation (23 clean prompts): we lacked
--single-turn (interactive-stdin wait risk), used -c 1024, temp 0.0, no
top-p/perf/ngl. s-v3 adopts the trace invocation VERBATIM + stdin=
DEVNULL (stdin waits impossible) + a 5-token leak canary that ABORTS in
minutes if the drain persists (discriminating flags vs file/patch).
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import struct
import subprocess
import time
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/kaggle_scratch")
OUT = WORK / "native-sparse-edge0phase35s-v1-results"
REMAINDER = True  # s-kernel: v1 MMLU + r-v2 transcode/mmlu_q2k complete
ATTACH = Path("/kaggle/input")  # r-v2 outputs via kernel_sources


def mem_gb():
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    except OSError:
        pass
    return -1.0
# NOTE: no mkdir at import (keeps `import edge0phase35_v1` side-effect-free
# for tests); setup_runtime() creates SCRATCH+OUT before anything needs them.

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
LLAMA = SCRATCH / "llama.cpp"
BUILD = SCRATCH / "build"
PERPLEXITY = BUILD / "bin" / "llama-perplexity"
CLI = BUILD / "bin" / "llama-cli"
QUANTIZE = BUILD / "bin" / "llama-quantize"
N_THREADS = 4
N_TASKS = 200

# Qwen3.6 IQ2_XXS (v5-verified pin)
B_REPO = "a483e9e6cbd595906af30beda3187c2663a1118c"
B_FILE = "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"
B_SIZE = 10_756_586_464
B_SHA = "2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef"
B_URL = ("https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/"
         f"{B_REPO}/{B_FILE}")
DATA_REVISION = "37884b81b4957f1950a53b6ff48d77c8dd5e430c"
DATA_URL = ("https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/"
            f"resolve/{DATA_REVISION}/mmlu-test.bin")

# frozen sanity gates (same pins as edge0phase2 v5)
REPO_PIN = "405b114ed9951b596cbf1b1482c985f8e31dc948"
GATE_FILES = {
    "swahili_eval_set.json":
        "a927c9dd342eef13e196bd5cf2d8ad1d35ff0444b766c2ee9e375c12f868a8d6",
    "falcon_probe_heldout.json":
        "6747c0907e63938f81942219858e1b9cf9fa131c5d915c8664885b2dc851caa4",
}
GATE_PATHS = {"swahili_eval_set.json": "data/swahili_eval_set.json",
              "falcon_probe_heldout.json":
                  "docs/research/falcon_probe_heldout.json"}


def fetch_gates():
    import hashlib
    got = {}
    for name, sha in GATE_FILES.items():
        url = ("https://raw.githubusercontent.com/qeinstein/"
               "adtc-llm-limited-hardware/"
               f"{REPO_PIN}/{GATE_PATHS[name]}")
        dest = SCRATCH / name
        urllib.request.urlretrieve(url, dest)
        d = hashlib.sha256(dest.read_bytes()).hexdigest()
        if d != sha:
            raise RuntimeError(f"gate {name} sha mismatch: {d}")
        got[name] = json.loads(dest.read_text())
    return got

CONVO = [
    "Explain why the sky is blue in two short paragraphs.",
    "Write a friendly reminder email about a dentist appointment tomorrow.",
    "What are three simple ways to save water at home?",
    "Summarize the plot of Cinderella in five sentences.",
    "Give step-by-step instructions for boiling rice.",
    "What is the difference between weather and climate?",
]
LAYER_PROMPTS = [
    "A 3-year-old has watery diarrhea six times today and is very thirsty. "
    "What should the caregiver do right now?",
    "Mtoto ana homa kali na kikohozi. Nifanye nini kabla ya kwenda hospitali?",
]
MOE_MAGIC = 0x34454F4D  # 'MOE4' le


def run(cmd, env=None, log=None, timeout=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=e, timeout=timeout)
    if log:
        log.write_text(f"$ {' '.join(cmd)}\nrc={p.returncode}\n--- out ---\n"
                       f"{p.stdout[-6000:]}\n--- err ---\n{p.stderr[-6000:]}")
    return p


def replace_once(path, old, new):
    t = Path(path).read_text()
    assert t.count(old) == 1, f"anchor x{t.count(old)}: {old[:60]!r}"
    Path(path).write_text(t.replace(old, new, 1))


K1K2_CODE = r'''
    // ---- edge0 k1/k2 (env; default native) ----
    int64_t edge0_k1 = n_expert_used;
    int64_t edge0_k2 = n_expert_used;
    if (const char * e1 = getenv("GGML_MOE_K1")) { int v = atoi(e1); if (v > 0) edge0_k1 = v; }
    if (const char * e2 = getenv("GGML_MOE_K2")) { int v = atoi(e2); if (v > 0) edge0_k2 = v; }
    if (edge0_k2 < edge0_k1) edge0_k2 = edge0_k1;
    if (edge0_k2 > n_expert) edge0_k2 = n_expert;
    n_expert_used = edge0_k1;
'''

K2_NORM_OLD = '''        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);'''

K2_NORM_NEW = '''        ggml_tensor * weights_sum = nullptr;
        if (edge0_k2 == edge0_k1) {
            weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        } else {
            // reference mass over top-k2 (paper 2609.04575 Eq.2 denominator)
            ggml_tensor * sel_k2 = ggml_argsort_top_k(ctx0, selection_probs, (int) edge0_k2); // [k2, T]
            ggml_tensor * rk2 = ggml_get_rows(ctx0, probs, sel_k2); // [1, k2, T]
            rk2 = ggml_reshape_2d(ctx0, rk2, edge0_k2, n_tokens); // [k2, T]
            weights_sum = ggml_sum_rows(ctx0, rk2); // [1, T]
        }
        cb(weights_sum, "ffn_moe_weights_sum", il);'''

MOE_HOOK = r'''
static void edge0_moe_out_hook(struct ggml_tensor * node) {
    const char * path = getenv("GGML_MOE_OUT");
    if (path == NULL || path[0] == '\0') return;
    if (strstr(node->name, "edge0_moe_out_") == NULL) return;
    int layer = -1;
    if (sscanf(node->name, "edge0_moe_out_%d", &layer) != 1) return;
    if (node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) return;
    if (node->ne[0] != 2048) return;
    int64_t ntok = node->ne[1];
    if (ntok < 1 || ntok > 8192) return;
    static FILE * fp = NULL;
    static uint32_t seq = 0;
    if (fp == NULL) {
        fp = fopen(path, "wb");
        if (fp == NULL) return;
        static char buf[1 << 20];
        setvbuf(fp, buf, _IOFBF, sizeof(buf));
    }
    uint8_t hdr[16];
    uint32_t magic = 0x34454F4Du;
    memcpy(hdr + 0, &magic, 4);
    memcpy(hdr + 4, &seq, 4);
    hdr[8] = (uint8_t) layer; hdr[9] = 0;
    uint16_t nt = (uint16_t) ntok;
    memcpy(hdr + 10, &nt, 2);
    hdr[12] = hdr[13] = hdr[14] = hdr[15] = 0;
    fwrite(hdr, 1, 16, fp);
    fwrite(node->data, 4, (size_t)(2048 * ntok), fp);
    seq++;
}
'''


def setup_runtime():
    for d in (SCRATCH, OUT):
        d.mkdir(parents=True, exist_ok=True)
    if not (LLAMA / ".git").exists():
        run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git",
             str(LLAMA)])
    p1 = run(["git", "-C", str(LLAMA), "fetch", "--depth", "1",
                "origin", LLAMA_COMMIT])
    assert p1.returncode == 0, f"pin fetch failed: {p1.stderr[-500:]}"
    p2 = run(["git", "-C", str(LLAMA), "checkout", LLAMA_COMMIT])
    assert p2.returncode == 0, f"pin checkout failed: {p2.stderr[-500:]}"
    g = LLAMA / "src" / "llama-graph.cpp"
    assert g.exists(), "llama checkout failed"
    replace_once(g, "#include <cstring>\n#include <numeric>",
                 "#include <cstdlib>\n#include <cstring>\n#include <numeric>")
    replace_once(g,
                 "    ggml_tensor * logits = nullptr;\n\n    if (probs_in == nullptr) {",
                 K1K2_CODE + "\n    ggml_tensor * logits = nullptr;\n\n    if (probs_in == nullptr) {")
    replace_once(g,
                 "    const uint32_t n_expert_used_il = hparams.n_expert_used(il);",
                 "    const uint32_t n_expert_used_il = (uint32_t) n_expert_used; "
                 "// edge0: follows k1 (uniform arch; native-identical unset)")
    replace_once(g, K2_NORM_OLD, K2_NORM_NEW)
    replace_once(g,
                 '    cb(moe_out, "ffn_moe_out", il);\n\n    return moe_out;',
                 '    cb(moe_out, "ffn_moe_out", il);\n\n'
                 '    if (getenv("GGML_MOE_OUT")) {\n'
                 '        ggml_format_name(moe_out, "edge0_moe_out_%d", il);\n'
                 '    }\n\n    return moe_out;')
    cpu = LLAMA / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
    replace_once(cpu, "static struct ggml_state g_state = {0};\n",
                 "static struct ggml_state g_state = {0};\n" + MOE_HOOK)
    replace_once(cpu,
                 "        if (node_n + 1 < cgraph->n_nodes) {\n"
                 "            ggml_barrier(state->threadpool);\n"
                 "        }\n    }",
                 "        if (node_n + 1 < cgraph->n_nodes) {\n"
                 "            ggml_barrier(state->threadpool);\n"
                 "            if (state->ith == 0) {\n"
                 "                edge0_moe_out_hook(node);\n"
                 "            }\n"
                 "        }\n    }")
    d = run(["git", "-C", str(LLAMA), "diff", "--stat"])
    (OUT / "patch-stat.txt").write_text(d.stdout)
    head = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"],
                          text=True, stdout=subprocess.PIPE).stdout.strip()
    print(f"built base: {head} (want {LLAMA_COMMIT})", flush=True)
    assert head == LLAMA_COMMIT, f"NOT on pin: {head}"
    return {"commit": LLAMA_COMMIT, "built_head": head}


def build():
    run(["cmake", "-S", str(LLAMA), "-B", str(BUILD),
         "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DLLAMA_CURL=ON"],
        log=OUT / "cmake.log")
    p = run(["cmake", "--build", str(BUILD), "--config", "Release",
             f"-j{N_THREADS}", "--target", "llama-cli", "llama-perplexity",
             "llama-quantize"], log=OUT / "build.log")
    if p.returncode:
        raise RuntimeError("build failed; see build.log")


def fetch_file(url, dest, size=None, sha=None):
    import hashlib
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if size and tmp.stat().st_size != size:
        raise RuntimeError(f"size {tmp.stat().st_size} != {size}")
    if sha:
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for ch in iter(lambda: f.read(1 << 20), b""):
                h.update(ch)
        if h.hexdigest() != sha:
            raise RuntimeError("sha mismatch")
    tmp.rename(dest)
    return dest.stat().st_size


def run_mmlu(model, dataset, label, k1=None, k2=None):
    env = {}
    if k1 is not None:
        env["GGML_MOE_K1"] = str(k1)
    if k2 is not None:
        env["GGML_MOE_K2"] = str(k2)
    cmd = [str(PERPLEXITY), "-m", str(model), "-f", str(dataset), "-ngl",
           "0", "-t", str(N_THREADS), "-c", "512", "-b", "256", "-ub",
           "64", "-np", "2", "--multiple-choice", "--multiple-choice-tasks",
           str(N_TASKS), "--no-repack", "--no-warmup"]
    print(f"+ {label} k1={k1} k2={k2}", flush=True)
    t0 = time.monotonic()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env={**os.environ, **env})
    el = time.monotonic() - t0
    (OUT / f"{label}.stdout.txt").write_text(p.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(p.stderr, encoding="utf-8")
    text = p.stdout + "\n" + p.stderr
    finals = re.findall(r"Final result:\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", text)
    if p.returncode or not finals:
        raise RuntimeError(f"{label} failed rc={p.returncode}: {text[-2000:]}")
    s, sg = finals[-1]
    print(f"  {label}: {s} +/- {sg} ({el:.0f}s)", flush=True)
    return {"label": label, "k1": k1, "k2": k2, "elapsed_sec": el,
            "score_percent": float(s), "sigma_percent": float(sg)}


GGML_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 9: "q8_1", 10: "q2_k", 11: "q3_k", 12: "q4_k",
    13: "q5_k", 14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs",
    18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s",
    23: "iq4_xs", 24: "i8", 25: "i16", 26: "i32", 27: "i64", 28: "f64",
    29: "iq1_m", 30: "bf16", 34: "tq1_0", 35: "tq2_0", 36: "mxfp4"}


def gguf_tensor_table(path):
    import struct as _st
    head = open(path, "rb").read(64 << 20)
    assert head[:4] == b"GGUF"
    n_tensors, n_kv = _st.unpack_from("<QQ", head, 8)  # spec order
    off = 24

    def read_str(o):
        (n,) = _st.unpack_from("<Q", head, o)
        return head[o + 8:o + 8 + n].decode(), o + 8 + n

    _SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                 10: 8, 11: 8, 12: 8}  # fixed-width GGUF metadata scalars

    def skip_value(o, typ):
        if typ in _SCALAR:
            return o + _SCALAR[typ]
        if typ in (8,):
            s, o2 = read_str(o)
            return o2
        if typ == 9:
            (at,) = _st.unpack_from("<I", head, o)
            (n,) = _st.unpack_from("<Q", head, o + 4)
            o += 12
            if at == 8:
                for _ in range(n):
                    _, o = read_str(o)
                return o
            return o + {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                        10: 8, 11: 8, 12: 8}[at] * n
        raise RuntimeError(f"kv type {typ}")

    for _ in range(n_kv):
        _, off = read_str(off)
        (typ,) = _st.unpack_from("<I", head, off)
        off = skip_value(off + 4, typ)
    out = []
    for _ in range(n_tensors):
        name, off = read_str(off)
        (nd,) = _st.unpack_from("<I", head, off)
        off += 4 + 8 * nd
        (typ,) = _st.unpack_from("<I", head, off)
        off += 4 + 8
        out.append((name, GGML_TYPE_NAMES.get(typ, f"T{typ}")))
    return out


def transcode_q2k(model_iq2, out_path):
    table = gguf_tensor_table(model_iq2)
    lines, n = [], 0
    for name, typ in table:
        tgt = "q2_k" if "_exps" in name else typ
        n += tgt == "q2_k"
        lines.append(f"^{re.escape(name)}$={tgt}")
    (OUT / "overrides_q2k.txt").write_text("\n".join(lines) + "\n")
    print(f"transcode q2k: {n}/{len(lines)} -> q2_k", flush=True)
    t0 = time.time()
    p = run([str(QUANTIZE), "--allow-requantize", "--tensor-type-file",
             str(OUT / "overrides_q2k.txt"), str(model_iq2), str(out_path),
             "Q8_0", str(N_THREADS)], log=OUT / "transcode_q2k.log")
    if p.returncode:
        raise RuntimeError("transcode failed")
    chk = gguf_tensor_table(out_path)
    nexp = sum(1 for x, t in chk if "_exps" in x and t == "q2_k")
    if nexp != 120:
        raise RuntimeError(f"q2k expert count {nexp} != 120")
    return {"size_bytes": out_path.stat().st_size,
            "elapsed_sec": time.time() - t0, "q2k_experts": nexp}


def cli_run(model, prompt, label, n_gen, temp, k1=None, k2=None,
            moe_out=None, threads=N_THREADS, timeout=1500):
    env = {}
    if k1 is not None:
        env["GGML_MOE_K1"] = str(k1)
    if k2 is not None:
        env["GGML_MOE_K2"] = str(k2)
    if moe_out is not None:
        env["GGML_MOE_OUT"] = str(moe_out)
    # Trace-verbatim invocation (edge0trace ran 23 prompts clean on the pin;
    # our variant without --single-turn died twice). stdin=DEVNULL forbids
    # any interactive wait; heartbeat+timeout bound any recurrence.
    cmd = [str(CLI), "-m", str(model), "-ngl", "0", "-t", str(threads),
           "-c", "512", "-n", str(n_gen), "--temp", str(temp), "--top-p",
           "0.9", "--seed", "42", "--single-turn", "--no-display-prompt",
           "--no-warmup", "--perf", "-lm", "mmap", "-lzm", "off", "--poll",
           "0", "-p", prompt]
    print(f"+ {label} t={threads} mem={mem_gb():.1f}GB", flush=True)
    t0 = time.monotonic()
    import threading
    alive = {"on": True}

    def beat():
        t = 0
        while alive["on"]:
            time.sleep(60)
            t += 60
            if alive["on"]:
                print(f"  {label}: +{t}s mem={mem_gb():.1f}GB", flush=True)

    th = threading.Thread(target=beat, daemon=True)
    th.start()
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                           env={**os.environ, **env}, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  {label}: TIMEOUT after {timeout}s mem={mem_gb():.1f}GB",
              flush=True)
        raise
    finally:
        alive["on"] = False
    el = time.monotonic() - t0
    print(f"  {label}: done {el:.0f}s mem={mem_gb():.1f}GB", flush=True)
    (OUT / f"{label}.stdout.txt").write_text(p.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(p.stderr, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"{label} rc={p.returncode}: {p.stderr[-1500:]}")
    m = re.search(r"eval time =\s*([0-9.]+) ms / \s*([0-9]+) runs", p.stderr)
    assert m, f"{label}: no perf line in stderr: {p.stderr[-800:]}"
    perf = {"ms": float(m.group(1)), "runs": int(m.group(2))}
    perf["tok_s"] = perf["runs"] / (perf["ms"] / 1000)
    return {"label": label, "elapsed_sec": el, "perf": perf,
            "output": p.stdout}


def parse_moe(path):
    raw = open(path, "rb").read()
    off, recs = 0, []
    import numpy as np
    while off < len(raw):
        magic, seq, layer, _, nt, _ = struct.unpack_from("<IIBBHH", raw, off)
        assert magic == MOE_MAGIC, (path, off)
        off += 16
        v = np.frombuffer(raw, dtype=np.float32, count=2048 * nt,
                          offset=off).copy().reshape(nt, 2048)
        off += 2048 * nt * 4
        recs.append({"layer": layer, "nt": nt, "seq": seq, "v": v})
    return recs


def analyze_layer(bins):
    import numpy as np
    cfgs = {}
    for cfg, path in bins.items():
        by_layer = {}
        for r in parse_moe(path):
            by_layer.setdefault(r["layer"], []).append(r)
        cfgs[cfg] = by_layer
    base = cfgs["k8"]
    table = []
    for L in range(40):
        rb = [r for r in base[L] if r["nt"] > 1]
        assert rb, f"L{L}: no prefill record"
        b = rb[0]["v"]
        row = {"layer": L, "ntok": rb[0]["nt"]}
        for cfg in ("k4", "k416"):
            rr = [r for r in cfgs[cfg][L] if r["nt"] > 1][0]["v"]
            assert rr.shape == b.shape, (cfg, L)
            d = rr - b
            row[f"{cfg}_relerr"] = float(
                np.linalg.norm(d) / np.linalg.norm(b))
            row[f"{cfg}_cos"] = float(
                (rr * b).sum() / np.linalg.norm(rr) / np.linalg.norm(b))
            row[f"{cfg}_gain"] = float(
                np.linalg.norm(rr) / np.linalg.norm(b))
        table.append(row)
    rep = {L: table[L] for L in (0, 20, 39)}
    mean = {k: float(np.mean([r[k] for r in table]))
            for k in table[0] if k != "layer" and k != "ntok"}
    return {"per_layer": table, "L0_L20_L39": rep, "mean": mean}


def main():
    t_start = time.time()
    runtime = setup_runtime()
    # Capacity preflight (r-v1 died 40min in on ENOSPC): IQ2 10.76GB +
    # Q2K out ~13.1GB = ~24GB exceeds the 20GB /kaggle/working cap, so
    # the model stages in SCRATCH (no need to persist it) and only the
    # transcode output lives in WORK. Fail in seconds if tight.
    wfree = shutil.disk_usage(WORK).free / 1e9
    sfree = shutil.disk_usage(SCRATCH).free / 1e9
    print(f"disk free: WORK {wfree:.1f}GB SCRATCH {sfree:.1f}GB", flush=True)
    if wfree < 15 or sfree < 15:
        raise RuntimeError(f"disk too tight: WORK {wfree:.1f} SCRATCH {sfree:.1f}")
    build()
    shutil.rmtree(LLAMA / ".git", ignore_errors=True)  # ~1GB back in SCRATCH
    model = SCRATCH / B_FILE  # 10.76GB stages in /tmp (see preflight)
    if not model.exists():
        fetch_file(B_URL, model, B_SIZE, B_SHA)
    # ds (mmlu-test.bin) NOT needed: no MMLU in the s-kernel.
    results = {"schema": "native-sparse-edge0phase35/v1", "runtime": runtime,
               "model": {"file": B_FILE, "size": B_SIZE},
               "hardware": {"platform": platform.platform(),
                            "cpu_count": os.cpu_count()}}
    # ---- MMLU arms, priority order ----
    arms = [("mmlu_native", None, None), ("mmlu_k4", 4, 4),
            ("mmlu_k416", 4, 16), ("mmlu_k8exp", 8, 8),
            ("mmlu_k48", 4, 8), ("mmlu_k412", 4, 12),
            ("mmlu_k424", 4, 24), ("mmlu_k432", 4, 32)]
    def dump():
        results["wall_sec"] = time.time() - t_start
        results["disk_free_gb"] = {
            "work": shutil.disk_usage(WORK).free / 1e9,
            "scratch": shutil.disk_usage(SCRATCH).free / 1e9}
        (OUT / "result.json").write_text(json.dumps(results, indent=1))

    mmlu = {}
    if REMAINDER:
        mmlu = {"note": "8 arms done in v1 (see PHASE3_5.md); skipped here"}
    else:
        for label, k1, k2 in arms:
            mmlu[label] = run_mmlu(model, ds, label, k1, k2)
    results["mmlu"] = mmlu
    dump()
    # ---- attach r-v2 outputs (q2k file + result.json) ----
    ggufs = sorted(ATTACH.rglob("*.gguf")) if ATTACH.exists() else []
    rj = sorted(ATTACH.rglob("result.json")) if ATTACH.exists() else []
    print(f"attached: {len(ggufs)} gguf {[g.name for g in ggufs]} "
          f"mem={mem_gb():.1f}GB", flush=True)
    if rj:
        prev = json.loads(rj[0].read_text())
        print(f"r-v2 keys: {sorted(prev.keys())} "
              f"q2k={prev.get('mmlu_q2k_k416', {}).get('score_percent')}",
              flush=True)
        results["r_v2"] = {k: prev.get(k) for k in
                           ("mmlu_q2k_k416", "transcode", "wall_sec")}
    if ggufs:
        q2k = ggufs[0]
        print(f"using attached q2k: {q2k} ({q2k.stat().st_size/1e9:.2f}GB)",
              flush=True)
    else:
        print("NO attached q2k; fallback transcode (slow path)", flush=True)
        q2k = WORK / "Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"
        results["transcode_fallback"] = transcode_q2k(model, q2k)
    dump()
    # ---- leak canary (s-v2: 0.7GB/min drain -> OOM at ~25min) ----
    # Trace-verbatim flags + DEVNULL stdin should be clean (23 prompts
    # proven). If the drain persists here, the cause is file/patch, not
    # flags -> fail FAST with the answer instead of burning 2h.
    m0 = mem_gb()
    rc = cli_run(model, LAYER_PROMPTS[0], "canary_n5", 5, 0.7, 8, 8,
                 timeout=600)
    results["canary"] = {"perf": rc["perf"], "mem_before": m0,
                         "mem_after": mem_gb()}
    dump()
    if m0 - mem_gb() > 3.0:
        raise RuntimeError(
            f"LEAK PERSISTS with trace flags: {m0:.1f}->{mem_gb():.1f}GB "
            f"on a 5-token run; cause is file/patch, not flags.")
    print(f"canary clean: mem {m0:.1f}->{mem_gb():.1f}GB", flush=True)
    # ---- layer SPEED probe (NO hook: r-v2 died with hook active) ----
    layer_perf = {}
    for pi, pr in enumerate(LAYER_PROMPTS):
        for cfg, k1, k2 in (("k8", 8, 8), ("k4", 4, 4), ("k416", 4, 16)):
            r = cli_run(model, pr, f"speed_p{pi}_{cfg}", 30, 0.7, k1, k2)
            layer_perf[f"p{pi}_{cfg}"] = r["perf"]
            results["layer_perf"] = layer_perf
            dump()
    # ---- layer CAPTURE canary (hook, 1 thread = race-free) ----
    layer_bins, cap_ok = {}, True
    for pi, pr in enumerate(LAYER_PROMPTS):
        for cfg, k1, k2 in (("k8", 8, 8), ("k4", 4, 4), ("k416", 4, 16)):
            binp = SCRATCH / f"moe_p{pi}_{cfg}.bin"
            try:
                r = cli_run(model, pr, f"layer_p{pi}_{cfg}", 30, 0.7,
                            k1, k2, moe_out=binp, threads=1, timeout=900)
            except subprocess.TimeoutExpired:
                print(f"CAPTURE ABORT at p{pi}_{cfg} (hook path hangs); "
                      f"keeping speed results", flush=True)
                results["capture_aborted_at"] = f"p{pi}_{cfg}"
                cap_ok = False
                break
            layer_bins.setdefault(cfg, {})[pi] = str(binp)
        if not cap_ok:
            break
    if cap_ok:
        probe = {}
        for pi in range(len(LAYER_PROMPTS)):
            probe[f"p{pi}"] = analyze_layer(
                {c: Path(layer_bins[c][pi]) for c in ("k8", "k4", "k416")})
        results["layer_probe"] = probe
    dump()
    # ---- sanity generations ----
    gates = fetch_gates()
    sw = gates["swahili_eval_set.json"]
    held = gates["falcon_probe_heldout.json"]["prompts"]
    want = {"h05", "h07", "h12", "h22"}
    hsel = [h for h in held if h["id"] in want]
    assert len(hsel) == 4, [h["id"] for h in held][:30]
    sanity = ([{"id": f"c{i:02d}", "text": t} for i, t in enumerate(CONVO)] +
              [{"id": "sw00", "text": sw[0]["query"]},
               {"id": "sw01", "text": sw[1]["query"]}] +
              [{"id": h["id"], "text": h["text"]} for h in hsel])
    gens = []
    for pr in sanity:
        r = cli_run(model, pr["text"], f"san_{pr['id']}_k416", 400, 0.7,
                    4, 16)
        gens.append({"id": pr["id"], "config": "k416",
                     "output": r["output"], "perf": r["perf"]})
        (OUT / "sanity.json").write_text(json.dumps(gens))
        results["n_sanity"] = len(gens)
        dump()
    for pr in [sanity[0], sanity[6], sanity[8]]:
        r = cli_run(model, pr["text"], f"san_{pr['id']}_k8", 400, 0.7, 8, 8)
        gens.append({"id": pr["id"], "config": "k8",
                     "output": r["output"], "perf": r["perf"]})
        (OUT / "sanity.json").write_text(json.dumps(gens))
        results["n_sanity"] = len(gens)
        dump()
    results["n_sanity"] = len(gens)
    model.unlink(missing_ok=True)  # last model use done; 10.76GB back
    print("freed IQ2 model from SCRATCH", flush=True)
    dump()
    # ---- Q2K decode speed ----
    r = cli_run(q2k, LAYER_PROMPTS[0], "speed_q2k_k416", 60, 0.7, 4, 16)
    results["speed_q2k_k416"] = r["perf"]
    dump()
    print(json.dumps({k: (v["score_percent"] if isinstance(v, dict) and
                          "score_percent" in v else v)
                      for k, v in results["mmlu"].items()}, indent=1),
          flush=True)


if __name__ == "__main__":
    main()
