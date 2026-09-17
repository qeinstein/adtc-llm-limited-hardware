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
OUT = WORK / "native-sparse-edge0phase35-v1-results"
for d in (SCRATCH, OUT):
    d.mkdir(parents=True, exist_ok=True)

LLAMA_COMMIT = "3057bb6cf3e9bfc8f2572a2a4c9b7d8a5e6f9e5c"
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
    if not (LLAMA / ".git").exists():
        run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git",
             str(LLAMA)])
    run(["git", "-C", str(LLAMA), "fetch", "--depth", "1", "origin",
         LLAMA_COMMIT])
    run(["git", "-C", str(LLAMA), "checkout", LLAMA_COMMIT])
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
    return {"commit": LLAMA_COMMIT}


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
    n_kv, n_tensors = _st.unpack_from("<QQ", head, 8)
    off = 24

    def read_str(o):
        (n,) = _st.unpack_from("<Q", head, o)
        return head[o + 8:o + 8 + n].decode(), o + 8 + n

    def skip_value(o, typ):
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


def cli_run(model, prompt, label, n_gen, temp, k1=None, k2=None, moe_out=None):
    env = {}
    if k1 is not None:
        env["GGML_MOE_K1"] = str(k1)
    if k2 is not None:
        env["GGML_MOE_K2"] = str(k2)
    if moe_out is not None:
        env["GGML_MOE_OUT"] = str(moe_out)
    cmd = [str(CLI), "-m", str(model), "-p", prompt, "-n", str(n_gen),
           "-t", str(N_THREADS), "-c", "1024", "--temp", str(temp), "-s",
           "42", "--no-warmup", "--no-display-prompt", "-lm", "mmap",
           "-lzm", "off", "--poll", "0"]
    t0 = time.monotonic()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env={**os.environ, **env})
    el = time.monotonic() - t0
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
    build()
    model = WORK / B_FILE
    if not model.exists():
        fetch_file(B_URL, model, B_SIZE, B_SHA)
    ds = SCRATCH / "mmlu-test.bin"
    if not ds.exists():
        urllib.request.urlretrieve(DATA_URL, ds)
    results = {"schema": "native-sparse-edge0phase35/v1", "runtime": runtime,
               "model": {"file": B_FILE, "size": B_SIZE},
               "hardware": {"platform": platform.platform(),
                            "cpu_count": os.cpu_count()}}
    # ---- MMLU arms, priority order ----
    arms = [("mmlu_native", None, None), ("mmlu_k4", 4, 4),
            ("mmlu_k416", 4, 16), ("mmlu_k8exp", 8, 8),
            ("mmlu_k48", 4, 8), ("mmlu_k412", 4, 12),
            ("mmlu_k424", 4, 24), ("mmlu_k432", 4, 32)]
    mmlu = {}
    for label, k1, k2 in arms:
        mmlu[label] = run_mmlu(model, ds, label, k1, k2)
    results["mmlu"] = mmlu
    # ---- Q2K deployment arm ----
    q2k = WORK / "Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"
    results["transcode"] = transcode_q2k(model, q2k)
    results["mmlu_q2k_k416"] = run_mmlu(q2k, ds, "mmlu_q2k_k416", 4, 16)
    # ---- layer probe (moe_out capture) ----
    layer_bins, layer_perf = {}, {}
    for pi, pr in enumerate(LAYER_PROMPTS):
        for cfg, k1, k2 in (("k8", 8, 8), ("k4", 4, 4), ("k416", 4, 16)):
            binp = SCRATCH / f"moe_p{pi}_{cfg}.bin"
            r = cli_run(model, pr, f"layer_p{pi}_{cfg}", 30, 0.0, k1, k2,
                        moe_out=binp)
            layer_bins.setdefault(cfg, {})[pi] = str(binp)
            layer_perf[f"p{pi}_{cfg}"] = r["perf"]
    results["layer_perf"] = layer_perf
    probe = {}
    for pi in range(len(LAYER_PROMPTS)):
        probe[f"p{pi}"] = analyze_layer(
            {c: Path(layer_bins[c][pi]) for c in ("k8", "k4", "k416")})
    results["layer_probe"] = probe
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
        r = cli_run(model, pr["text"], f"san_{pr['id']}_k416", 400, 0.0,
                    4, 16)
        gens.append({"id": pr["id"], "config": "k416",
                     "output": r["output"], "perf": r["perf"]})
    for pr in [sanity[0], sanity[6], sanity[8]]:
        r = cli_run(model, pr["text"], f"san_{pr['id']}_k8", 400, 0.0, 8, 8)
        gens.append({"id": pr["id"], "config": "k8",
                     "output": r["output"], "perf": r["perf"]})
    (OUT / "sanity.json").write_text(json.dumps(gens))
    results["n_sanity"] = len(gens)
    # ---- Q2K decode speed ----
    r = cli_run(q2k, LAYER_PROMPTS[0], "speed_q2k_k416", 60, 0.0, 4, 16)
    results["speed_q2k_k416"] = r["perf"]
    results["wall_sec"] = time.time() - t_start
    (OUT / "result.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: (v["score_percent"] if isinstance(v, dict) and
                          "score_percent" in v else v)
                      for k, v in results["mmlu"].items()}, indent=1),
          flush=True)


if __name__ == "__main__":
    main()
