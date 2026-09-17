"""Edge0-trace pilot: per-token/per-layer post-attn hidden states + router
top-8 for OUR OWN prerouter training/measurement (Phase 3/4 infrastructure).

LAUNCH STATUS: Phase 2 decided 3.6 (efficiency tie-break); BASE="3.6".
Hook anchors verified exactly-once in pin 3057bb6; cmake = v5 proven recipe.
No quality assumption; no eval data used (training-distribution prompts).

Method (mirrors proven route-corpus hook, same pin):
  - patch ggml-cpu dispatch (pre-computation): on MUL_MAT_ID with
    ffn_up_exps (or fused ffn_gate_up_exps) weights at decode shape,
    dump src[1] (the post-attn-norm hidden, 2048 f32 -> fp16 bits);
  - on MUL_MAT_ID with ffn_down_exps weights, dump the ids row (i32 x8)
    for offline-topk parity (kind=2 records).
  - router logits/topk are recomputed OFFLINE from the F32 gate weights
    (exact math, no quant noise) and parity-checked vs kind-2 ids.
Prompts: the 32 route-corpus prompts (training distribution; disjoint
from all eval/holdout sets by construction — same list, same purpose).
Output per prompt: npz {hidden_fp16 [T,40,2048], topk_ids/probs [T,40,8]}
+ manifest + result.json. ~80 tok/prompt, ~2500 total, ~0.8 GB.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import platform
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-edge0trace-v1")
OUT = WORK / "native-sparse-edge0trace-v1-results"
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
CLI = BUILD / "bin" / "llama-cli"

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
THREADS = 4
N_GEN = 80

# BASE SWITCH (Phase 2 verdict: TIE on quality, 3.6 wins on efficiency —
# ~27% shorter thinking; see probes/edge0_port/PHASE2_REPORT.md)
BASE = "3.6"
MODELS = {
    "3.5": {
        "repo": "bc014a17be43adabd7066b7a86075ff935c6a4e2",
        "file": "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf",
        "size": 10_656_955_008,
        "sha": "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b",
        "hf": "unsloth/Qwen3.5-35B-A3B-GGUF",
    },
    "3.6": {
        "repo": "a483e9e6cbd595906af30beda3187c2663a1118c",
        "file": "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
        "size": 10_756_586_464,
        "sha": "2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef",
        "hf": "unsloth/Qwen3.6-35B-A3B-GGUF",
    },
}

# training-distribution prompts (copied from route-corpus kernel; NOT eval)
PROMPTS = [
    ("medical", "A 3-year-old has watery diarrhoea and is thirsty. Explain how to assess dehydration and what home treatment to start."),
    ("medical", "A pregnant woman has headache, swollen face, and high blood pressure. What danger signs and referral steps apply?"),
    ("medical", "List the steps to prepare oral rehydration solution at home and when to seek care."),
    ("medical", "A newborn feels cold and feeds poorly. Describe warming, feeding, and referral actions."),
    ("medical", "Explain why deworming tablets are given at school and what side effects to watch for."),
    ("medical", "A child has fast breathing and chest indrawing. Outline assessment and urgent actions."),
    ("general", "Explain how rain forms and why some seasons are drier than others."),
    ("general", "Describe how a market day works in a small town, from morning to evening."),
    ("general", "Summarize the plot of a story about a fisherman who finds something unexpected."),
    ("general", "Compare travelling by bus and by motorcycle taxi for a long trip."),
    ("general", "Explain what a budget is and how a family can make one."),
    ("general", "Describe the steps to plant maize from land preparation to harvest."),
    ("swahili", "Eleza kwa kifupi kwa nini maji ya kunywa yenye chumvi na sukari husaidia mtoto mwenye kuharisha."),
    ("swahili", "Ni dalili zipi za hatari zinahitaji mtoto apelekwe hospitali haraka?"),
    ("swahili", "Andika mpango mfupi wa usafi wa mikono kwa shule ya kijijini."),
    ("swahili", "Tofautisha ukweli, maoni, na dhana kwa lugha rahisi ya Kiswahili."),
    ("swahili", "Mweleze mgonjwa kwa heshima kwa nini anatakiwa kufuata dozi aliyopewa."),
    ("instruction", "Answer in exactly three bullet points: how to prepare for a job interview."),
    ("instruction", "Give a two-sentence answer and one caveat about using online medical advice."),
    ("instruction", "Create a small table comparing prevention, diagnosis, and treatment."),
    ("instruction", "Ask one useful clarification question before answering an underspecified request."),
    ("short", "Hello."),
    ("short", "What should I do next?"),
]

MAGIC = 0x54524143  # 'TRAC'
# magic u32, pid u16, seq u32, layer u8, kind u8, n u16, reserved u16 (=16B)
REC_HDR = "<IHIBBHH"
REC_HDR_SZ = struct.calcsize(REC_HDR)
assert REC_HDR_SZ == 16


def run_checked(cmd, cwd=None, log=None):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       check=False)
    if log is not None:
        log.write_text(p.stdout + "\n--- STDERR ---\n" + p.stderr,
                       encoding="utf-8")
    if p.returncode:
        print(p.stdout[-3000:], flush=True)
        print(p.stderr[-3000:], flush=True)
        raise RuntimeError(f"command failed ({p.returncode})")
    return p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def replace_once(path, old, new):
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"patch anchor mismatch in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


HOOK = r'''
static void edge0_trace_hook(const struct ggml_compute_params * params,
                             const struct ggml_tensor * tensor) {
    if (params->ith != 0) return;
    if (tensor->op != GGML_OP_MUL_MAT_ID) return;
    const char * path = getenv("GGML_EDGE0_TRACE");
    if (path == NULL || path[0] == '\0') return;
    const struct ggml_tensor * weights = tensor->src[0];
    if (weights == NULL) return;
    int is_up = strstr(weights->name, "ffn_up_exps") != NULL;
    int is_fused = strstr(weights->name, "ffn_gate_up_exps") != NULL;
    int is_down = strstr(weights->name, "ffn_down_exps") != NULL;
    if (!is_up && !is_fused && !is_down) return;
    int layer = -1;
    if (sscanf(weights->name, "blk.%d.", &layer) != 1) return;
    static FILE * fp = NULL;
    static uint32_t seq = 0;
    if (fp == NULL) {
        fp = fopen(path, "ab");
        if (fp == NULL) return;
        static char buf[1 << 20];
        setvbuf(fp, buf, _IOFBF, sizeof(buf));
    }
    if (is_down) {
        const struct ggml_tensor * ids = tensor->src[2];
        if (ids == NULL || ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids)) return;
        if (ids->ne[0] != 8 || ids->ne[1] != 1) return; // decode only
        uint8_t hdr[16];
        uint32_t magic = 0x54524143;
        memcpy(hdr, &magic, 4);
        uint16_t pid = 0, zero = 0;
        const char * pe = getenv("GGML_EDGE0_PID");
        if (pe) pid = (uint16_t)atoi(pe);
        memcpy(hdr + 4, &pid, 2);
        memcpy(hdr + 6, &seq, 4);
        seq++;
        hdr[10] = (uint8_t)layer; hdr[11] = 2;
        uint16_t n = 8;
        memcpy(hdr + 12, &n, 2);
        memcpy(hdr + 14, &zero, 2);
        fwrite(hdr, 1, 16, fp);
        fwrite(ids->data, 4, 8, fp);
        return;
    }
    // hidden state = MUL_MAT_ID input (post-attn-norm MoE input)
    const struct ggml_tensor * act = tensor->src[1];
    if (act == NULL || act->type != GGML_TYPE_F32 || !ggml_is_contiguous(act)) return;
    if (act->ne[0] != 2048 || act->ne[1] != 1) return; // decode only
    const float * x = (const float *) act->data;
    uint8_t hdr[16];
    uint32_t magic = 0x54524143;
    memcpy(hdr, &magic, 4);
    uint16_t pid = 0, zero = 0;
    const char * pe = getenv("GGML_EDGE0_PID");
    if (pe) pid = (uint16_t)atoi(pe);
    memcpy(hdr + 4, &pid, 2);
    memcpy(hdr + 6, &seq, 4);
    seq++;
    hdr[10] = (uint8_t)layer; hdr[11] = 0;
    uint16_t n = 2048;
    memcpy(hdr + 12, &n, 2);
    memcpy(hdr + 14, &zero, 2);
    fwrite(hdr, 1, 16, fp);
    static _Thread_local ggml_fp16_t hbuf[2048];
    ggml_fp32_to_fp16_row(x, hbuf, 2048);
    fwrite(hbuf, 2, 2048, fp);
}
'''


def setup_runtime():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not LLAMA.exists():
        run_checked(["git", "clone", "--filter=blob:none",
                     "https://github.com/ggml-org/llama.cpp.git",
                     str(LLAMA)])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT],
                cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    cpu = LLAMA / "ggml/src/ggml-cpu/ggml-cpu.c"
    replace_once(cpu, "static struct ggml_state g_state = {0};\n",
                 HOOK + "\nstatic struct ggml_state g_state = {0};\n")
    replace_once(cpu,
                 "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
                 "        return;\n"
                 "    }\n"
                 "\n"
                 "    // extra_buffer op?\n",
                 "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
                 "        return;\n"
                 "    }\n"
                 "\n"
                 "    edge0_trace_hook(params, tensor);\n"
                 "\n"
                 "    // extra_buffer op?\n")
    patch = run_checked(["git", "diff", "--", "ggml/src/ggml-cpu/ggml-cpu.c"],
                        cwd=LLAMA).stdout
    (OUT / "trace-runtime.patch").write_text(patch, encoding="utf-8")
    return {"llama_commit": LLAMA_COMMIT,
            "runtime_patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}


def build():
    # cmake recipe = edge0phase2 v5's EXACT proven flags (built 4+ clean runs
    # on this image); no unproven deltas.
    run_checked(["cmake", "-S", str(LLAMA), "-B", str(BUILD),
                 "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON",
                 "-DLLAMA_CURL=ON"],
                log=OUT / "cmake-configure.log")
    run_checked(["cmake", "--build", str(BUILD), "--config", "Release",
                 "-j4", "--target", "llama-cli"],
                log=OUT / "cmake-build.log")


def fetch_model():
    m = MODELS[BASE]
    url = f"https://huggingface.co/{m['hf']}/resolve/{m['repo']}/{m['file']}"
    dest = WORK / m["file"]
    if not dest.exists() or dest.stat().st_size != m["size"]:
        run_checked(["curl", "-L", "--fail", "--retry", "5",
                     "--retry-delay", "5", "-C", "-", "-o", str(dest), url],
                    log=OUT / "model-download.log")
    d = sha256(dest)
    if dest.stat().st_size != m["size"] or d != m["sha"]:
        raise RuntimeError("model identity mismatch")
    return {"base": BASE, "file": m["file"], "size_bytes": m["size"],
            "sha256": d, "repo_commit": m["repo"]}


def parse_records(path):
    """Returns list of dicts; validates magic/framing."""
    raw = path.read_bytes()
    recs, off = [], 0
    while off < len(raw):
        magic, pid, seq, layer, kind, n, _ = struct.unpack_from(
            REC_HDR, raw, off)
        if magic != MAGIC:
            raise RuntimeError(f"bad magic at {off} in {path.name}")
        off += REC_HDR_SZ
        if kind == 0:
            vals = np.frombuffer(raw, dtype=np.uint16, count=n, offset=off)
            off += n * 2
            recs.append({"pid": pid, "seq": seq, "layer": layer,
                         "kind": kind, "raw": vals.copy()})
        elif kind == 2:
            vals = np.frombuffer(raw, dtype=np.int32, count=n, offset=off)
            off += n * 4
            recs.append({"pid": pid, "seq": seq, "layer": layer,
                         "kind": kind, "vals": vals.copy()})
        else:
            raise RuntimeError(f"bad kind {kind}")
    return recs


def match_ids_to_sweeps(h, ids, nt, n_layers=40):
    """Match runtime-ids records to hidden-eval sweeps by global seq order.

    h: hidden records (nt sweeps x n_layers, layer runs in order).
    ids: runtime-ids records (any subset; v1 showed prefix sweeps may
    lack ids entirely — a llama.cpp batching quirk, hidden still valid).
    Returns {layer: [(eval_t, ids8), ...]}; raises on framing violations
    (out-of-range or duplicate (layer, eval) matches).
    """
    assert len(h) == nt * n_layers, (len(h), nt)
    assert [r["layer"] for r in h[:n_layers]] == list(range(n_layers))
    sweep_start = [h[t * n_layers]["seq"] for t in range(nt)]
    matched, seen = {}, set()
    for r in ids:
        t = bisect.bisect_right(sweep_start, r["seq"]) - 1
        assert 0 <= t < nt, r["seq"]
        assert (r["layer"], t) not in seen, (r["layer"], t)
        seen.add((r["layer"], t))
        matched.setdefault(r["layer"], []).append((t, r["vals"]))
    return matched


def read_gate_weights(model_path):
    """F32 router gates per layer from the GGUF: {layer: (256,2048) f32}."""
    import struct as st
    with open(model_path, "rb") as f:
        head = f.read(64 << 20)
    assert head[0:4] == b"GGUF"
    _ver, n_tensors, n_kv = st.unpack_from("<IQQ", head, 4)
    off = 24

    def read_str(o):
        (n,) = st.unpack_from("<Q", head, o)
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
            (at,) = st.unpack_from("<I", head, o)
            o += 4
            (n,) = st.unpack_from("<Q", head, o)
            o += 8
            if at == 8:
                for _ in range(n):
                    _, o = read_str(o)
                return o
            return o + {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                        10: 8, 11: 8, 12: 8}[at] * n
        raise RuntimeError("kv")

    for _ in range(n_kv):
        _, off = read_str(off)
        (typ,) = st.unpack_from("<I", head, off)
        off = skip_value(off + 4, typ)
    # tensor infos
    infos = []
    for _ in range(n_tensors):
        name, off = read_str(off)
        (nd,) = st.unpack_from("<I", head, off)
        off += 4
        shape = st.unpack_from("<" + "Q" * nd, head, off)
        off += 8 * nd
        (typ,) = st.unpack_from("<I", head, off)
        off += 4
        (to,) = st.unpack_from("<Q", head, off)
        off += 8
        infos.append((name, shape, typ, to))
    # data start (align 32)
    data_start = off
    while data_start % 32:
        data_start += 1
    # fix: header may exceed 64MB read; offsets are from data start; recompute
    # data start by scanning (GGUF: tensor data follows header, 32-aligned)
    data_start = off + (32 - off % 32) % 32
    gates = {}
    with open(model_path, "rb") as f:
        for name, shape, typ, to in infos:
            if ".ffn_gate_inp." in name and typ == 0 and len(shape) == 2:
                layer = int(name.split("blk.")[1].split(".")[0])
                f.seek(data_start + to)
                n = int(shape[0] * shape[1])
                w = np.frombuffer(f.read(n * 4), dtype=np.float32)
                assert tuple(shape) == (2048, 256), (name, shape)
                gates[layer] = w.reshape(256, 2048).copy()
    if len(gates) != 40:
        raise RuntimeError(f"gates found: {len(gates)}")
    return gates


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runtime = setup_runtime()
    build()
    model = fetch_model()
    gates = read_gate_weights(WORK / MODELS[BASE]["file"])
    prompt_results, all_toks = [], 0
    for pid, (category, prompt) in enumerate(PROMPTS):
        trace = OUT / f"trace_p{pid:02d}.bin"
        trace.unlink(missing_ok=True)
        cmd = [str(CLI), "-m", str(WORK / MODELS[BASE]["file"]), "-ngl",
               "0", "-t", str(THREADS), "-c", "512", "-n", str(N_GEN),
               "--temp", "0.7", "--top-p", "0.9", "--seed", str(1000 + pid),
               "--single-turn", "--no-display-prompt", "--no-warmup",
               "--perf", "-lm", "mmap", "-lzm", "off", "--poll", "0",
               "-p", prompt]
        env = dict(os.environ, GGML_EDGE0_TRACE=str(trace),
                   GGML_EDGE0_PID=str(pid))
        print(f"===== prompt {pid:02d} {category} =====", flush=True)
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, env=env, check=False)
        (OUT / f"prompt_{pid:02d}.stdout.txt").write_text(p.stdout,
                                                         encoding="utf-8")
        (OUT / f"prompt_{pid:02d}.stderr.txt").write_text(p.stderr,
                                                         encoding="utf-8")
        if p.returncode != 0:
            raise RuntimeError(f"prompt {pid} failed: {p.stderr[-2000:]}")
        recs = parse_records(trace) if trace.exists() else []
        # frame: hidden arrives in layer runs of 40 per eval; runtime ids
        # are matched to hidden-evals by SEQ ORDER (v1 post-mortem: the
        # first 3 evals of prompt 0 carried hidden but no ids — a llama.cpp
        # batching quirk, hidden RMS-proven valid; ids are only a parity
        # cross-check, labels always come from exact offline router math).
        h = [r for r in recs if r["kind"] == 0]
        ids = [r for r in recs if r["kind"] == 2]
        nt = len(h) // 40
        assert len(h) % 40 == 0, (pid, len(h))
        matched = match_ids_to_sweeps(h, ids, nt)
        # fp16 -> f32 (numpy) + RMS validation (post-norm => ~1.0)
        H = np.stack([(r["raw"].copy().view(np.float16).astype(np.float32))
                      for r in h]).reshape(nt, 40, 2048)
        rms = float(np.sqrt((H ** 2).mean()))
        if not 0.3 < rms < 3.0:
            raise RuntimeError(f"prompt {pid}: hidden RMS {rms} (wrong tensor?)")
        # offline router: logits = W @ h, softmax, top8, renorm
        topk_ids = np.zeros((nt, 40, 8), np.int32)
        topk_pr = np.zeros((nt, 40, 8), np.float32)
        for t in range(nt):
            for L in range(40):
                z = gates[L] @ H[t, L]
                z = z - z.max()
                e = np.exp(z)
                pr = e / e.sum()
                idx = np.argpartition(-pr, 8)[:8]
                idx = idx[np.argsort(-pr[idx])]
                topk_ids[t, L] = idx
                topk_pr[t, L] = pr[idx] / pr[idx].sum()
        # parity vs hooked runtime ids, over SEQ-MATCHED pairs only
        par = tot = 0
        for L in range(40):
            for t, v in matched.get(L, []):
                tot += 1
                if (np.sort(v) == np.sort(topk_ids[t, L])).all():
                    par += 1
        coverage = tot / (nt * 40)
        parity = par / tot if tot else 0.0
        if coverage < 0.5:
            raise RuntimeError(f"pid {pid}: ids coverage {coverage} < 0.5")
        np.savez_compressed(OUT / f"trace_p{pid:02d}.npz",
                            hidden_fp16=H.astype(np.float16),
                            topk_ids=topk_ids, topk_probs=topk_pr)
        trace.unlink()
        all_toks += nt
        prompt_results.append({"prompt_id": pid, "category": category,
                               "trace_evals": nt, "hidden_rms": rms,
                               "ids_coverage": coverage,
                               "topk_parity": parity})
        print(f"  pid={pid} tok={nt} rms={rms:.3f} cov={coverage:.4f} "
              f"parity={parity:.4f}", flush=True)
        # NOTE: threshold is 0.999, not 1.0 — fp16 rounding of the dumped
        # hidden can flip a rank-8 boundary tie on rare tokens. A wrong
        # tensor/orientation would score LOW parity, so 0.999 still guards
        # the failure mode without aborting on quant noise.
        if parity < 0.999:
            raise RuntimeError(f"pid {pid}: topk parity {parity} < 0.999")
    result = {"schema": "native-sparse-edge0trace/v1", "status": "ok",
              "runtime": runtime, "model": model,
              "hardware": {"platform": platform.platform(),
                           "cpu_count": os.cpu_count()},
              "decode": {"tokens_requested_per_prompt": N_GEN,
                         "prompts": len(PROMPTS), "trace_tokens": all_toks},
              "prompt_results": prompt_results,
              "wall_sec": time.time() - started}
    (OUT / "result.json").write_text(json.dumps(result, indent=2),
                                     encoding="utf-8")
    print(json.dumps({"status": "ok", "tokens": all_toks}), flush=True)


if __name__ == "__main__":
    main()
