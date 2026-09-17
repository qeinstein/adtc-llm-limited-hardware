"""Kaggle: dynskip contextual validation - L00/L20/L39 decode dumps + frozen L20 policy A/B.

Single kernel, exact Qwen3.5-35B-A3B-UD-IQ2_XXS revision, pinned llama.cpp.
Same 32 prompts as native-sparse-l20hdump-v1 (prompt-disjoint held split reused
locally: 22 train / 10 held). Greedy decode (temp 0) for exact flip measurement.

Phase C (control, GGML_DYNSKIP=2 log-only): per decode token, dump for L00/L20/L39:
  x  = MoE input            (router MUL_MAT src[1], proven hook)
  h0 = block input          (attn_norm RMS_NORM src[0])
  h1 = post-mixer residual  (post_attention_norm RMS_NORM src[0])
plus route-hook ids (all 40 layers) and generated text.
Phase P (policy, GGML_DYNSKIP=1 armed): L20 routed experts EXACT-ZEROED
numerically (weights row -> 0, a numerical twin of skipping) iff the runtime
pre-norm top-8 slot-0 weight (== full-distribution top1 prob, topk is sorted)
is <= 0.02. Records skip log + routes (must equal control ids: routing
semantics preserved) + generated text (exact flip measurement vs control).

Observation hooks fail loud (mechanical asserts per prompt); the skip patch is
env-gated and control runs never mutate (log-only mode + determinism re-run).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tarfile
import time
from pathlib import Path


SCRATCH = Path("/tmp/native-sparse-dynskip-ctx")
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
CLI = BUILD / "bin" / "llama-cli"
OUT = Path("/kaggle/working/native-sparse-dynskip-ctx-v1-results")
LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL = SCRATCH / "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_SHA256 = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
THREADS = 4
N_GEN = 72
LAYERS = (0, 20, 39)
KINDS = ("x", "h0", "h1", "B")

PROMPTS = [
    ("clinical", "Explain why oral rehydration solution helps a child with watery diarrhoea."),
    ("clinical", "What are the danger signs in a child with pneumonia that require urgent care?"),
    ("clinical", "Give a short differential diagnosis for fever and neck stiffness."),
    ("clinical", "How should a community health worker counsel a mother about malaria prevention?"),
    ("clinical", "A patient has chest pain after exertion. What key questions should be asked first?"),
    ("clinical", "Describe safe first aid for a minor burn while waiting for medical review."),
    ("mcqa", "Which intervention most directly prevents cholera: A clean water B antibiotics C vitamins D bed rest?"),
    ("mcqa", "A patient is hypotensive and tachycardic. Which finding is most concerning? A dry lips B shock C mild cough D rash."),
    ("mcqa", "What is the best first step for suspected hypoglycaemia: A give glucose B restrict fluids C induce vomiting D give iron?"),
    ("mcqa", "Which organ primarily filters blood: A heart B liver C kidney D lung?"),
    ("reasoning", "A clinic has two nurses and four rooms. Propose a simple morning patient-flow plan and explain it."),
    ("reasoning", "Compare the risks and benefits of sending a patient to hospital versus monitoring at home."),
    ("reasoning", "If a medicine works for 8 hours and is taken twice daily, explain a practical schedule."),
    ("reasoning", "A village has intermittent clean water. Suggest a low-cost plan with immediate and long-term actions."),
    ("reasoning", "What evidence would distinguish a true outbreak from an increase in reporting?"),
    ("general", "Write a concise explanation of why the sky appears blue."),
    ("general", "How do I make a weekly study plan when I work irregular hours?"),
    ("general", "Give three polite ways to ask someone to clarify their request."),
    ("general", "Summarize the idea of opportunity cost in plain language."),
    ("general", "What is the difference between a fact, an opinion, and a hypothesis?"),
    ("general", "Write a short encouraging message for someone learning a difficult skill."),
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


def run_checked(cmd: list[str], *, cwd: Path | None = None, log: Path | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if log is not None:
        log.write_text(p.stdout + "\n--- STDERR ---\n" + p.stderr, encoding="utf-8")
    if p.returncode:
        print(p.stdout[-3000:], flush=True)
        print(p.stderr[-3000:], flush=True)
        raise RuntimeError(f"command failed ({p.returncode})")
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"patch anchor count mismatch in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")

HOOKS = r'''
static void dynskip_route_trace(const struct ggml_compute_params * params,
                                const struct ggml_tensor * tensor) {
    if (params->ith != 0 || tensor->op != GGML_OP_MUL_MAT_ID) return;
    const char * path = getenv("GGML_DYNSKIP_ROUTES");
    if (path == NULL || path[0] == '\0') return;
    const struct ggml_tensor * weights = tensor->src[0];
    const struct ggml_tensor * ids = tensor->src[2];
    if (weights == NULL || ids == NULL || ids->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(ids) || strstr(weights->name, "ffn_down_exps") == NULL) return;
    static FILE * fp = NULL;
    static uint64_t event_id = 0;
    if (fp == NULL) {
        fp = fopen(path, "a");
        if (fp == NULL) return;
        setvbuf(fp, NULL, _IOLBF, 0);
    }
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const int64_t n = ggml_nelements(ids);
    const int32_t * values = (const int32_t *) ids->data;
    fprintf(fp, "{\"event\":%" PRIu64 ",\"mono_ns\":%" PRIu64
        ",\"weight\":\"%s\",\"weight_nbytes\":%zu,\"shape\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] ,\"ids\":[",
        event_id++, (uint64_t)ts.tv_sec * (uint64_t)1000000000 + (uint64_t)ts.tv_nsec,
        weights->name, ggml_nbytes(weights), ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3]);
    for (int64_t i = 0; i < n; ++i) fprintf(fp, "%s%d", i ? "," : "", values[i]);
    fprintf(fp, "]}\n");
}

static void dynskip_state_dump(const struct ggml_compute_params * params,
                               const struct ggml_tensor * tensor) {
    if (params->ith != 0) return;
    const char * pre = getenv("GGML_DYNSKIP_PREFIX");
    if (pre == NULL || pre[0] == '\0') return;
    const struct ggml_tensor * weights = NULL;
    const struct ggml_tensor * act = NULL;
    const char * kind = NULL;
    int layer = -1;
    if (tensor->op == GGML_OP_MUL_MAT) {
        weights = tensor->src[0];
        act = tensor->src[1];
        if (weights == NULL || act == NULL || act->type != GGML_TYPE_F32) return;
        if      (!strcmp(weights->name, "blk.0.ffn_gate_inp.weight"))  { kind = "x"; layer = 0; }
        else if (!strcmp(weights->name, "blk.20.ffn_gate_inp.weight")) { kind = "x"; layer = 20; }
        else if (!strcmp(weights->name, "blk.39.ffn_gate_inp.weight")) { kind = "x"; layer = 39; }
        else return;
    } else {
        return;
    }
    if (act->ne[0] != 2048) return;
    const int64_t nrow = ggml_nelements(act) / 2048;
    if (nrow != 1) return; // decode-only
    const int li = (layer == 0) ? 0 : (layer == 20) ? 1 : 2;
    const int ki = (kind[0] == 'x') ? 0 : ((kind[1] == '0') ? 1 : 2);
    static FILE * fb[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    static FILE * fi[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    static uint64_t ev[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    if (fb[li][ki] == NULL) {
        char pb[1024];
        char pi[1024];
        snprintf(pb, sizeof(pb), "%s.L%02d.%s.bin", pre, layer, kind);
        snprintf(pi, sizeof(pi), "%s.L%02d.%s.idx", pre, layer, kind);
        fb[li][ki] = fopen(pb, "ab");
        fi[li][ki] = fopen(pi, "a");
        if (fb[li][ki] == NULL || fi[li][ki] == NULL) return;
        setvbuf(fi[li][ki], NULL, _IOLBF, 0);
    }
    if (fwrite(act->data, sizeof(float), 2048, fb[li][ki]) != 2048) return;
    fflush(fb[li][ki]);
    fprintf(fi[li][ki], "{\"event\":%llu,\"n_rows\":1}\n", (unsigned long long) ev[li][ki]++);
}

static void dynskip_w8n_hook(const struct ggml_compute_params * params,
                             const struct ggml_tensor * tensor) {
    if (tensor->op != GGML_OP_DIV) return;
    if (strcmp(tensor->name, "dynskip_w8n_20") != 0) return;
    const char * mode_s = getenv("GGML_DYNSKIP");
    const int mode = (mode_s == NULL || mode_s[0] == '\0') ? 0 : atoi(mode_s);
    const struct ggml_tensor * w8 = tensor->src[0];
    if (w8 == NULL || w8->type != GGML_TYPE_F32 || w8->ne[0] != 8) {
        fprintf(stderr, "dynskip: w8n_20 layout mismatch (null=%d type=%d ne0=%lld)\n",
            w8 == NULL, w8 ? (int) w8->type : -1, w8 ? (long long) w8->ne[0] : -1);
        abort();
    }
    const int64_t nrow = w8->ne[1];
    if (params->ith == 0 && mode != 0) {
        const char * log = getenv("GGML_DYNSKIP_LOG");
        FILE * fp = (log != NULL && log[0] != '\0') ? fopen(log, "a") : NULL;
        static uint64_t fwd = 0;
        for (int64_t r = 0; r < nrow; ++r) {
            float * row = (float *) ((char *) w8->data + r * w8->nb[1]);
            const int allzero = (row[0] == 0.0f && row[1] == 0.0f && row[2] == 0.0f && row[3] == 0.0f &&
                                 row[4] == 0.0f && row[5] == 0.0f && row[6] == 0.0f && row[7] == 0.0f);
            if (mode == 1 && allzero) {
                // re-visit after our own zeroing (executor double-visit, same
                // memory): decision already applied, log a marker not a top1.
                if (fp != NULL) {
                    fprintf(fp, "{\"fwd\":%llu,\"row\":%lld,\"n_rows\":%lld,\"revisit\":1}\n",
                        (unsigned long long) fwd, (long long) r, (long long) nrow);
                }
                continue;
            }
            const float top1 = row[0];
            const int skip = (mode == 1 && top1 <= 0.02f) ? 1 : 0;
            if (skip) memset(row, 0, 8 * sizeof(float));
            if (fp != NULL) {
                fprintf(fp, "{\"fwd\":%llu,\"row\":%lld,\"n_rows\":%lld,\"top1\":%.6f,\"skip\":%d}\n",
                    (unsigned long long) fwd, (long long) r, (long long) nrow, top1, skip);
            }
        }
        if (fp != NULL) fclose(fp);
        fwd++;
    }
    ggml_barrier(params->threadpool);
}

static void dynskip_post_dump(const struct ggml_compute_params * params,
                              const struct ggml_tensor * tensor) {
    if (params->ith != 0) return;
    // NOTE: called AFTER the op computed, so tensor->data is fresh.
    static const char * pre = NULL;
    static int init = 0;
    if (!init) { pre = getenv("GGML_DYNSKIP_PREFIX"); init = 1; }
    if (pre == NULL || pre[0] == '\0') return;
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] != 2048) return;
    if (ggml_nelements(tensor) / 2048 != 1) return; // decode-only
    const char * nm = tensor->name;
    int layer = -1;
    const char * kind = NULL;
    if (!strncmp(nm, "dynskip_h1_", 11)) {
        layer = atoi(nm + 11); kind = "h1";
        if (layer != 0 && layer != 20 && layer != 39) return;
    } else if (!strncmp(nm, "dynskip_B_", 10)) {
        layer = atoi(nm + 10); kind = "B";
        if (layer != 0 && layer != 20 && layer != 39) return;
    } else if (!strcmp(nm, "dynskip_h0_0")) {
        layer = 0; kind = "h0";
    } else if (!strcmp(nm, "dynskip_lout_19")) {
        layer = 20; kind = "h0";
    } else if (!strcmp(nm, "dynskip_lout_38")) {
        layer = 39; kind = "h0";
    } else {
        return;
    }
    const int li = (layer == 0) ? 0 : (layer == 20) ? 1 : 2;
    const int ki = (!strcmp(kind, "h1")) ? 0 : ((!strcmp(kind, "B")) ? 1 : 2);
    static FILE * fb[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    static FILE * fi[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    static uint64_t ev[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    if (fb[li][ki] == NULL) {
        char pb[1024];
        char pi[1024];
        snprintf(pb, sizeof(pb), "%s.L%02d.%s.bin", pre, layer, kind);
        snprintf(pi, sizeof(pi), "%s.L%02d.%s.idx", pre, layer, kind);
        fb[li][ki] = fopen(pb, "ab");
        fi[li][ki] = fopen(pi, "a");
        if (fb[li][ki] == NULL || fi[li][ki] == NULL) return;
        setvbuf(fi[li][ki], NULL, _IOLBF, 0);
    }
    if (fwrite(tensor->data, sizeof(float), 2048, fb[li][ki]) != 2048) return;
    fflush(fb[li][ki]);
    fprintf(fi[li][ki], "{\"event\":%llu,\"n_rows\":1}\n", (unsigned long long) ev[li][ki]++);
}
'''


def setup_runtime() -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not LLAMA.exists():
        run_checked(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    cpu = LLAMA / "ggml/src/ggml-cpu/ggml-cpu.c"
    replace_once(cpu, "static struct ggml_state g_state = {0};\n", HOOKS + "\nstatic struct ggml_state g_state = {0};\n")
    replace_once(cpu,
        "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
        "        return;\n"
        "    }\n\n"
        "    // extra_buffer op?",
        "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
        "        return;\n"
        "    }\n\n"
        "    dynskip_route_trace(params, tensor);\n"
        "    dynskip_state_dump(params, tensor);\n"
        "    dynskip_w8n_hook(params, tensor);\n\n"
        "    // extra_buffer op?")
    replace_once(cpu,
        '        case GGML_OP_COUNT:\n'
        '            {\n'
        '                GGML_ABORT("fatal error");\n'
        '            }\n'
        '    }\n'
        '}\n'
        '\n'
        "// Android's libc",
        '        case GGML_OP_COUNT:\n'
        '            {\n'
        '                GGML_ABORT("fatal error");\n'
        '            }\n'
        '    }\n'
        '\n'
        '    dynskip_post_dump(params, tensor);\n'
        '}\n'
        '\n'
        "// Android's libc")
    graph = LLAMA / "src/llama-graph.cpp"
    replace_once(graph,
        '        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]\n'
        '        cb(weights, "ffn_moe_weights_norm", il);\n',
        '        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]\n'
        '        cb(weights, "ffn_moe_weights_norm", il);\n'
        '        ggml_format_name(weights, "dynskip_w8n_%d", il);\n')
    qwen = LLAMA / "src/models/qwen35moe.cpp"
    replace_once(qwen,
        '    cb(inpL, "model.input_embed", -1);\n',
        '    cb(inpL, "model.input_embed", -1);\n'
        '    ggml_format_name(inpL, "dynskip_h0_0");\n')
    replace_once(qwen,
        '        cb(cur, "attn_residual", il);\n',
        '        cb(cur, "attn_residual", il);\n'
        '        ggml_format_name(cur, "dynskip_h1_%d", il);\n')
    # NOTE: name B in the ctor AFTER its cb: the ctor re-cbs the returned
    # tensor ("ffn_out") and cb_func renames, clobbering an earlier name.
    replace_once(qwen,
        '        cur = build_layer_ffn(attn_post_norm, il);\n'
        '        cb(cur, "ffn_out", il);\n',
        '        cur = build_layer_ffn(attn_post_norm, il);\n'
        '        cb(cur, "ffn_out", il);\n'
        '        ggml_format_name(cur, "dynskip_B_%d", il);\n')
    replace_once(qwen,
        '        cb(cur, "l_out", il);\n',
        '        cb(cur, "l_out", il);\n'
        '        ggml_format_name(cur, "dynskip_lout_%d", il);\n')
    patch_cpu = run_checked(["git", "diff", "--", "ggml/src/ggml-cpu/ggml-cpu.c"], cwd=LLAMA).stdout
    patch_graph = run_checked(["git", "diff", "--", "src/llama-graph.cpp"], cwd=LLAMA).stdout
    patch_qwen = run_checked(["git", "diff", "--", "src/models/qwen35moe.cpp"], cwd=LLAMA).stdout
    (OUT / "dynskip-runtime.patch").write_text(patch_cpu, encoding="utf-8")
    (OUT / "dynskip-graph.patch").write_text(patch_graph, encoding="utf-8")
    (OUT / "dynskip-qwen.patch").write_text(patch_qwen, encoding="utf-8")
    return {"llama_commit": LLAMA_COMMIT,
            "runtime_patch_sha256": hashlib.sha256(patch_cpu.encode()).hexdigest(),
            "graph_patch_sha256": hashlib.sha256(patch_graph.encode()).hexdigest(),
            "qwen_patch_sha256": hashlib.sha256(patch_qwen.encode()).hexdigest()}


def build() -> None:
    run_checked(["cmake", "-S", str(LLAMA), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
                 "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF",
                 "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF"],
                log=OUT / "cmake-configure.log")
    run_checked(["cmake", "--build", str(BUILD), "--config", "Release", "-j4", "--target", "llama-cli"],
                log=OUT / "cmake-build.log")


def fetch_model() -> dict:
    if not MODEL.exists() or MODEL.stat().st_size != MODEL_SIZE:
        run_checked(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5", "-C", "-",
                     "-o", str(MODEL), MODEL_URL], log=OUT / "model-download.log")
    digest = sha256(MODEL)
    if MODEL.stat().st_size != MODEL_SIZE or digest != MODEL_SHA256:
        raise RuntimeError(f"model identity mismatch: {MODEL.stat().st_size} {digest}")
    return {"file": MODEL_FILE, "size_bytes": MODEL.stat().st_size, "sha256": digest,
            "repo_commit": MODEL_REPO_COMMIT}

import numpy as np


def cli_cmd(prompt: str, seed: int) -> list[str]:
    return [str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(THREADS), "-c", "512",
            "-n", str(N_GEN), "--temp", "0", "--seed", str(seed),
            "--single-turn", "--no-display-prompt", "--no-warmup", "--perf",
            "-lm", "mmap", "-lzm", "off", "--poll", "0", "-p", prompt]


def run_cli(prompt: str, seed: int, env_extra: dict, stdout_p: Path, stderr_p: Path) -> None:
    env = dict(os.environ)
    env.update(env_extra)
    p = subprocess.run(cli_cmd(prompt, seed), text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=env, check=False)
    stdout_p.write_text(p.stdout, encoding="utf-8")
    stderr_p.write_text(p.stderr, encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"llama-cli failed rc={p.returncode}: {p.stderr[-2000:]}")


def dedupe_rows(raw: np.ndarray, what: str) -> np.ndarray:
    n = raw.shape[0]
    if n % 2 == 1:
        return raw
    d = np.abs(raw[0::2] - raw[1::2]).max()
    if d == 0.0:
        return raw[0::2]
    raise RuntimeError(f"{what}: even count {n} but pairs differ (maxdiff {d})")


def verify_control_streams(pid: int, prefix: Path) -> dict:
    counts = {}
    for layer in LAYERS:
        for kind in KINDS:
            binp = Path(str(prefix) + f".L{layer:02d}.{kind}.bin")
            idxp = Path(str(prefix) + f".L{layer:02d}.{kind}.idx")
            if not binp.exists() or not idxp.exists():
                raise RuntimeError(f"p{pid}: missing stream L{layer}.{kind}")
            n_ev = sum(1 for _ in idxp.open(encoding="utf-8"))
            if n_ev == 0:
                raise RuntimeError(f"p{pid}: zero events L{layer}.{kind} (hook missed?)")
            if binp.stat().st_size != n_ev * 2048 * 4:
                raise RuntimeError(f"p{pid}: L{layer}.{kind} size {binp.stat().st_size} != {n_ev}*8192")
            raw = np.fromfile(binp, dtype=np.float32).reshape(-1, 2048)
            counts[(layer, kind)] = dedupe_rows(raw, f"p{pid} L{layer}.{kind}").shape[0]
    vals = set(counts.values())
    if len(vals) != 1:
        raise RuntimeError(f"p{pid}: stream count mismatch {counts}")
    n_dec = vals.pop()
    # routes: filter [8,1] decode events, expect 40/token cycling 0..39
    rpath = OUT / f"prompt_{pid:02d}.routes.jsonl"
    lays = []
    n_evt = 0
    for line in rpath.open(encoding="utf-8"):
        row = json.loads(line)
        n_evt += 1
        if row["shape"][0] == 8 and row["shape"][1] == 1:
            lays.append(int(row["weight"].split("blk.", 1)[1].split(".", 1)[0]))
    if len(lays) % 40 != 0 or len(lays) // 40 != n_dec:
        raise RuntimeError(f"p{pid}: routes {len(lays)}/40 != streams {n_dec} (raw events {n_evt})")
    for t in range(n_dec):
        if lays[t * 40:(t + 1) * 40] != list(range(40)):
            raise RuntimeError(f"p{pid}: route layer order broken at token {t}")
    return {"n_dec": n_dec, "route_events": n_evt}


def generation_of(raw: bytes) -> bytes:
    """Extract generated text between the prompt echo and the perf line.

    llama-cli stdout carries banner/spinner/timings (nondeterministic); only
    the generation span is comparable across runs.
    """
    import re
    m = re.search(rb"\n> .*?\n\n(.*)\n\n\[ Prompt:", raw, re.S)
    if m is None:
        raise RuntimeError("could not locate generation span in CLI stdout")
    return m.group(1)


def route_ids_of(path: Path) -> list:
    return [json.loads(l)["ids"] for l in path.open(encoding="utf-8")]


def parse_skiplog(path: Path) -> dict:
    fwds: dict[int, list] = {}
    n_revisit = 0
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("revisit"):
            n_revisit += 1
            continue
        fwds.setdefault(int(r["fwd"]), []).append(r)
    ids = sorted(fwds)
    deduped = False
    if len(ids) % 2 == 0 and len(ids) > 0:
        ok = True
        for a, b in zip(ids[0::2], ids[1::2]):
            ra, rb = fwds[a], fwds[b]
            if len(ra) != len(rb) or any(x["top1"] != y["top1"] or x["skip"] != y["skip"] for x, y in zip(ra, rb)):
                ok = False
                break
        if ok:
            ids = ids[0::2]
            deduped = True
    dec = [fwds[i][0] for i in ids if fwds[i][0]["n_rows"] == 1]
    top1 = [d["top1"] for d in dec]
    sk = [d["skip"] for d in dec]
    return {"n_decode_fwds": len(dec), "top1": top1, "skip": sk,
            "skip_rate": (sum(sk) / len(sk)) if sk else 0.0,
            "pred_rate": (sum(1 for t in top1 if t <= 0.02) / len(top1)) if top1 else 0.0,
            "revisits": n_revisit, "deduped": deduped, "raw_fwds": len(fwds)}


def check_routes(path: Path) -> dict:
    lays: list[int] = []
    n_evt = 0
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        n_evt += 1
        if row["shape"][0] == 8 and row["shape"][1] == 1:
            lays.append(int(row["weight"].split("blk.", 1)[1].split(".", 1)[0]))
    if len(lays) % 40 != 0:
        raise RuntimeError(f"{path.name}: {len(lays)} decode route rows not a multiple of 40")
    n_tok = len(lays) // 40
    for t in range(n_tok):
        if lays[t * 40:(t + 1) * 40] != list(range(40)):
            raise RuntimeError(f"{path.name}: route layer order broken at token {t}")
    return {"n_tok": n_tok, "events": n_evt}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runtime = setup_runtime()
    build()
    model = fetch_model()
    prompt_results = []
    total_dec = 0
    for pid, (category, prompt) in enumerate(PROMPTS):
        seed = 1000 + pid
        prefix = OUT / f"prompt_{pid:02d}"
        trace = OUT / f"prompt_{pid:02d}.routes.jsonl"
        slog_c = OUT / f"prompt_{pid:02d}.skipctl.jsonl"
        out_c = OUT / f"prompt_{pid:02d}.control.txt"
        err_c = OUT / f"prompt_{pid:02d}.control.stderr.txt"
        for f in list(OUT.glob(f"prompt_{pid:02d}.L*.bin")) + list(OUT.glob(f"prompt_{pid:02d}.L*.idx")):
            f.unlink()
        for f in (trace, slog_c, out_c, err_c):
            f.unlink(missing_ok=True)
        print(f"===== control p{pid:02d} {category} =====", flush=True)
        run_cli(prompt, seed, {"GGML_DYNSKIP": "2", "GGML_DYNSKIP_PREFIX": str(prefix),
                               "GGML_DYNSKIP_ROUTES": str(trace), "GGML_DYNSKIP_LOG": str(slog_c)},
                out_c, err_c)
        if pid == 0:
            # determinism re-run: byte-compare text + routes + one dump
            t_out, t_err = OUT / "det.stdout", OUT / "det.stderr"
            t_tr, t_log = OUT / "det.routes", OUT / "det.skip"
            for f in (t_out, t_err, t_tr, t_log):
                f.unlink(missing_ok=True)
            for f in list(OUT.glob("det.L*.bin")) + list(OUT.glob("det.L*.idx")):
                f.unlink()
            run_cli(prompt, seed, {"GGML_DYNSKIP": "2", "GGML_DYNSKIP_PREFIX": str(OUT / "det"),
                                   "GGML_DYNSKIP_ROUTES": str(t_tr), "GGML_DYNSKIP_LOG": str(t_log)},
                    t_out, t_err)
            try:
                det_txt = generation_of(t_out.read_bytes()) == generation_of(out_c.read_bytes())
                det_rt = route_ids_of(t_tr) == route_ids_of(trace)
                det_dump = all(
                    Path(str(prefix) + sfx).read_bytes() == (OUT / ("det" + sfx)).read_bytes()
                    for sfx in [f".L{L:02d}.{k}.bin" for L in LAYERS for k in KINDS])
            finally:
                for f in (t_out, t_err, t_tr, t_log):
                    f.unlink(missing_ok=True)
                for f in list(OUT.glob("det.L*.bin")) + list(OUT.glob("det.L*.idx")):
                    f.unlink()
            print(f"determinism: text={det_txt} routes={det_rt} dump={det_dump}", flush=True)
            if not (det_txt and det_rt and det_dump):
                raise RuntimeError("determinism pre-check FAILED")
        v = verify_control_streams(pid, prefix)
        sc = parse_skiplog(slog_c)
        if sc["n_decode_fwds"] != v["n_dec"]:
            raise RuntimeError(f"p{pid}: skiplog decode {sc['n_decode_fwds']} != streams {v['n_dec']}")
        if any(sc["skip"]):
            raise RuntimeError(f"p{pid}: log-only run mutated weights!")
        if not (0.02 <= sc["pred_rate"] <= 0.60):
            raise RuntimeError(f"p{pid}: runtime top1<=0.02 rate {sc['pred_rate']:.3f} outside [0.02,0.60]")
        total_dec += v["n_dec"]
        prompt_results.append({"prompt_id": pid, "category": category, "n_dec": v["n_dec"],
                               "route_events": v["route_events"],
                               "control_sha256": sha256(out_c),
                               "skip_pred_rate_ctl": sc["pred_rate"],
                               "skiplog_revisits_ctl": sc["revisits"],
                               "skiplog_deduped_ctl": sc["deduped"]})
    print(f"control done: {total_dec} decode tokens", flush=True)
    if total_dec < 2000:
        raise RuntimeError(f"only {total_dec} decode tokens (<2000)")
    # POLICY loop (armed, no state dumps; routes must equal control ids)
    for rec in prompt_results:
        pid = rec["prompt_id"]
        category, prompt = PROMPTS[pid]
        seed = 1000 + pid
        trace_p = OUT / f"prompt_{pid:02d}.routes_policy.jsonl"
        slog_p = OUT / f"prompt_{pid:02d}.skip.jsonl"
        out_p = OUT / f"prompt_{pid:02d}.policy.txt"
        err_p = OUT / f"prompt_{pid:02d}.policy.stderr.txt"
        for f in (trace_p, slog_p, out_p, err_p):
            f.unlink(missing_ok=True)
        print(f"===== policy p{pid:02d} {category} =====", flush=True)
        run_cli(prompt, seed, {"GGML_DYNSKIP": "1",
                               "GGML_DYNSKIP_ROUTES": str(trace_p), "GGML_DYNSKIP_LOG": str(slog_p)},
                out_p, err_p)
        # policy routes: structural check only here; prefix-equality vs control
        # (pre-divergence) is verified locally with flip-position awareness,
        # since post-flip trajectories legitimately re-route.
        rp = check_routes(trace_p)
        rec["n_tok_policy"] = rp["n_tok"]
        rec["route_events_policy"] = rp["events"]
        sp = parse_skiplog(slog_p)
        rec["skip_rate_policy"] = sp["skip_rate"]
        rec["skip_pred_rate_policy"] = sp["pred_rate"]
        rec["skiplog_revisits_policy"] = sp["revisits"]
        if sp["n_decode_fwds"] != rp["n_tok"]:
            raise RuntimeError(f"p{pid}: policy skiplog {sp['n_decode_fwds']} != routes {rp['n_tok']}")
        rec["policy_sha256"] = sha256(out_p)
        a = generation_of((OUT / f"prompt_{pid:02d}.control.txt").read_bytes())
        b = generation_of(out_p.read_bytes())
        rec["text_identical"] = (a == b)
        rec["first_diff"] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                                 (min(len(a), len(b)) if len(a) != len(b) else -1))
    n_flip = sum(1 for r in prompt_results if not r["text_identical"])
    print(f"policy done: {n_flip}/{len(prompt_results)} prompts diverged", flush=True)
    # package dumps, then delete raws (keep download small)
    tar_p = OUT / "dumps.tar.gz"
    with tarfile.open(tar_p, "w:gz", compresslevel=6) as tf:
        for f in sorted(OUT.glob("prompt_*.L*.bin")) + sorted(OUT.glob("prompt_*.L*.idx")):
            tf.add(f, arcname=f.name)
    with tarfile.open(tar_p, "r:gz") as tf:
        members = tf.getnames()
    expect = 32 * len(LAYERS) * len(KINDS) * 2
    if len(members) != expect:
        raise RuntimeError(f"tarball has {len(members)} members, expected {expect}")
    for f in list(OUT.glob("prompt_*.L*.bin")) + list(OUT.glob("prompt_*.L*.idx")):
        f.unlink()
    result = {
        "schema": "native-sparse-dynskip-ctx/v1", "status": "ok",
        "runtime": runtime, "model": model,
        "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                     "cpu_count": os.cpu_count(), "threads": THREADS},
        "decode": {"tokens_requested_per_prompt": N_GEN, "prompts": len(PROMPTS),
                   "decode_tokens": total_dec, "layers": list(LAYERS), "kinds": list(KINDS),
                   "greedy": True, "policy": "skip L20 routed iff runtime top1 <= 0.02 (exact-zero)"},
        "prompt_results": prompt_results,
        "summary": {"prompts_diverged": n_flip,
                    "mean_skip_pred_ctl": sum(r["skip_pred_rate_ctl"] for r in prompt_results) / len(prompt_results),
                    "mean_skip_rate_policy": sum(r["skip_rate_policy"] for r in prompt_results) / len(prompt_results)},
        "wall_sec": time.time() - started,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "decode_tokens": total_dec, "diverged": n_flip}), flush=True)


if __name__ == "__main__":
    main()
