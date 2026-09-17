"""Phase 5E: collect a diverse exact native route corpus on Kaggle.

Each prompt is decoded independently with the unchanged Qwen3.5 native top-8
router.  The CPU hook records the exact IDs consumed by routed MUL_MAT_ID
nodes; this script only observes those IDs and never changes execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path


SCRATCH = Path("/tmp/native-sparse-route-corpus")
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
CLI = BUILD / "bin" / "llama-cli"
OUT = Path("/kaggle/working/native-sparse-route-corpus-v1-results")
LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
RESEARCH_BASE_COMMIT = "cd2be78"
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
N_GEN = 64

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


def setup_runtime() -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not LLAMA.exists():
        run_checked(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    cpu = LLAMA / "ggml/src/ggml-cpu/ggml-cpu.c"
    qwen = LLAMA / "src/models/qwen35moe.cpp"
    hook = r'''
static void phase5_route_trace(const struct ggml_compute_params * params,
                               const struct ggml_tensor * tensor) {
    if (params->ith != 0 || tensor->op != GGML_OP_MUL_MAT_ID) return;
    const char * path = getenv("GGML_ROUTE_CORPUS_TRACE");
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
'''
    replace_once(cpu, "static struct ggml_state g_state = {0};\n", hook + "\nstatic struct ggml_state g_state = {0};\n")
    replace_once(cpu,
        "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
        "        return;\n"
        "    }\n\n"
        "    // extra_buffer op?",
        "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
        "        return;\n"
        "    }\n\n"
        "    phase5_route_trace(params, tensor);\n\n"
        "    // extra_buffer op?")
    patch = run_checked(["git", "diff", "--", "ggml/src/ggml-cpu/ggml-cpu.c"], cwd=LLAMA).stdout
    (OUT / "route-runtime.patch").write_text(patch, encoding="utf-8")
    return {"llama_commit": LLAMA_COMMIT, "runtime_patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}


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


def normalize_trace(path: Path, prompt_id: int, category: str) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["shape"][0] != 8 or row["shape"][1] != 1:
            continue
        layer = int(row["weight"].split("blk.", 1)[1].split(".", 1)[0])
        row["layer"] = layer
        events.append(row)
    tokens: list[dict] = []
    current: list[list[int]] = []
    current_ns = None
    for row in events:
        if row["layer"] == 0 and current:
            if len(current) != 40:
                raise RuntimeError(f"prompt {prompt_id}: incomplete route token ({len(current)} layers)")
            tokens.append({"prompt_id": prompt_id, "category": category, "token": len(tokens),
                           "mono_ns": current_ns, "layers": current})
            current, current_ns = [], None
        if row["layer"] != len(current):
            raise RuntimeError(f"prompt {prompt_id}: expected layer {len(current)}, got {row['layer']}")
        ids = [int(x) for x in row["ids"]]
        if len(ids) != 8 or len(set(ids)) != 8 or not all(0 <= x < 256 for x in ids):
            raise RuntimeError(f"prompt {prompt_id}: invalid IDs {ids}")
        current.append(ids)
        current_ns = row["mono_ns"] if current_ns is None else current_ns
    if current:
        if len(current) != 40:
            raise RuntimeError(f"prompt {prompt_id}: final incomplete route token ({len(current)} layers)")
        tokens.append({"prompt_id": prompt_id, "category": category, "token": len(tokens),
                       "mono_ns": current_ns, "layers": current})
    return tokens


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runtime = setup_runtime()
    build()
    model = fetch_model()
    all_tokens: list[dict] = []
    prompt_results = []
    for prompt_id, (category, prompt) in enumerate(PROMPTS):
        trace = OUT / f"prompt_{prompt_id:02d}.routes.jsonl"
        stdout = OUT / f"prompt_{prompt_id:02d}.stdout.txt"
        stderr = OUT / f"prompt_{prompt_id:02d}.stderr.txt"
        trace.unlink(missing_ok=True)
        cmd = [str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(THREADS), "-c", "512",
               "-n", str(N_GEN), "--temp", "0.7", "--top-p", "0.9", "--seed", str(1000 + prompt_id),
               "--single-turn", "--no-display-prompt", "--no-warmup", "--perf",
               "-lm", "mmap", "-lzm", "off", "--poll", "0", "-p", prompt]
        env = dict(os.environ, GGML_ROUTE_CORPUS_TRACE=str(trace))
        print(f"===== prompt {prompt_id:02d} {category} =====", flush=True)
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)
        stdout.write_text(p.stdout, encoding="utf-8")
        stderr.write_text(p.stderr, encoding="utf-8")
        tokens = normalize_trace(trace, prompt_id, category) if trace.exists() else []
        for token in tokens:
            token["global_token"] = len(all_tokens)
            all_tokens.append(token)
        prompt_results.append({"prompt_id": prompt_id, "category": category, "prompt": prompt,
                               "returncode": p.returncode, "decode_tokens": len(tokens),
                               "trace_file": trace.name})
        if p.returncode != 0:
            raise RuntimeError(f"prompt {prompt_id} failed: {p.stderr[-2000:]}")
    corpus = OUT / "route_corpus.jsonl"
    corpus.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in all_tokens), encoding="utf-8")
    result = {
        "schema": "native-sparse-route-corpus/v1", "status": "ok",
        "hypothesis": "Cache-policy conclusions should be tested against diverse exact native routes rather than the original 23-token smoke trace.",
        "runtime": runtime, "model": model,
        "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                     "cpu_count": os.cpu_count(), "threads": THREADS,
                     "lscpu": run_checked(["lscpu"]).stdout},
        "decode": {"tokens_requested_per_prompt": N_GEN, "prompts": len(PROMPTS),
                   "corpus_tokens": len(all_tokens), "route_shape": [40, 8], "expert_count": 256},
        "prompt_results": prompt_results,
        "corpus_file": corpus.name, "corpus_sha256": sha256(corpus),
        "wall_sec": time.time() - started,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "prompts": len(PROMPTS), "tokens": len(all_tokens),
                      "corpus_sha256": result["corpus_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
