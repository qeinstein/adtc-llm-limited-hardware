"""Phase 6G: explicit pread-backed bounded expert execution.

The exact Qwen graph and CPU dot kernels remain the control.  The experiment
only changes routed expert ownership: loader metadata registers GGUF offsets,
then MUL_MAT_ID fills fixed cache slots with the selected layer/expert bundle
using pread.  The mapped routed tensor is never dereferenced by the custom arm.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path


WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-bounded-v2")
OUT = WORK / "native-sparse-bounded-v2-results"
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
CLI = BUILD / "bin" / "llama-cli"
MODEL = SCRATCH / "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_SHA256 = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
N_THREADS = max(1, min(4, os.cpu_count() or 1))
N_GEN = 64
N_REPS = 3
PROMPT = "Give one concise reason oral rehydration solution helps a child with watery diarrhoea."
CACHE_BYTES = 2_000_000_000

OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
        log: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    print(p.stdout[-5000:], flush=True)
    if log:
        log.write_text(p.stdout, encoding="utf-8")
    if check and p.returncode:
        raise RuntimeError(f"command exited {p.returncode}: {cmd}")
    return p


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


BOUNDED_C = r'''

// Phase 6G explicit byte-bounded expert store.  The loader registers the
// original GGUF offsets, while this C backend owns only fixed anonymous slots.
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

struct phase6_record {
    const struct ggml_tensor * tensor;
    int fd;
    size_t base_offset;
    size_t slice_bytes;
    int layer;
    int kind; // gate=0, up=1, down=2
    int valid;
};

struct phase6_slot {
    int layer;
    int expert;
    uint64_t age;
    int valid;
};

static struct phase6_record phase6_records[40][3];
static int phase6_bundle_slots[40][256];
static struct phase6_slot * phase6_slots = NULL;
static unsigned char * phase6_storage = NULL;
static size_t phase6_slot_bytes = 0;
static size_t phase6_slot_count = 0;
static uint64_t phase6_age = 0;
static uint64_t phase6_requests = 0;
static uint64_t phase6_hits = 0;
static uint64_t phase6_misses = 0;
static uint64_t phase6_evictions = 0;
static uint64_t phase6_read_calls = 0;
static uint64_t phase6_read_bytes = 0;
static uint64_t phase6_read_ns = 0;
static int phase6_initialized = 0;
static int phase6_report_registered = 0;

static int phase6_enabled(void) {
    const char * value = getenv("GGML_PHASE6_BOUNDED_CACHE");
    return value != NULL && atoi(value) != 0;
}

static int phase6_kind(const char * name) {
    if (name == NULL || strstr(name, "ffn_") == NULL || strstr(name, "_exps") == NULL) return -1;
    if (strstr(name, "ffn_gate_exps")) return 0;
    if (strstr(name, "ffn_up_exps")) return 1;
    if (strstr(name, "ffn_down_exps")) return 2;
    return -1;
}

static int phase6_layer(const char * name) {
    int layer = -1;
    if (name != NULL) sscanf(name, "blk.%d.", &layer);
    return layer;
}

void ggml_cpu_phase6_register_lazy_tensor(const struct ggml_tensor * tensor,
        int fd, size_t base_offset, size_t nbytes, size_t expert_stride) {
    if (!phase6_enabled() || tensor == NULL || tensor->name == NULL) return;
    const int layer = phase6_layer(tensor->name);
    const int kind = phase6_kind(tensor->name);
    if (layer < 0 || layer >= 40 || kind < 0 || kind >= 3 || expert_stride == 0 ||
            expert_stride * 256 > nbytes) return;
    // llama_model_loader owns the original descriptor and may destroy it
    // after model construction.  The explicit cache owns this duplicate so
    // pread remains valid during generation.
    const int owned_fd = dup(fd);
    if (owned_fd < 0) {
        fprintf(stderr, "PHASE6_BOUNDED_CACHE_ERROR dup failed errno=%d\n", errno);
        abort();
    }
    phase6_records[layer][kind] = (struct phase6_record) {
        tensor, owned_fd, base_offset, expert_stride, layer, kind, 1
    };
}

static void phase6_fail(const char * message) {
    fprintf(stderr, "PHASE6_BOUNDED_CACHE_ERROR %s\n", message);
    abort();
}

static uint64_t phase6_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

static void phase6_report(void) {
    if (!phase6_enabled()) return;
    fprintf(stderr,
        "PHASE6_BOUNDED_CACHE slot_bytes=%zu slots=%zu reserved_bytes=%zu "
        "requests=%llu hits=%llu misses=%llu evictions=%llu read_calls=%llu "
        "read_bytes=%llu read_ns=%llu\n",
        phase6_slot_bytes, phase6_slot_count, phase6_slot_bytes * phase6_slot_count,
        (unsigned long long) phase6_requests, (unsigned long long) phase6_hits,
        (unsigned long long) phase6_misses, (unsigned long long) phase6_evictions,
        (unsigned long long) phase6_read_calls, (unsigned long long) phase6_read_bytes,
        (unsigned long long) phase6_read_ns);
#if defined(__linux__)
    for (int layer = 0; layer < 40; ++layer) {
        for (int kind = 0; kind < 3; ++kind) {
            if (phase6_records[layer][kind].valid && phase6_records[layer][kind].fd >= 0) {
                close(phase6_records[layer][kind].fd);
                phase6_records[layer][kind].fd = -1;
            }
        }
    }
#endif
}

static void phase6_init(void) {
    if (phase6_initialized) return;
    for (int layer = 0; layer < 40; ++layer) {
        for (int expert = 0; expert < 256; ++expert) phase6_bundle_slots[layer][expert] = -1;
    }
    for (int layer = 0; layer < 40; ++layer) {
        size_t total = 0;
        for (int kind = 0; kind < 3; ++kind) {
            if (!phase6_records[layer][kind].valid) phase6_fail("incomplete routed tensor registration");
            total += phase6_records[layer][kind].slice_bytes;
        }
        if (layer == 0) phase6_slot_bytes = total;
        if (total != phase6_slot_bytes) phase6_fail("nonuniform bundle size is not supported by v1");
    }
    const char * capacity_text = getenv("GGML_PHASE6_CACHE_BYTES");
    const unsigned long long capacity = capacity_text ? strtoull(capacity_text, NULL, 10) : 0;
    phase6_slot_count = phase6_slot_bytes ? (size_t) (capacity / phase6_slot_bytes) : 0;
    if (phase6_slot_count == 0) phase6_fail("cache capacity does not hold one routed bundle");
    phase6_slots = (struct phase6_slot *) calloc(phase6_slot_count, sizeof(*phase6_slots));
#if defined(__linux__)
    phase6_storage = (unsigned char *) mmap(NULL, phase6_slot_bytes * phase6_slot_count,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (phase6_storage == MAP_FAILED) phase6_storage = NULL;
#endif
    if (phase6_storage == NULL || phase6_slots == NULL) phase6_fail("fixed cache allocation failed");
    if (!phase6_report_registered) {
        phase6_report_registered = 1;
        atexit(phase6_report);
    }
    phase6_initialized = 1;
}

static int phase6_find_slot(int layer, int expert) {
    if (layer < 0 || layer >= 40 || expert < 0 || expert >= 256) return -1;
    return phase6_bundle_slots[layer][expert];
}

static void phase6_read_exact(int fd, void * dst, size_t bytes, size_t offset) {
    size_t done = 0;
    const uint64_t start = phase6_now_ns();
    while (done < bytes) {
        const ssize_t got = pread(fd, (char *) dst + done, bytes - done, offset + done);
        if (got < 0) {
            if (errno == EINTR) continue;
            phase6_fail("pread failed");
        }
        if (got == 0) phase6_fail("short pread");
        done += (size_t) got;
    }
    const uint64_t end = phase6_now_ns();
    phase6_read_ns += end - start;
    phase6_read_calls += 3;
    phase6_read_bytes += bytes;
}

static int phase6_load(int layer, int expert) {
    phase6_init();
    phase6_requests++;
    int slot = phase6_find_slot(layer, expert);
    if (slot >= 0) {
        phase6_hits++;
        phase6_slots[slot].age = ++phase6_age;
        return slot;
    }
    phase6_misses++;
    slot = -1;
    for (size_t i = 0; i < phase6_slot_count; ++i) {
        if (!phase6_slots[i].valid) { slot = (int) i; break; }
    }
    if (slot < 0) {
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < phase6_slot_count; ++i) {
            if (phase6_slots[i].age < oldest) { oldest = phase6_slots[i].age; slot = (int) i; }
        }
        if (phase6_slots[slot].valid) {
            phase6_bundle_slots[phase6_slots[slot].layer][phase6_slots[slot].expert] = -1;
            phase6_evictions++;
        }
#if defined(__linux__)
        madvise(phase6_storage + (size_t) slot * phase6_slot_bytes,
                phase6_slot_bytes, MADV_DONTNEED);
#endif
    }
    unsigned char * dst = phase6_storage + (size_t) slot * phase6_slot_bytes;
    size_t in_slot = 0;
    for (int kind = 0; kind < 3; ++kind) {
        const struct phase6_record * record = &phase6_records[layer][kind];
        phase6_read_exact(record->fd, dst + in_slot, record->slice_bytes,
                          record->base_offset + (size_t) expert * record->slice_bytes);
        in_slot += record->slice_bytes;
    }
    phase6_slots[slot] = (struct phase6_slot) { layer, expert, ++phase6_age, 1 };
    phase6_bundle_slots[layer][expert] = slot;
    return slot;
}

static void phase6_prepare(const struct ggml_tensor * tensor, const struct ggml_tensor * ids) {
    if (!phase6_enabled()) return;
    const int layer = phase6_layer(tensor->name);
    if (layer < 0 || layer >= 40 || ids == NULL || ids->type != GGML_TYPE_I32) {
        phase6_fail("invalid routed node metadata");
    }
    for (int64_t row = 0; row < ids->ne[1]; ++row) {
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int expert = *(const int32_t *) ((const char *) ids->data + row * ids->nb[1] + i * ids->nb[0]);
            if (expert < 0 || expert >= 256) phase6_fail("invalid native expert ID");
            phase6_load(layer, expert);
        }
    }
}

static const char * phase6_tensor_ptr(const struct ggml_tensor * tensor, int expert) {
    const int layer = phase6_layer(tensor->name);
    const int kind = phase6_kind(tensor->name);
    if (layer < 0 || kind < 0 || !phase6_records[layer][kind].valid) return NULL;
    const int slot = phase6_find_slot(layer, expert);
    if (slot < 0) phase6_fail("selected expert was not prepared");
    size_t in_slot = 0;
    for (int k = 0; k < kind; ++k) in_slot += phase6_records[layer][k].slice_bytes;
    return (const char *) phase6_storage + (size_t) slot * phase6_slot_bytes + in_slot;
}
'''


ROUTE_C = r'''

static void ggml_phase6_route_trace(const struct ggml_compute_params * params,
                                    const struct ggml_tensor * tensor) {
    if (params->ith != 0 || tensor->op != GGML_OP_MUL_MAT_ID) return;
    const char * path = getenv("GGML_PHASE6_ROUTE_TRACE");
    if (path == NULL || path[0] == '\0') return;
    const struct ggml_tensor * weights = tensor->src[0];
    const struct ggml_tensor * ids = tensor->src[2];
    if (weights == NULL || ids == NULL || ids->type != GGML_TYPE_I32 ||
            strstr(weights->name, "ffn_") == NULL || strstr(weights->name, "_exps") == NULL) return;
    static FILE * fp = NULL;
    static uint64_t event_id = 0;
    if (fp == NULL) {
        fp = fopen(path, "a");
        if (fp == NULL) return;
        setvbuf(fp, NULL, _IOLBF, 0);
    }
    const int64_t n = ggml_nelements(ids);
    const int32_t * values = (const int32_t *) ids->data;
    fprintf(fp, "{\"event\":%" PRIu64 ",\"weight\":\"%s\",\"shape\":[%" PRId64 ",%" PRId64 "],\"ids\":[",
        event_id++, weights->name, ids->ne[0], ids->ne[1]);
    for (int64_t i = 0; i < n; ++i) fprintf(fp, "%s%d", i ? "," : "", values[i]);
    fprintf(fp, "]}\n");
}
'''


def patch_runtime() -> dict:
    qwen = LLAMA / "src" / "models" / "qwen35moe.cpp"
    replace_once(qwen,
        '        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);\n'
        '        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);',
        '        const int expert_flags = flags | TENSOR_READ_LAZY;\n'
        '        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, expert_flags);\n'
        '        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, expert_flags);')

    loader = LLAMA / "src" / "llama-model-loader.cpp"
    replace_once(loader, '#include <cstring>\n', '#include <cstring>\n#include <cstdlib>\n')
    replace_once(loader, '#include <regex>\n', '#include <regex>\n\nextern "C" void ggml_cpu_phase6_register_lazy_tensor(const struct ggml_tensor *, int, size_t, size_t, size_t);\n')
    replace_once(loader,
        '        size_t n_size = ggml_nbytes(cur);\n\n        const bool from_mapping = use_mmap || lazy.has(cur);',
        '        size_t n_size = ggml_nbytes(cur);\n\n'
        '        if (lazy.has(cur) && std::getenv("GGML_PHASE6_BOUNDED_CACHE") != nullptr) {\n'
        '            const auto & file = files.at(weight->idx);\n'
        '            ggml_cpu_phase6_register_lazy_tensor(cur, file->file_id(), weight->offs, n_size, cur->nb[2]);\n'
        '        }\n\n'
        '        const bool from_mapping = use_mmap || lazy.has(cur);')

    cpu = LLAMA / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
    replace_once(cpu, 'static struct ggml_state g_state = {0};\n', BOUNDED_C + ROUTE_C + '\nstatic struct ggml_state g_state = {0};\n')
    replace_once(cpu,
        '    const int ith = params->ith;\n'
        '    const int nth = params->nth;\n\n'
        '    const enum ggml_type type = src0->type;\n\n'
        '    const bool src1_cont = ggml_is_contiguous(src1);\n',
        '    const int ith = params->ith;\n'
        '    const int nth = params->nth;\n\n'
        '    const enum ggml_type type = src0->type;\n\n'
        '    const bool src1_cont = ggml_is_contiguous(src1);\n'
        '    const bool phase6 = phase6_enabled();\n')
    replace_once(cpu,
        '    const bool iqp = ggml_cpu_iqp_supports_mul_mat_id(dst) && !params->use_ref;\n',
        '    const bool iqp = !phase6 && ggml_cpu_iqp_supports_mul_mat_id(dst) && !params->use_ref;\n')
    replace_once(cpu,
        '    // reset current_chunk\n'
        '    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {\n'
        '        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);\n'
        '        *current_chunk_ctr = nth;\n'
        '    }\n\n'
        '    ggml_barrier(params->threadpool);\n\n'
        '    for (int cur_a = 0; cur_a < n_as; ++cur_a) {',
        '    // reset current_chunk\n'
        '    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {\n'
        '        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);\n'
        '        *current_chunk_ctr = nth;\n'
        '    }\n\n'
        '    ggml_barrier(params->threadpool);\n\n'
        '    if (phase6 && ith == 0) phase6_prepare(src0, ids);\n'
        '    ggml_barrier(params->threadpool);\n\n'
        '    for (int cur_a = 0; cur_a < n_as; ++cur_a) {')
    replace_once(cpu,
        '        const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n',
        '        const char * src0_cur = phase6 ? phase6_tensor_ptr(src0, cur_a)\n'
        '            : (const char *) src0->data + cur_a * nb02;\n')
    replace_once(cpu,
        '    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n        return;\n    }\n\n    // extra_buffer op?',
        '    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n        return;\n    }\n\n'
        '    ggml_phase6_route_trace(params, tensor);\n\n    // extra_buffer op?')

    diff = run(["git", "diff", "--", str(qwen.relative_to(LLAMA)),
                str(loader.relative_to(LLAMA)), str(cpu.relative_to(LLAMA))], cwd=LLAMA).stdout
    (OUT / "bounded-runtime.patch").write_text(diff, encoding="utf-8")
    return {"llama_commit": LLAMA_COMMIT,
            "patch_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def clone_and_patch() -> dict:
    if not LLAMA.exists():
        run(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", LLAMA])
    run(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    return patch_runtime()


def build() -> None:
    run(["cmake", "-S", LLAMA, "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release",
         "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF",
         "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF"],
        log=OUT / "cmake-configure.log")
    run(["cmake", "--build", BUILD, "--config", "Release", f"-j{N_THREADS}",
         "--target", "llama-cli"], log=OUT / "cmake-build.log")
    run([CLI, "--version"], log=OUT / "llama-version.log")


def fetch_model() -> dict:
    if not MODEL.exists() or MODEL.stat().st_size != MODEL_SIZE:
        run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
             "-C", "-", "-o", MODEL, MODEL_URL], log=OUT / "model-download.log")
    size = MODEL.stat().st_size
    digest = sha256(MODEL)
    if size != MODEL_SIZE or digest != MODEL_SHA256:
        raise RuntimeError(f"model identity mismatch: size={size} sha256={digest}")
    return {"repo_commit": MODEL_REPO_COMMIT, "file": MODEL_FILE,
            "size_bytes": size, "sha256": digest, "url": MODEL_URL}


def drop_file_cache(path: Path) -> dict:
    os.sync()
    if not hasattr(os, "posix_fadvise"):
        return {"available": False}
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return {"available": True, "called": True}
    except OSError as exc:
        return {"available": True, "called": False, "error": repr(exc)}
    finally:
        os.close(fd)


def proc_sample(pid: int) -> dict:
    row = {"mono_ns": time.monotonic_ns(), "rss_kib": 0, "rss_anon_kib": 0,
           "rss_file_kib": 0, "read_bytes": 0, "rchar": 0, "valid": False}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"): row["rss_kib"] = int(line.split()[1])
            elif line.startswith("RssAnon:"): row["rss_anon_kib"] = int(line.split()[1])
            elif line.startswith("RssFile:"): row["rss_file_kib"] = int(line.split()[1])
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("read_bytes", "rchar"): row[key] = int(value.strip())
        row["valid"] = True
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return row


PERF_RE = re.compile(r"eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) runs\s*\(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\s*\)")
SUMMARY_RE = re.compile(r"\[\s*Prompt:\s*([0-9.]+) t/s\s*\|\s*Generation:\s*([0-9.]+) t/s\s*\]")


def parse_perf(text: str) -> dict:
    detailed = PERF_RE.findall(text)
    if detailed:
        elapsed, runs, ms, tps = detailed[-1]
        return {"ms_per_token": float(ms), "tokens_per_second": float(tps),
                "eval_ms": float(elapsed), "eval_runs": int(runs)}
    summary = SUMMARY_RE.findall(text)
    if summary:
        prompt, generation = summary[-1]
        return {"prompt_tokens_per_second": float(prompt),
                "tokens_per_second": float(generation),
                "ms_per_token": 1000.0 / float(generation)}
    raise RuntimeError("no parseable llama performance line")


def response_payload(stdout: str) -> str:
    marker = stdout.find("[Start thinking]")
    if marker < 0:
        marker = stdout.rfind(PROMPT) + len(PROMPT)
    tail = stdout[marker:]
    perf = SUMMARY_RE.search(tail)
    if perf: tail = tail[:perf.start()]
    if tail.rstrip().endswith("Exiting..."): tail = tail.rstrip()[:-len("Exiting...")]
    return tail.strip()


def run_arm(name: str, bounded: bool, rep: int) -> dict:
    prefix = f"{name}_rep{rep}"
    trace = OUT / f"{prefix}.routes.jsonl"
    stdout_path = OUT / f"{prefix}.stdout.txt"
    stderr_path = OUT / f"{prefix}.stderr.txt"
    samples_path = OUT / f"{prefix}.process.jsonl"
    for path in (trace, stdout_path, stderr_path, samples_path): path.unlink(missing_ok=True)
    cmd = [str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(N_THREADS),
           "-c", "512", "-n", str(N_GEN), "--temp", "0", "--seed", "1234",
           "--single-turn", "--no-display-prompt", "--no-warmup", "--perf",
           "-lm", "mmap", "-lzm", "on" if bounded else "off", "-p", PROMPT]
    env = dict(os.environ, GGML_PHASE6_ROUTE_TRACE=str(trace))
    if bounded:
        env.update(GGML_PHASE6_BOUNDED_CACHE="1", GGML_PHASE6_CACHE_BYTES=str(CACHE_BYTES))
    else:
        env.pop("GGML_PHASE6_BOUNDED_CACHE", None)
        env.pop("GGML_PHASE6_CACHE_BYTES", None)
    print(f"\n===== {prefix} bounded={bounded} =====", flush=True)
    cache_drop = drop_file_cache(MODEL)
    start = time.monotonic_ns()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    samples: list[dict] = []
    stop = threading.Event()

    def sample_loop() -> None:
        while not stop.is_set():
            samples.append(proc_sample(proc.pid))
            stop.wait(0.02)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    out, err = proc.communicate()
    stop.set(); sampler.join(timeout=2)
    samples.append(proc_sample(proc.pid))
    elapsed = (time.monotonic_ns() - start) / 1e9
    stdout_path.write_text(out, encoding="utf-8")
    stderr_path.write_text(err, encoding="utf-8")
    samples_path.write_text("".join(json.dumps(x) + "\n" for x in samples), encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{prefix} exited {proc.returncode}: {err[-3000:]}")
    valid = [x for x in samples if x.get("valid")]
    stats_match = re.findall(r"PHASE6_BOUNDED_CACHE .*", err)
    payload = response_payload(out)
    return {"name": name, "rep": rep, "bounded": bounded, "command": cmd,
            "elapsed_sec": elapsed, "cache_drop": cache_drop,
            "peak_rss_mib": max((x["rss_kib"] for x in valid), default=0) / 1024,
            "peak_rss_anon_mib": max((x["rss_anon_kib"] for x in valid), default=0) / 1024,
            "peak_rss_file_mib": max((x["rss_file_kib"] for x in valid), default=0) / 1024,
            "read_bytes_max": max((x["read_bytes"] for x in valid), default=0),
            "rchar_max": max((x["rchar"] for x in valid), default=0),
            "decode_perf": parse_perf(out + "\n" + err),
            "stdout_sha256": hashlib.sha256(out.encode()).hexdigest(),
            "response_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "response": payload, "bounded_cache_stats": stats_match[-1] if stats_match else None,
            "sample_count": len(samples), "trace": str(trace)}


def main() -> None:
    started = time.time()
    runtime = clone_and_patch()
    build()
    model = fetch_model()
    hardware = {"platform": platform.platform(), "python": platform.python_version(),
                "cpu_count": os.cpu_count(), "threads_used": N_THREADS,
                "cpuinfo": Path("/proc/cpuinfo").read_text()[:12000],
                "meminfo": Path("/proc/meminfo").read_text(),
                "filesystem": run(["df", "-h", "/kaggle/working"]).stdout}
    (OUT / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
    arms = []
    for rep in range(1, N_REPS + 1):
        arms.append(run_arm("control", False, rep))
        arms.append(run_arm("bounded", True, rep))
    controls = [x for x in arms if not x["bounded"]]
    bounded = [x for x in arms if x["bounded"]]
    exactness = {"response_hashes": sorted({x["response_sha256"] for x in arms}),
                 "all_response_equal": len({x["response_sha256"] for x in arms}) == 1,
                 "native_k": 8, "layers": 40, "no_weight_or_route_change": True}
    if not exactness["all_response_equal"]:
        raise AssertionError(f"bounded output differs: {exactness}")
    result = {"schema_version": 1, "status": "complete", "hypothesis":
              "Explicit pread into a fixed global-LRU expert cache can replace routed mmap residency while preserving exact output.",
              "runtime": runtime, "model": model,
              "hardware": {k: v for k, v in hardware.items() if k not in ("cpuinfo", "meminfo")},
              "prompt": PROMPT, "n_gen": N_GEN, "repetitions": N_REPS,
              "cache_capacity_bytes": CACHE_BYTES, "arms": arms, "exactness": exactness,
              "quality": {"result": "deterministic smoke equivalence only"},
              "wall_sec": time.time() - started}
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(__file__, OUT / Path(__file__).name)
    print(json.dumps({"exactness": exactness, "control": controls,
                      "bounded": bounded, "wall_sec": result["wall_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
