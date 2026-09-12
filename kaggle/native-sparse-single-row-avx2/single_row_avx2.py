"""Phase 3: exact single-row IQ2_XXS selected-expert AVX2 A/B.

The optimized arm uses the runtime's existing exact IQ2_XXS panel decoder and
8-output-row AVX2 GEMV, but calls it directly for each one-row selected expert.
It does not change routing, K, weights, quantization, or the model graph.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import threading
import time
from pathlib import Path


SCRATCH = Path("/tmp/native-sparse-single-row-avx2")
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
BENCH = BUILD / "bin" / "llama-bench"
CLI = BUILD / "bin" / "llama-cli"
MODEL = SCRATCH / "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
OUT = Path("/kaggle/working/native-sparse-single-row-avx2-results")

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_SHA256 = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
RESEARCH_BASE_COMMIT = "6f363b3"
PROMPT = "Give one concise reason oral rehydration solution helps a child with watery diarrhoea."
THREADS = 4
N_GEN = 64
N_REPEATS = 3
LOGICAL_SELECTED_EXPERT_BYTES_PER_TOKEN = 280_494_080

SUMMARY_PERF_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+) t/s\s*\|\s*Generation:\s*([0-9.]+) t/s\s*\]"
)


def command_text(cmd: list[str]) -> str:
    return " ".join(str(x) for x in cmd)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(cmd: list[str], *, cwd: Path | None = None, log: Path | None = None) -> subprocess.CompletedProcess:
    print("+", command_text(cmd), flush=True)
    process = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if log is not None:
        log.write_text(process.stdout + "\n--- STDERR ---\n" + process.stderr, encoding="utf-8")
    if process.returncode:
        print(process.stdout[-2000:], flush=True)
        print(process.stderr[-2000:], flush=True)
        raise RuntimeError(f"command exited {process.returncode}: {command_text(cmd)}")
    return process


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_runtime() -> dict:
    cpu = LLAMA / "ggml/src/ggml-cpu/ggml-cpu.c"
    iqp = LLAMA / "ggml/src/ggml-cpu/iqp.cpp"
    iqp_h = LLAMA / "ggml/src/ggml-cpu/iqp.h"
    qwen = LLAMA / "src/models/qwen35moe.cpp"

    # Keep the exact Phase 0 loading intervention: only routed expert tensors
    # are lazy, while router/shared/trunk tensors remain the resident floor.
    replace_once(qwen,
        "        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, \"weight\", il), { n_ff_exp, n_embd, n_expert }, flags);\n"
        "        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);",
        "        // Phase 3: route-selected expert payload is lazy; all other model\n"
        "        // tensors retain the normal loading policy.  Native math/routes\n"
        "        // are unchanged.\n"
        "        const int expert_flags = flags | TENSOR_READ_LAZY;\n"
        "        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, \"weight\", il), { n_ff_exp, n_embd, n_expert }, expert_flags);\n"
        "        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, expert_flags);")

    # Expose the existing exact panel decoder's single-activation GEMV as an
    # opt-in MUL_MAT_ID path.  The generic path remains the control arm.
    replace_once(iqp_h,
        "bool ggml_cpu_iqp_supports_mul_mat_id(const struct ggml_tensor * dst);\n",
        "bool ggml_cpu_iqp_supports_mul_mat_id(const struct ggml_tensor * dst);\n"
        "\n"
        "// Same capability test without the upstream per-expert batch threshold.\n"
        "bool ggml_cpu_iqp_supports_mul_mat_id_single(const struct ggml_tensor * dst);\n")
    replace_once(iqp_h,
        "void ggml_compute_forward_mul_mat_id_iqp(const struct ggml_compute_params * params,\n",
        "// Measurement-only direct raw IQ2_XXS selected-expert rows; no panel scratch.\n"
        "void ggml_compute_forward_mul_mat_id_raw_iq2(const struct ggml_compute_params * params,\n"
        "                                             struct ggml_tensor *               dst,\n"
        "                                             int64_t                            cur_a,\n"
        "                                             const int32_t *                    expert_rows);\n\n"
        "void ggml_compute_forward_mul_mat_id_iqp(const struct ggml_compute_params * params,\n")
    replace_once(iqp,
        "#include \"iqp.h\"\n\n#define UNUSED GGML_UNUSED\n",
        "#include \"iqp.h\"\n#include \"quants.h\"\n\n#define UNUSED GGML_UNUSED\n")
    replace_once(iqp,
        "bool ggml_cpu_iqp_supports_mul_mat_id(const struct ggml_tensor * dst) {\n"
        "    const struct ggml_tensor * ids = dst->src[2];\n\n"
        "    if (!iqp_supported_common(dst)) {\n"
        "        return false;\n"
        "    }\n\n"
        "    // skip the node entirely (work buffer included) if no expert can reach the per expert threshold\n"
        "    if (!ggml_cpu_iqp_mul_mat_id_min_batch(ids->ne[0] * ids->ne[1])) {\n"
        "        return false;\n"
        "    }\n\n"
        "    return true;\n"
        "}\n",
        "bool ggml_cpu_iqp_supports_mul_mat_id(const struct ggml_tensor * dst) {\n"
        "    const struct ggml_tensor * ids = dst->src[2];\n\n"
        "    if (!iqp_supported_common(dst)) {\n"
        "        return false;\n"
        "    }\n\n"
        "    // skip the node entirely (work buffer included) if no expert can reach the per expert threshold\n"
        "    if (!ggml_cpu_iqp_mul_mat_id_min_batch(ids->ne[0] * ids->ne[1])) {\n"
        "        return false;\n"
        "    }\n\n"
        "    return true;\n"
        "}\n\n"
        "bool ggml_cpu_iqp_supports_mul_mat_id_single(const struct ggml_tensor * dst) {\n"
        "    return iqp_supported_common(dst);\n"
        "}\n")

    # Add the dedicated cne1 == 1 branch before the panel's four-row loop.
    replace_once(iqp,
        "        // the dst rows are scattered, so the gemm writes into tmp and it is copied out row by row\n"
        "        float tmp[4 * IQP_NB_ROWS];\n\n"
        "        for (int64_t k = 0; k < cne1; k += 4) {\n",
        "        // Decode the selected expert's eight output rows once, then use the\n"
        "        // one-activation AVX2 GEMV.  This avoids the panel path's four-lane\n"
        "        // tail duplication for the normal single-token MoE case.\n"
        "        if (cne1 == 1) {\n"
        "            const void * row = (const char *) params->wdata +\n"
        "                ((expert_rows[0] % ne11) + expert_rows[1] * ne11) * nbw1;\n"
        "            float * dst_col = (float *) ((char *) dst->data +\n"
        "                expert_rows[0] * nb1 + expert_rows[1] * nb2);\n"
        "            iqp_gemv_8x8_q8_K(ne00, dst_col + r, ne01, panel, row, 1, IQP_NB_ROWS);\n"
        "            continue;\n"
        "        }\n\n"
        "        // the dst rows are scattered, so the gemm writes into tmp and it is copied out row by row\n"
        "        float tmp[4 * IQP_NB_ROWS];\n\n"
        "        for (int64_t k = 0; k < cne1; k += 4) {\n")

    # Permit the optimized arm to allocate the IQP scratch for a node whose
    # per-expert counts are one, while keeping it process-local and opt-in.
    old = "    const bool iqp = ggml_cpu_iqp_supports_mul_mat_id(dst) && !params->use_ref;\n"
    new = ("    const bool single_row_iqp = getenv(\"GGML_SINGLE_ROW_IQP\") != NULL &&\n"
           "        atoi(getenv(\"GGML_SINGLE_ROW_IQP\")) != 0;\n"
           "    const bool iqp = !params->use_ref &&\n"
           "        (ggml_cpu_iqp_supports_mul_mat_id(dst) ||\n"
           "         (single_row_iqp && ggml_cpu_iqp_supports_mul_mat_id_single(dst)));\n")
    replace_once(cpu, old, new)
    old = "        if (iqp && ggml_cpu_iqp_mul_mat_id_min_batch(cne1)) {\n"
    new = ("        if (iqp && (ggml_cpu_iqp_mul_mat_id_min_batch(cne1) ||\n"
           "                   (single_row_iqp && cne1 == 1))) {\n")
    replace_once(cpu, old, new)
    replace_once(cpu,
        "        if (iqp && (ggml_cpu_iqp_mul_mat_id_min_batch(cne1) ||\n"
        "                   (single_row_iqp && cne1 == 1))) {\n"
        "            ggml_compute_forward_mul_mat_id_iqp(params, dst, cur_a, cne1, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0),\n"
        "                                                iqp_panels);\n\n"
        "            continue;\n"
        "        }\n",
        "        const char * raw_iq2_env = getenv(\"GGML_RAW_SINGLE_ROW_IQ2\");\n"
        "        const bool raw_iq2 = raw_iq2_env != NULL && atoi(raw_iq2_env) != 0;\n"
        "        if (raw_iq2 && cne1 == 1 && src0->type == GGML_TYPE_IQ2_XXS) {\n"
        "            ggml_compute_forward_mul_mat_id_raw_iq2(params, dst, cur_a, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0));\n"
        "            continue;\n"
        "        }\n\n"
        "        if (iqp && (ggml_cpu_iqp_mul_mat_id_min_batch(cne1) ||\n"
        "                   (single_row_iqp && cne1 == 1))) {\n"
        "            ggml_compute_forward_mul_mat_id_iqp(params, dst, cur_a, cne1, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0),\n"
        "                                                iqp_panels);\n\n"
        "            continue;\n"
        "        }\n")

    # Reuse the observation-only route marker to split process RSS into the
    # pre-routed floor and the pages touched by selected experts.
    route_hook = r'''
static void ggml_phase3_route_trace(const struct ggml_compute_params * params,
                                    const struct ggml_tensor * tensor) {
    if (params->ith != 0 || tensor->op != GGML_OP_MUL_MAT_ID) return;
    const char * path = getenv("GGML_PHASE0_ROUTE_TRACE");
    if (path == NULL || path[0] == '\0') return;
    const struct ggml_tensor * weights = tensor->src[0];
    const struct ggml_tensor * ids = tensor->src[2];
    if (weights == NULL || ids == NULL || ids->type != GGML_TYPE_I32 ||
            strstr(weights->name, "ffn_") == NULL ||
            strstr(weights->name, "_exps") == NULL) return;
    static FILE * fp = NULL;
    static uint64_t event_id = 0;
    if (fp == NULL) {
        fp = fopen(path, "a");
        if (fp == NULL) return;
        setvbuf(fp, NULL, _IOLBF, 0);
    }
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    fprintf(fp, "{\"event\":%" PRIu64 ",\"mono_ns\":%" PRIu64
        ",\"weight\":\"%s\",\"shape\":[%" PRId64 ",%" PRId64 "]}\n",
        event_id++, (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec,
        weights->name, ids->ne[0], ids->ne[1]);
}
'''
    replace_once(cpu, "static struct ggml_state g_state = {0};\n", route_hook + "\nstatic struct ggml_state g_state = {0};\n")
    replace_once(cpu,
        "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
        "        return;\n"
        "    }\n\n"
        "    // extra_buffer op?\n",
        "    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {\n"
        "        return;\n"
        "    }\n\n"
        "    ggml_phase3_route_trace(params, tensor);\n\n"
        "    // RAM-floor arm: retain the graph and all non-MoE work, but replace\n"
        "    // selected-expert output with zeros before any routed weight access.\n"
        "    // This arm is measurement-only and is never used for correctness.\n"
        "    const char * skip_moe = getenv(\"GGML_PHASE3_SKIP_MOE\");\n"
        "    if (tensor->op == GGML_OP_MUL_MAT_ID && skip_moe != NULL &&\n"
        "            atoi(skip_moe) != 0) {\n"
        "        if (params->ith == 0) {\n"
        "            memset(tensor->data, 0, ggml_nbytes(tensor));\n"
        "        }\n"
        "        ggml_barrier(params->threadpool);\n"
        "        return;\n"
        "    }\n\n"
        "    // extra_buffer op?\n")

    # Low-overhead aggregate TSC counters for the dedicated arm.  The counters
    # are emitted at process exit, and are only enabled for profile runs.
    replace_once(iqp,
        "#include <cstring>\n\n#include \"iqp.h\"\n",
        "#include <cstring>\n#include <atomic>\n#include <cstdio>\n\n#if defined(__x86_64__) || defined(_M_X64)\n#include <x86intrin.h>\n#endif\n\n#include \"iqp.h\"\n")
    replace_once(iqp,
        "#define UNUSED GGML_UNUSED\n\n",
        "#define UNUSED GGML_UNUSED\n\n"
        "static bool phase3_profile_enabled() {\n"
        "    const char * value = getenv(\"GGML_PHASE3_PROFILE\");\n"
        "    return value != nullptr && atoi(value) != 0;\n"
        "}\n\n"
        "#if defined(__x86_64__) || defined(_M_X64)\n"
        "static inline uint64_t phase3_tsc() { return __rdtsc(); }\n"
        "#else\n"
        "static inline uint64_t phase3_tsc() { return 0; }\n"
        "#endif\n\n"
        "static std::atomic<uint64_t> phase3_decode_cycles{0};\n"
        "static std::atomic<uint64_t> phase3_gemv_cycles{0};\n"
        "static std::atomic<uint64_t> phase3_decode_calls{0};\n"
        "static std::atomic<uint64_t> phase3_gemv_calls{0};\n"
        "static void phase3_report() {\n"
        "    if (phase3_profile_enabled()) {\n"
        "        fprintf(stderr, \"PHASE3_IQP_PROFILE decode_cycles=%llu gemv_cycles=%llu decode_calls=%llu gemv_calls=%llu\\n\",\n"
        "            (unsigned long long) phase3_decode_cycles.load(),\n"
        "            (unsigned long long) phase3_gemv_cycles.load(),\n"
        "            (unsigned long long) phase3_decode_calls.load(),\n"
        "            (unsigned long long) phase3_gemv_calls.load());\n"
        "    }\n"
        "}\n\n"
        "static bool phase3_report_registered = []() {\n"
        "    if (phase3_profile_enabled()) { atexit(phase3_report); }\n"
        "    return true;\n"
        "}();\n\n")
    replace_once(iqp,
        "        iqp_decode_panel_8(src0->type, src0_cur + r * nb01, nb01, nblocks, panel);\n\n"
        "        // Decode the selected expert's eight output rows once, then use the\n",
        "        const uint64_t decode_start = phase3_profile_enabled() ? phase3_tsc() : 0;\n"
        "        iqp_decode_panel_8(src0->type, src0_cur + r * nb01, nb01, nblocks, panel);\n"
        "        if (phase3_profile_enabled()) {\n"
        "            phase3_decode_cycles.fetch_add(phase3_tsc() - decode_start);\n"
        "            phase3_decode_calls.fetch_add(1);\n"
        "        }\n\n"
        "        // Decode the selected expert's eight output rows once, then use the\n")
    replace_once(iqp,
        "            iqp_gemv_8x8_q8_K(ne00, dst_col + r, ne01, panel, row, 1, IQP_NB_ROWS);\n"
        "            continue;\n",
        "            const uint64_t gemv_start = phase3_profile_enabled() ? phase3_tsc() : 0;\n"
        "            iqp_gemv_8x8_q8_K(ne00, dst_col + r, ne01, panel, row, 1, IQP_NB_ROWS);\n"
        "            if (phase3_profile_enabled()) {\n"
        "                phase3_gemv_cycles.fetch_add(phase3_tsc() - gemv_start);\n"
        "                phase3_gemv_calls.fetch_add(1);\n"
        "            }\n"
        "            continue;\n")
    replace_once(iqp,
        "    }\n}\n\nsize_t ggml_cpu_iqp_scratch_size(const struct ggml_tensor * dst) {\n",
        "    }\n}\n\n"
        "void ggml_compute_forward_mul_mat_id_raw_iq2(const struct ggml_compute_params * params,\n"
        "                                               struct ggml_tensor *               dst,\n"
        "                                               int64_t                            cur_a,\n"
        "                                               const int32_t *                    expert_rows) {\n"
        "    const struct ggml_tensor * src0 = dst->src[0];\n"
        "    const struct ggml_tensor * src1 = dst->src[1];\n"
        "    GGML_TENSOR_BINARY_OP_LOCALS\n"
        "    const int ith = params->ith;\n"
        "    const int nth = params->nth;\n"
        "    const size_t nbw1 = ggml_row_size(GGML_TYPE_Q8_K, ne10);\n"
        "    const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n"
        "    const void * src1_row = (const char *) params->wdata +\n"
        "        ((expert_rows[0] % ne11) + expert_rows[1] * ne11) * nbw1;\n"
        "    const int64_t r0 = (ne01 * ith) / nth;\n"
        "    const int64_t r1 = (ne01 * (ith + 1)) / nth;\n"
        "    for (int64_t r = r0; r < r1; ++r) {\n"
        "        float * dst_col = (float *) ((char *) dst->data +\n"
        "            expert_rows[0] * nb1 + expert_rows[1] * nb2);\n"
        "        ggml_vec_dot_iq2_xxs_q8_K(ne00, dst_col + r, 0,\n"
        "            src0_cur + r * nb01, 0, src1_row, 0, 1);\n"
        "    }\n"
        "}\n\nsize_t ggml_cpu_iqp_scratch_size(const struct ggml_tensor * dst) {\n")

    patch = run_checked(["git", "diff", "--", "ggml/src/ggml-cpu/ggml-cpu.c",
                         "ggml/src/ggml-cpu/iqp.cpp", "ggml/src/ggml-cpu/iqp.h"], cwd=LLAMA).stdout
    (OUT / "single-row-runtime.patch").write_text(patch, encoding="utf-8")
    return {"description": "opt-in exact cne1=1 IQ2_XXS AVX2 GEMV dispatch", "sha256": hashlib.sha256(patch.encode()).hexdigest()}


def clone_patch_build() -> dict:
    if not LLAMA.exists():
        run_checked(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    source_head = run_checked(["git", "rev-parse", "HEAD"], cwd=LLAMA).stdout.strip()
    patch = patch_runtime()
    configure = ["cmake", "-S", str(LLAMA), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
                 "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF",
                 "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF"]
    run_checked(configure, log=OUT / "cmake-configure.log")
    build = ["cmake", "--build", str(BUILD), "--config", "Release", "-j4",
             "--target", "llama-bench", "llama-cli"]
    run_checked(build, log=OUT / "cmake-build.log")
    version = run_checked([str(CLI), "--version"])
    (OUT / "llama-version.txt").write_text(version.stdout, encoding="utf-8")
    return {"commit": LLAMA_COMMIT, "source_head": source_head, "patch": patch,
            "configure_command": configure, "build_command": build,
            "compiler": run_checked(["c++", "--version"]).stdout.splitlines()[0]}


def fetch_model() -> dict:
    if not MODEL.exists() or MODEL.stat().st_size != MODEL_SIZE:
        run_checked(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
                     "-C", "-", "-o", str(MODEL), MODEL_URL], log=OUT / "model-download.log")
    size, digest = MODEL.stat().st_size, sha256(MODEL)
    if size != MODEL_SIZE or digest != MODEL_SHA256:
        raise RuntimeError(f"model identity mismatch: size={size}, sha256={digest}")
    return {"repo": "unsloth/Qwen3.5-35B-A3B-GGUF", "repo_commit": MODEL_REPO_COMMIT,
            "file": MODEL_FILE, "size_bytes": size, "sha256": digest, "url": MODEL_URL}


def read_proc(pid: int) -> dict:
    result = {"mono_ns": time.monotonic_ns(), "rss_kib": 0, "anon_kib": 0,
              "file_kib": 0, "minor_faults": 0, "major_faults": 0, "read_bytes": 0}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"): result["rss_kib"] = int(line.split()[1])
            elif line.startswith("RssAnon:"): result["anon_kib"] = int(line.split()[1])
            elif line.startswith("RssFile:"): result["file_kib"] = int(line.split()[1])
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key == "read_bytes": result[key] = int(value)
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        result["minor_faults"], result["major_faults"] = int(tail[7]), int(tail[9])
        result["valid"] = True
    except (OSError, ValueError):
        result["valid"] = False
    return result


def run_benchmark(label: str, env_overrides: dict[str, str], profile: bool = False) -> dict:
    cmd = [str(BENCH), "-m", str(MODEL), "-p", "0", "-n", str(N_GEN), "-r", str(N_REPEATS),
           "-t", str(THREADS), "-ngl", "0", "-lm", "mmap", "-lzm", "off", "--poll", "0",
           "--output", "json"]
    env = dict(os.environ); env.update(env_overrides)
    if profile: env["GGML_PHASE3_PROFILE"] = "1"
    print("+", command_text(cmd), "ENV", env_overrides, flush=True)
    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    samples = []
    while process.poll() is None:
        samples.append(read_proc(process.pid)); time.sleep(0.02)
    stdout, stderr = process.communicate()
    (OUT / f"{label}.stdout.json").write_text(stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(stderr, encoding="utf-8")
    try: parsed = json.loads(stdout.strip()); row = parsed[0] if len(parsed) == 1 else None
    except (json.JSONDecodeError, TypeError, IndexError): row = None
    valid = [x for x in samples if x.get("valid")]
    profile_lines = [x for x in stderr.splitlines() if x.startswith("PHASE3_IQP_PROFILE")]
    result = {"label": label, "environment": env_overrides, "profile": profile,
              "command": cmd, "returncode": process.returncode,
              "decode_row": row, "sample_tok_s": row.get("samples_ts") if row else None,
              "decode_tok_s": row.get("avg_ts") if row else None,
              "ms_per_token": 1000.0 / row["avg_ts"] if row and row.get("avg_ts") else None,
              "peak_rss_mib": max((x["rss_kib"] for x in valid), default=0) / 1024,
              "peak_anon_mib": max((x["anon_kib"] for x in valid), default=0) / 1024,
              "peak_file_mib": max((x["file_kib"] for x in valid), default=0) / 1024,
              "minor_faults": max((x["minor_faults"] for x in valid), default=0),
              "major_faults": max((x["major_faults"] for x in valid), default=0),
              "profile_lines": profile_lines, "stderr_tail": stderr[-3000:]}
    return result


def generated_response(stdout: str) -> str:
    marker = stdout.find("[Start thinking]")
    if marker < 0: raise RuntimeError("missing generated response marker")
    tail = stdout[marker:]
    summary = SUMMARY_PERF_RE.search(tail)
    if summary: tail = tail[:summary.start()]
    if tail.rstrip().endswith("Exiting..."): tail = tail.rstrip()[:-len("Exiting...")]
    response = tail.strip()
    if not response: raise RuntimeError("empty generated response")
    return response


def run_correctness(label: str, environment: dict[str, str]) -> dict:
    cmd = [str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(THREADS), "-c", "512", "-n", "24",
           "--temp", "0", "--seed", "1234", "--single-turn", "--no-display-prompt", "--no-warmup",
           "--perf", "-lm", "mmap", "-lzm", "off", "--poll", "0", "-p", PROMPT]
    env = dict(os.environ); env.update(environment)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)
    (OUT / f"{label}.stdout.txt").write_text(p.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(p.stderr, encoding="utf-8")
    response = generated_response(p.stdout) if p.returncode == 0 else None
    return {"label": label, "environment": environment, "command": cmd, "returncode": p.returncode,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest() if response is not None else None}


def run_ram_floor() -> dict:
    """Measure the pre-first-routed-expert RSS and short-context floor.

    The route hook is installed in ggml-cpu.c.  The largest process sample
    before its first routed MUL_MAT_ID event is the load/context floor before
    any selected expert is touched; it is paired with 1- and 64-token lazy
    peaks to expose page residency growth.
    """
    results = []
    for label, n_gen, ctx, skip_moe in (("lazy_floor_ctx64", 1, 64, False),
                                        ("lazy_floor_ctx512", 1, 512, False),
                                        ("lazy_steady_ctx512", 64, 512, False),
                                        ("non_moe_floor_ctx512", 64, 512, True)):
        trace = OUT / f"{label}.routes.jsonl"; trace.unlink(missing_ok=True)
        cmd = [str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(THREADS), "-c", str(ctx),
               "-n", str(n_gen), "--temp", "0", "--seed", "1234", "--single-turn",
               "--no-display-prompt", "--no-warmup", "--perf", "-lm", "mmap", "-lzm", "on",
               "--poll", "0", "-p", PROMPT]
        env = dict(os.environ, GGML_PHASE0_ROUTE_TRACE=str(trace))
        if skip_moe:
            env["GGML_PHASE3_SKIP_MOE"] = "1"
        p = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        samples = []
        while p.poll() is None:
            samples.append(read_proc(p.pid)); time.sleep(0.02)
        stdout, stderr = p.communicate()
        valid = [x for x in samples if x.get("valid")]
        first_route_ns = None
        if trace.exists():
            for line in trace.read_text().splitlines():
                first_route_ns = json.loads(line)["mono_ns"]; break
        pre = [x for x in valid if first_route_ns is not None and x["mono_ns"] < first_route_ns]
        results.append({"label": label, "n_gen": n_gen, "context": ctx, "command": cmd,
                        "returncode": p.returncode, "first_route_mono_ns": first_route_ns,
                        "pre_first_route_peak_rss_mib": max((x["rss_kib"] for x in pre), default=0) / 1024,
                        "pre_first_route_peak_anon_mib": max((x["anon_kib"] for x in pre), default=0) / 1024,
                        "pre_first_route_peak_file_mib": max((x["file_kib"] for x in pre), default=0) / 1024,
                        "peak_rss_mib": max((x["rss_kib"] for x in valid), default=0) / 1024,
                        "peak_anon_mib": max((x["anon_kib"] for x in valid), default=0) / 1024,
                        "peak_file_mib": max((x["file_kib"] for x in valid), default=0) / 1024,
                        "minor_faults": max((x["minor_faults"] for x in valid), default=0),
                        "major_faults": max((x["major_faults"] for x in valid), default=0),
                        "trace_events": sum(1 for _ in trace.open()) if trace.exists() else 0,
                        "stderr_tail": stderr[-1000:]})
    return {"description": "lazy mmap RSS before first routed expert and after short decode",
            "logical_model_bytes": MODEL_SIZE,
            "logical_routed_expert_bytes": 8_975_810_560,
            "logical_non_routed_file_bytes": MODEL_SIZE - 8_975_810_560,
            "runs": results}


def hardware_snapshot() -> dict:
    def read(path: str) -> str | None:
        try: return Path(path).read_text(encoding="utf-8")
        except OSError: return None
    try: cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError): cpus = list(range(os.cpu_count() or 0))
    return {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "allowed_cpu_ids": cpus,
            "cpuinfo": read("/proc/cpuinfo"), "lscpu": subprocess.run(["lscpu"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout}


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    result = {"schema": "native-sparse-single-row-avx2/v3", "status": "failed",
              "hypothesis": "A direct exact one-activation IQ2_XXS AVX2 GEMV can raise the resident decode ceiling by removing four-row panel tail duplication.",
              "started_unix": time.time(), "research_source": {"base_commit": RESEARCH_BASE_COMMIT,
              "script_sha256": sha256(Path(__file__))}, "benchmark": {"n_prompt": 0, "n_gen": N_GEN,
              "repeats": N_REPEATS, "threads": THREADS, "poll": 0, "load_mode": "mmap",
              "lazy_mode": "off", "gpu_layers": 0}, "traffic": {"resident_expert_pool": True,
              "scheduled_fresh_expert_bytes_per_token": 0, "logical_selected_expert_bytes_per_token": LOGICAL_SELECTED_EXPERT_BYTES_PER_TOKEN},
              "hardware": hardware_snapshot(), "runs": []}
    try:
        result["build"] = clone_patch_build(); result["model"] = fetch_model()
        result["runs"] = [run_benchmark("generic_control", {"GGML_SINGLE_ROW_IQP": "0", "GGML_RAW_SINGLE_ROW_IQ2": "0"}),
                           run_benchmark("single_row_iqp", {"GGML_SINGLE_ROW_IQP": "1"}),
                           run_benchmark("raw_single_row_iq2", {"GGML_RAW_SINGLE_ROW_IQ2": "1"}),
                           run_benchmark("single_row_iqp_profile", {"GGML_SINGLE_ROW_IQP": "1"}, profile=True)]
        correctness = [run_correctness("correctness_generic", {"GGML_SINGLE_ROW_IQP": "0", "GGML_RAW_SINGLE_ROW_IQ2": "0"}),
                       run_correctness("correctness_single_row_iqp", {"GGML_SINGLE_ROW_IQP": "1"}),
                       run_correctness("correctness_raw_single_row_iq2", {"GGML_RAW_SINGLE_ROW_IQ2": "1"})]
        result["correctness"] = {"runs": correctness, "response_equal": all(x["returncode"] == 0 for x in correctness) and len({x["response_sha256"] for x in correctness}) == 1,
                                  "native_k": 8, "layers": 40, "router_or_weights_changed": False, "no_drop_or_substitution": True}
        result["ram_floor"] = run_ram_floor()
        result["status"] = "ok" if all(x["returncode"] == 0 and x["decode_tok_s"] for x in result["runs"]) and result["correctness"]["response_equal"] else "invalid_result"
    except Exception as exc:
        result["error"] = repr(exc); print(f"ERROR: {exc}", flush=True)
    result["finished_unix"] = time.time()
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "runs": [[x["label"], x.get("decode_tok_s")] for x in result.get("runs", [])], "correctness": result.get("correctness", {}).get("response_equal"), "error": result.get("error")}), flush=True)
    if result["status"] != "ok": raise SystemExit(2)


if __name__ == "__main__":
    main()
