"""Phase 5B: external W2 LUT and exact IQ2-derived layout ceiling.

This is a shape-matched CPU microbenchmark, not a model-quality experiment.
It compares the pinned llama.cpp IQ2_XXS AVX2 access pattern with an exact
offline-resolved IQ2 metadata layout and a T-MAC-style W2A8 lookup kernel.
The latter intentionally uses a different unsigned-W2 representation and is
reported only as an external compute ceiling.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path


SCRATCH = Path("/tmp/native-sparse-lowbit-ceiling")
LLAMA = SCRATCH / "llama.cpp"
OUT = Path("/kaggle/working/native-sparse-lowbit-ceiling-v1-results")
LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
RESEARCH_BASE_COMMIT = "cd2be78"
EXPERTS = 8
REPEATS = 80


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
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_setup() -> dict:
    if not LLAMA.exists():
        run_checked(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    quants = LLAMA / "ggml/src/ggml-cpu/arch/x86/quants.c"
    text = quants.read_text(encoding="utf-8")
    start = text.index("static const int8_t keven_signs_q2xs[1024] = {")
    end = text.index("};", start) + 2
    return {"commit": run_checked(["git", "rev-parse", "HEAD"], cwd=LLAMA).stdout.strip(),
            "quants_file": str(quants), "sign_table_sha256": hashlib.sha256(text[start:end].encode()).hexdigest(),
            "sign_table_decl": text[start:end]}


def benchmark_source(sign_table_decl: str) -> str:
    return r'''#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <immintrin.h>

#define EXPERTS 8
#define REPEATS 80
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

''' + sign_table_decl + r'''

static constexpr int KQ = 256;
static constexpr int GROUP = 64;

struct ResolvedPair {
    uint64_t grid;
    uint64_t signs;
};

struct CustomGroup {
    uint16_t code[8];
    uint8_t scale;
};

struct CustomBlock {
    ggml_half d;
    CustomGroup group[4];
};

static_assert(sizeof(block_iq2_xxs) == 66, "unexpected IQ2_XXS size");
static_assert(sizeof(CustomBlock) == 74, "unexpected custom block size");

static inline uint64_t sign_word(int index) {
    uint64_t out = 0;
    for (int i = 0; i < 8; ++i) {
        const uint8_t s = (uint8_t) keven_signs_q2xs[index * 8 + i];
        out |= (uint64_t) s << (8 * i);
    }
    return out;
}

static inline int iq2_dot(const block_iq2_xxs * x, const int8_t * y, int cols) {
    const uint64_t * signs64 = (const uint64_t *) keven_signs_q2xs;
    __m256 accum = _mm256_setzero_ps();
    for (int i = 0; i < cols / 256; ++i) {
        const int8_t * q8 = y + i * 256;
        const uint16_t * q2 = x[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int g = 0; g < 4; ++g) {
            uint32_t aux[4];
            std::memcpy(aux, q2 + g * 8, sizeof(aux));
            const uint8_t * bytes = (const uint8_t *) aux;
            const __m256i q21 = _mm256_set_epi64x(
                iq2xxs_grid[bytes[3]], iq2xxs_grid[bytes[2]],
                iq2xxs_grid[bytes[1]], iq2xxs_grid[bytes[0]]);
            const __m256i q22 = _mm256_set_epi64x(
                iq2xxs_grid[bytes[11]], iq2xxs_grid[bytes[10]],
                iq2xxs_grid[bytes[9]], iq2xxs_grid[bytes[8]]);
            const __m256i s1 = _mm256_set_epi64x(
                signs64[(aux[1] >> 21) & 127], signs64[(aux[1] >> 14) & 127],
                signs64[(aux[1] >> 7) & 127], signs64[aux[1] & 127]);
            const __m256i s2 = _mm256_set_epi64x(
                signs64[(aux[3] >> 21) & 127], signs64[(aux[3] >> 14) & 127],
                signs64[(aux[3] >> 7) & 127], signs64[aux[3] & 127]);
            const __m256i a = _mm256_loadu_si256((const __m256i *)(q8 + g * 64));
            const __m256i b = _mm256_loadu_si256((const __m256i *)(q8 + g * 64 + 32));
            const __m256i d1 = _mm256_maddubs_epi16(q21, _mm256_sign_epi8(a, s1));
            const __m256i d2 = _mm256_maddubs_epi16(q22, _mm256_sign_epi8(b, s2));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(d1, _mm256_set1_epi16(2 * (aux[1] >> 28) + 1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(d2, _mm256_set1_epi16(2 * (aux[3] >> 28) + 1)));
        }
        accum = _mm256_add_ps(accum, _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)));
    }
    alignas(32) int32_t tmp[8];
    _mm256_store_si256((__m256i *) tmp, _mm256_cvtps_epi32(accum));
    int sum = 0;
    for (int i = 0; i < 8; ++i) sum += tmp[i];
    return sum;
}

static inline int custom_dot(const CustomBlock * x, const int8_t * y, int cols,
                             const ResolvedPair * table) {
    __m256 accum = _mm256_setzero_ps();
    for (int i = 0; i < cols / 256; ++i) {
        const int8_t * q8 = y + i * 256;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int g = 0; g < 4; ++g) {
            const CustomGroup & cg = x[i].group[g];
            const ResolvedPair & p0 = table[cg.code[0]];
            const ResolvedPair & p1 = table[cg.code[1]];
            const ResolvedPair & p2 = table[cg.code[2]];
            const ResolvedPair & p3 = table[cg.code[3]];
            const ResolvedPair & p4 = table[cg.code[4]];
            const ResolvedPair & p5 = table[cg.code[5]];
            const ResolvedPair & p6 = table[cg.code[6]];
            const ResolvedPair & p7 = table[cg.code[7]];
            const __m256i q21 = _mm256_set_epi64x(p3.grid, p2.grid, p1.grid, p0.grid);
            const __m256i q22 = _mm256_set_epi64x(p7.grid, p6.grid, p5.grid, p4.grid);
            const __m256i s1 = _mm256_set_epi64x(p3.signs, p2.signs, p1.signs, p0.signs);
            const __m256i s2 = _mm256_set_epi64x(p7.signs, p6.signs, p5.signs, p4.signs);
            const __m256i a = _mm256_loadu_si256((const __m256i *)(q8 + g * 64));
            const __m256i b = _mm256_loadu_si256((const __m256i *)(q8 + g * 64 + 32));
            const __m256i d1 = _mm256_maddubs_epi16(q21, _mm256_sign_epi8(a, s1));
            const __m256i d2 = _mm256_maddubs_epi16(q22, _mm256_sign_epi8(b, s2));
            const int lo = cg.scale & 15;
            const int hi = cg.scale >> 4;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(d1, _mm256_set1_epi16(2 * lo + 1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(d2, _mm256_set1_epi16(2 * hi + 1)));
        }
        accum = _mm256_add_ps(accum, _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)));
    }
    alignas(32) int32_t tmp[8];
    _mm256_store_si256((__m256i *) tmp, _mm256_cvtps_epi32(accum));
    int sum = 0;
    for (int i = 0; i < 8; ++i) sum += tmp[i];
    return sum;
}

static void repack(const std::vector<block_iq2_xxs> & src,
                   std::vector<CustomBlock> & dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i].d = src[i].d;
        for (int g = 0; g < 4; ++g) {
            uint32_t aux[4];
            std::memcpy(aux, src[i].qs + g * 8, sizeof(aux));
            const uint8_t * bytes = (const uint8_t *) aux;
            for (int j = 0; j < 4; ++j) {
                dst[i].group[g].code[j] = (uint16_t) bytes[j] | (uint16_t)(((aux[1] >> (7 * j)) & 127) << 8);
                dst[i].group[g].code[j + 4] = (uint16_t) bytes[j + 8] | (uint16_t)(((aux[3] >> (7 * j)) & 127) << 8);
            }
            dst[i].group[g].scale = (uint8_t)(((aux[3] >> 28) << 4) | (aux[1] >> 28));
        }
    }
}

struct W2Matrix { int rows; int cols; std::vector<uint16_t> packed; };

static W2Matrix make_w2(int rows, int cols, std::mt19937 & rng) {
    W2Matrix m{rows, cols, std::vector<uint16_t>((size_t) rows * (cols / 8))};
    std::uniform_int_distribution<int> d(0, 255);
    for (auto & x : m.packed) {
        // Offline W2 packing: low and high bit planes are already separated
        // into one byte each.  The hot path performs lookup only.
        x = (uint16_t)d(rng) | (uint16_t)(d(rng) << 8);
    }
    return m;
}

static std::vector<int32_t> make_lut(const int8_t * act, int cols) {
    const int groups = cols / 8;
    std::vector<int32_t> lut((size_t) groups * 2 * 256);
    for (int g = 0; g < groups; ++g) {
        for (int mask = 0; mask < 256; ++mask) {
            int sum = 0;
            for (int j = 0; j < 8; ++j) if (mask & (1 << j)) sum += act[g * 8 + j];
            lut[((size_t)g * 2 + 0) * 256 + mask] = sum;
            lut[((size_t)g * 2 + 1) * 256 + mask] = sum;
        }
    }
    return lut;
}

static double tmac_w2_work(const std::vector<W2Matrix> & matrices,
                            const std::vector<int32_t> & lut,
                            const int8_t * act, int repeats,
                            volatile int64_t * checksum) {
    const int groups = matrices[0].cols / 8;
    auto start = std::chrono::steady_clock::now();
    int64_t local = 0;
    for (int rep = 0; rep < repeats; ++rep) {
        for (const W2Matrix & m : matrices) {
            for (int r = 0; r < m.rows; ++r) {
                int sum = 0;
                for (int g = 0; g < groups; ++g) {
                    const uint16_t w = m.packed[(size_t)r * groups + g];
                    const uint8_t lo = (uint8_t)(w & 255);
                    const uint8_t hi = (uint8_t)(w >> 8);
                    sum += lut[((size_t)g * 2 + 0) * 256 + lo];
                    sum += 2 * lut[((size_t)g * 2 + 1) * 256 + hi];
                }
                local += sum;
            }
        }
    }
    *checksum += local;
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static double tmac_w2_gather_work(const std::vector<W2Matrix> & matrices,
                                  const std::vector<int32_t> & lut,
                                  int repeats, volatile int64_t * checksum) {
    const int groups = matrices[0].cols / 8;
    auto start = std::chrono::steady_clock::now();
    int64_t local = 0;
    for (int rep = 0; rep < repeats; ++rep) {
        for (const W2Matrix & m : matrices) {
            for (int g = 0; g < groups; ++g) {
                const int * base0 = lut.data() + ((size_t)g * 2 + 0) * 256;
                const int * base1 = lut.data() + ((size_t)g * 2 + 1) * 256;
                for (int r = 0; r < m.rows; r += 8) {
                    const uint16_t * p = m.packed.data() + (size_t)r * groups + g;
                    const __m256i idx0 = _mm256_set_epi32(
                        p[7 * groups] & 255, p[6 * groups] & 255,
                        p[5 * groups] & 255, p[4 * groups] & 255,
                        p[3 * groups] & 255, p[2 * groups] & 255,
                        p[1 * groups] & 255, p[0 * groups] & 255);
                    const __m256i idx1 = _mm256_set_epi32(
                        (p[7 * groups] >> 8) & 255, (p[6 * groups] >> 8) & 255,
                        (p[5 * groups] >> 8) & 255, (p[4 * groups] >> 8) & 255,
                        (p[3 * groups] >> 8) & 255, (p[2 * groups] >> 8) & 255,
                        (p[1 * groups] >> 8) & 255, (p[0 * groups] >> 8) & 255);
                    const __m256i a = _mm256_i32gather_epi32(base0, idx0, 4);
                    const __m256i b = _mm256_i32gather_epi32(base1, idx1, 4);
                    alignas(32) int32_t sums[8];
                    _mm256_store_si256((__m256i *)sums, _mm256_add_epi32(a, _mm256_slli_epi32(b, 1)));
                    for (int j = 0; j < 8; ++j) local += sums[j];
                }
            }
        }
    }
    *checksum += local;
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static double iq2_work(int rows, int cols, int experts, int repeats,
                       const std::vector<int8_t> & act, volatile int64_t * checksum,
                       bool custom, const std::vector<block_iq2_xxs> & original,
                       const std::vector<CustomBlock> & packed,
                       const ResolvedPair * table) {
    const int blocks = cols / KQ;
    double seconds = 0.0;
    for (int rep = 0; rep < repeats; ++rep) {
        auto start = std::chrono::steady_clock::now();
        int64_t local = 0;
        for (int e = 0; e < experts; ++e) {
            for (int r = 0; r < rows; ++r) {
                const size_t off = ((size_t)e * rows + r) * blocks;
            local += custom ? custom_dot(packed.data() + off, act.data(), cols, table)
                            : iq2_dot(original.data() + off, act.data(), cols);
            }
        }
        *checksum += local;
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    return seconds;
}

int main() {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::vector<ResolvedPair> table(32768);
    for (int sign = 0; sign < 128; ++sign) {
        for (int grid = 0; grid < 256; ++grid) {
            table[(sign << 8) | grid] = {iq2xxs_grid[grid], sign_word(sign)};
        }
    }
    volatile int64_t checksum = 0;
    std::vector<std::pair<int, int>> shapes = {{512, 2048}, {2048, 512}};
    std::printf("schema=lowbit-ceiling/v1 experts=%d repeats=%d\n", EXPERTS, REPEATS);
    for (auto [rows, cols] : shapes) {
        const int blocks = cols / KQ;
        const size_t count = (size_t) EXPERTS * rows * blocks;
        std::vector<block_iq2_xxs> original(count);
        for (auto & b : original) {
            b.d = 0x3c00;
            for (auto & q : b.qs) q = (uint16_t)(byte_dist(rng) | (byte_dist(rng) << 8));
        }
        std::vector<CustomBlock> packed;
        repack(original, packed);
        const int8_t * act_ptr = nullptr;
        std::vector<int8_t> act(cols);
        for (auto & x : act) x = (int8_t)(byte_dist(rng) & 127);
        act_ptr = act.data();
        const int exact_a = iq2_dot(original.data(), act_ptr, cols);
        const int exact_b = custom_dot(packed.data(), act_ptr, cols, table.data());
        if (exact_a != exact_b) {
            std::fprintf(stderr, "exact mismatch shape=%d,%d original=%d custom=%d\n", rows, cols, exact_a, exact_b);
            return 2;
        }
        const double iq_seconds = iq2_work(rows, cols, EXPERTS, REPEATS, act, &checksum, false, original, packed, table.data());
        const double custom_seconds = iq2_work(rows, cols, EXPERTS, REPEATS, act, &checksum, true, original, packed, table.data());
        std::vector<W2Matrix> w2s;
        for (int e = 0; e < EXPERTS; ++e) w2s.push_back(make_w2(rows, cols, rng));
        const std::vector<int32_t> lut = make_lut(act_ptr, cols);
        const double lut_seconds = tmac_w2_work(w2s, lut, act_ptr, REPEATS, &checksum);
        const double gather_seconds = tmac_w2_gather_work(w2s, lut, REPEATS, &checksum);
        const double macs = (double) EXPERTS * rows * cols * REPEATS;
        const double original_bytes = (double) count * sizeof(block_iq2_xxs);
        const double custom_bytes = (double) packed.size() * sizeof(CustomBlock);
        const double w2_bytes = (double) w2s.size() * w2s[0].packed.size() * sizeof(uint16_t);
        std::printf("shape=%dx%d original_seconds=%.9f custom_seconds=%.9f lut_w2_seconds=%.9f gather_w2_seconds=%.9f "
                    "original_gmacs=%.6f custom_gmacs=%.6f lut_w2_gmacs=%.6f gather_w2_gmacs=%.6f "
                    "original_bytes=%.0f custom_bytes=%.0f custom_overhead_pct=%.3f w2_bytes=%.0f\n",
                    rows, cols, iq_seconds, custom_seconds, lut_seconds, gather_seconds,
                    macs / iq_seconds / 1e9, macs / custom_seconds / 1e9,
                    macs / lut_seconds / 1e9, macs / gather_seconds / 1e9,
                    original_bytes, custom_bytes,
                    100.0 * (custom_bytes / original_bytes - 1.0), w2_bytes);
    }
    std::printf("checksum=%lld\n", (long long) checksum);
    return 0;
}
'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "native-sparse-lowbit-ceiling/v1", "status": "failed",
        "hypothesis": "A decoder-natural exact IQ2 layout and a T-MAC-style W2A8 lookup path will expose whether GGUF IQ2 metadata handling, rather than 2-bit arithmetic, is the main expert-kernel limit.",
        "research_source": {"base_commit": RESEARCH_BASE_COMMIT, "script_sha256": sha256(Path(__file__))},
        "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                     "cpu_count": os.cpu_count(), "cpuinfo": Path("/proc/cpuinfo").read_text(),
                     "lscpu": run_checked(["lscpu"]).stdout, "compiler": run_checked(["c++", "--version"]).stdout},
        "representation": {
            "original": {"name": "IQ2_XXS", "bytes_per_256": 66, "bpw": 2.0625},
            "custom": {"name": "resolved_code_sign_index", "bytes_per_256": 74, "bpw": 2.3125,
                       "semantics": "exact offline repack: combined grid/sign indices plus two 4-bit scales"},
            "external": {"name": "unsigned W2A8 T-MAC-style LUT", "bpw": 2.0,
                         "semantics": "performance ceiling only; not IQ2-equivalent"},
        },
    }
    try:
        setup = source_setup()
        result["runtime"] = {k: v for k, v in setup.items() if k != "sign_table_decl"}
        source = benchmark_source(setup["sign_table_decl"])
        cpp = SCRATCH / "lowbit_ceiling.cpp"
        binary = SCRATCH / "lowbit_ceiling"
        cpp.write_text(source, encoding="utf-8")
        result["source_sha256"] = sha256(cpp)
        compile_cmd = ["c++", "-O3", "-DNDEBUG", "-mavx2", "-mfma", "-march=haswell",
                       "-std=c++17", "-I", str(LLAMA / "ggml/src"), str(cpp), "-o", str(binary)]
        run_checked(compile_cmd, log=OUT / "compile.log")
        result["compile_command"] = compile_cmd
        bench = run_checked([str(binary)], log=OUT / "benchmark.log")
        (OUT / "benchmark.stdout.txt").write_text(bench.stdout, encoding="utf-8")
        (OUT / "benchmark.stderr.txt").write_text(bench.stderr, encoding="utf-8")
        result["benchmark_stdout"] = bench.stdout
        result["status"] = "ok"
    except Exception as exc:
        result["error"] = repr(exc)
    result["finished_unix"] = time.time()
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "error": result.get("error")}), flush=True)
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
