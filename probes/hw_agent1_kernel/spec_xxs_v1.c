// Specialized IQ2_XXS x Q8_K expert GEMV for N100 (AVX2+FMA+GFNI, no AVX512).
// Exact shapes: gate/up rows K=2048 (nb=8), 512 rows. Bit-exact vs ggml
// ggml_vec_dot_iq2_xxs_q8_K AVX2 path (same int32 accumulation order,
// same fmadd-per-block, same hsum, same final x0.125).
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include <stdint.h>
#include <string.h>
#include <x86intrin.h>

static inline float spec_hsum8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// ---------------- v1: 2-row interleave, stock scalar decode ----------------
// Isolates ILP effect; decode identical to ggml.
extern const int8_t spec_keven[1024]; // exact copy of ggml keven_signs_q2xs

void spec_xxs_v1_gemv(float * y, const void * w, const void * act, int rows, int nb) {
    const int row_bytes = nb * (int)sizeof(block_iq2_xxs);
    const uint64_t * sg = (const uint64_t *)spec_keven;
    for (int r = 0; r < rows; r += 2) {
        const block_iq2_xxs * x0 = (const block_iq2_xxs *)((const char *)w + (r + 0) * row_bytes);
        const block_iq2_xxs * x1 = (const block_iq2_xxs *)((const char *)w + (r + 1) * row_bytes);
        const block_q8_K * yq = (const block_q8_K *)act;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const float d0 = GGML_CPU_FP16_TO_FP32(x0[i].d) * yq[i].d;
            const float d1 = GGML_CPU_FP16_TO_FP32(x1[i].d) * yq[i].d;
            const uint16_t * q20 = x0[i].qs;
            const uint16_t * q21 = x1[i].qs;
            const int8_t * q8 = yq[i].qs;
            __m256i s10 = _mm256_setzero_si256(), s20 = _mm256_setzero_si256();
            __m256i s11 = _mm256_setzero_si256(), s21 = _mm256_setzero_si256();
            for (int ib = 0; ib < QK_K / 32; ib += 2) {
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
                uint32_t a0[4], a1[4];
                memcpy(a0, q20, 16); q20 += 8;
                memcpy(a1, q21, 16); q21 += 8;
                const uint8_t * b0 = (const uint8_t *)a0;
                const uint8_t * b1 = (const uint8_t *)a1;
                // row 0 grid
                const __m256i g0_1 = _mm256_set_epi64x((long long)iq2xxs_grid[b0[3]], (long long)iq2xxs_grid[b0[2]], (long long)iq2xxs_grid[b0[1]], (long long)iq2xxs_grid[b0[0]]);
                const __m256i g0_2 = _mm256_set_epi64x((long long)iq2xxs_grid[b0[11]], (long long)iq2xxs_grid[b0[10]], (long long)iq2xxs_grid[b0[9]], (long long)iq2xxs_grid[b0[8]]);
                // row 1 grid
                const __m256i g1_1 = _mm256_set_epi64x((long long)iq2xxs_grid[b1[3]], (long long)iq2xxs_grid[b1[2]], (long long)iq2xxs_grid[b1[1]], (long long)iq2xxs_grid[b1[0]]);
                const __m256i g1_2 = _mm256_set_epi64x((long long)iq2xxs_grid[b1[11]], (long long)iq2xxs_grid[b1[10]], (long long)iq2xxs_grid[b1[9]], (long long)iq2xxs_grid[b1[8]]);
                const __m256i n0_1 = _mm256_set_epi64x((long long)sg[(a0[1] >> 21) & 127], (long long)sg[(a0[1] >> 14) & 127], (long long)sg[(a0[1] >> 7) & 127], (long long)sg[(a0[1] >> 0) & 127]);
                const __m256i n0_2 = _mm256_set_epi64x((long long)sg[(a0[3] >> 21) & 127], (long long)sg[(a0[3] >> 14) & 127], (long long)sg[(a0[3] >> 7) & 127], (long long)sg[(a0[3] >> 0) & 127]);
                const __m256i n1_1 = _mm256_set_epi64x((long long)sg[(a1[1] >> 21) & 127], (long long)sg[(a1[1] >> 14) & 127], (long long)sg[(a1[1] >> 7) & 127], (long long)sg[(a1[1] >> 0) & 127]);
                const __m256i n1_2 = _mm256_set_epi64x((long long)sg[(a1[3] >> 21) & 127], (long long)sg[(a1[3] >> 14) & 127], (long long)sg[(a1[3] >> 7) & 127], (long long)sg[(a1[3] >> 0) & 127]);
                const __m256i q8s0_1 = _mm256_sign_epi8(q8_1, n0_1);
                const __m256i q8s0_2 = _mm256_sign_epi8(q8_2, n0_2);
                const __m256i q8s1_1 = _mm256_sign_epi8(q8_1, n1_1);
                const __m256i q8s1_2 = _mm256_sign_epi8(q8_2, n1_2);
                const __m256i d00_1 = _mm256_maddubs_epi16(g0_1, q8s0_1);
                const __m256i d00_2 = _mm256_maddubs_epi16(g0_2, q8s0_2);
                const __m256i d01_1 = _mm256_maddubs_epi16(g1_1, q8s1_1);
                const __m256i d01_2 = _mm256_maddubs_epi16(g1_2, q8s1_2);
                const __m256i p00_1 = _mm256_madd_epi16(d00_1, _mm256_set1_epi16((short)(2 * (a0[1] >> 28) + 1)));
                const __m256i p00_2 = _mm256_madd_epi16(d00_2, _mm256_set1_epi16((short)(2 * (a0[3] >> 28) + 1)));
                const __m256i p01_1 = _mm256_madd_epi16(d01_1, _mm256_set1_epi16((short)(2 * (a1[1] >> 28) + 1)));
                const __m256i p01_2 = _mm256_madd_epi16(d01_2, _mm256_set1_epi16((short)(2 * (a1[3] >> 28) + 1)));
                s10 = _mm256_add_epi32(s10, p00_1);
                s20 = _mm256_add_epi32(s20, p00_2);
                s11 = _mm256_add_epi32(s11, p01_1);
                s21 = _mm256_add_epi32(s21, p01_2);
            }
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(_mm256_add_epi32(s10, s20)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(_mm256_add_epi32(s11, s21)), acc1);
        }
        y[r + 0] = 0.125f * spec_hsum8(acc0);
        y[r + 1] = 0.125f * spec_hsum8(acc1);
    }
}
