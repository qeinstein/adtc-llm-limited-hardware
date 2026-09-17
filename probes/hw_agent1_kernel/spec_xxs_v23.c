// v2: single-row + GFNI in-register sign expansion (verified bit-exact vs LUT).
// v3: v2 + vpgatherqq grid decode (replaces 8 scalar loads + 2 set_epi64x).
// v4: v3 + 2-row interleave (re-test ILP once frontend is narrow).
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

static inline uint8_t unpack7(uint32_t v) {
    uint32_t p = v ^ (v >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    return (uint8_t)(v ^ ((p & 1) << 7));
}

// expand 4 sign-groups (7-bit fields of w) to 32x {-1,+1} bytes for sign_epi8
static inline __m256i expand_signs(uint32_t w, const __m256i bcast, const __m256i sel, const __m256i one) {
    uint32_t s = (uint32_t)unpack7((w >> 0) & 127) | ((uint32_t)unpack7((w >> 7) & 127) << 8) |
                 ((uint32_t)unpack7((w >> 14) & 127) << 16) | ((uint32_t)unpack7((w >> 21) & 127) << 24);
    __m256i sv = _mm256_shuffle_epi8(_mm256_set1_epi32((int32_t)s), bcast);
    __m256i m = _mm256_gf2p8affine_epi64_epi8(sel, sv, 0);
    return _mm256_or_si256(m, one);
}

static inline __m256i gather_grid4(const uint64_t * grid, uint32_t packed4) {
    // packed4 bytes -> 4 u64 grid entries, lane order [b0,b1,b2,b3]
    __m128i b = _mm_cvtsi32_si128((int)packed4);
    __m256i idx32 = _mm256_cvtepu8_epi32(b);
    __m256i idx64 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(idx32));
    __m256i g = _mm256_setzero_si256();
    g = _mm256_mask_i64gather_epi64(g, (const long long *)grid, idx64,
                                      _mm256_castsi128_si256(_mm_set1_epi32(-1)), 8);
    return g;
}

void spec_xxs_v2_gemv(float * y, const void * w, const void * act, int rows, int nb) {
    const __m256i bcast = _mm256_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
                                           2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i sel = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    const __m256i one = _mm256_set1_epi8(1);
    const int row_bytes = nb * (int)sizeof(block_iq2_xxs);
    for (int r = 0; r < rows; r++) {
        const block_iq2_xxs * x = (const block_iq2_xxs *)((const char *)w + r * row_bytes);
        const block_q8_K * yq = (const block_q8_K *)act;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * yq[i].d;
            const uint16_t * q2 = x[i].qs;
            const int8_t * q8 = yq[i].qs;
            __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
            for (int ib = 0; ib < QK_K / 32; ib += 2) {
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
                uint32_t a[4];
                memcpy(a, q2, 16); q2 += 8;
                const uint8_t * b = (const uint8_t *)a;
                const __m256i g_1 = _mm256_set_epi64x((long long)iq2xxs_grid[b[3]], (long long)iq2xxs_grid[b[2]], (long long)iq2xxs_grid[b[1]], (long long)iq2xxs_grid[b[0]]);
                const __m256i g_2 = _mm256_set_epi64x((long long)iq2xxs_grid[b[11]], (long long)iq2xxs_grid[b[10]], (long long)iq2xxs_grid[b[9]], (long long)iq2xxs_grid[b[8]]);
                const __m256i n_1 = expand_signs(a[1], bcast, sel, one);
                const __m256i n_2 = expand_signs(a[3], bcast, sel, one);
                const __m256i q8s_1 = _mm256_sign_epi8(q8_1, n_1);
                const __m256i q8s_2 = _mm256_sign_epi8(q8_2, n_2);
                const __m256i dot1 = _mm256_maddubs_epi16(g_1, q8s_1);
                const __m256i dot2 = _mm256_maddubs_epi16(g_2, q8s_2);
                const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((short)(2 * (a[1] >> 28) + 1)));
                const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((short)(2 * (a[3] >> 28) + 1)));
                sumi1 = _mm256_add_epi32(sumi1, p1);
                sumi2 = _mm256_add_epi32(sumi2, p2);
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), acc);
        }
        y[r] = 0.125f * spec_hsum8(acc);
    }
}

void spec_xxs_v3_gemv(float * y, const void * w, const void * act, int rows, int nb) {
    const __m256i bcast = _mm256_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
                                           2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i sel = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    const __m256i one = _mm256_set1_epi8(1);
    const int row_bytes = nb * (int)sizeof(block_iq2_xxs);
    for (int r = 0; r < rows; r++) {
        const block_iq2_xxs * x = (const block_iq2_xxs *)((const char *)w + r * row_bytes);
        const block_q8_K * yq = (const block_q8_K *)act;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * yq[i].d;
            const uint16_t * q2 = x[i].qs;
            const int8_t * q8 = yq[i].qs;
            __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
            for (int ib = 0; ib < QK_K / 32; ib += 2) {
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
                uint32_t a[4];
                memcpy(a, q2, 16); q2 += 8;
                const __m256i g_1 = gather_grid4(iq2xxs_grid, a[0]);
                const __m256i g_2 = gather_grid4(iq2xxs_grid, a[2]);
                const __m256i n_1 = expand_signs(a[1], bcast, sel, one);
                const __m256i n_2 = expand_signs(a[3], bcast, sel, one);
                const __m256i q8s_1 = _mm256_sign_epi8(q8_1, n_1);
                const __m256i q8s_2 = _mm256_sign_epi8(q8_2, n_2);
                const __m256i dot1 = _mm256_maddubs_epi16(g_1, q8s_1);
                const __m256i dot2 = _mm256_maddubs_epi16(g_2, q8s_2);
                const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16((short)(2 * (a[1] >> 28) + 1)));
                const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16((short)(2 * (a[3] >> 28) + 1)));
                sumi1 = _mm256_add_epi32(sumi1, p1);
                sumi2 = _mm256_add_epi32(sumi2, p2);
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), acc);
        }
        y[r] = 0.125f * spec_hsum8(acc);
    }
}
