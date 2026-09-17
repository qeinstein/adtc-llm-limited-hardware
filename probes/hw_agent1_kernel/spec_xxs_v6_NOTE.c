// v6: pair-LUT. Fold grid+sign per (grid_byte, field7) pair into ONE L1/L2 lookup.
// key15 = (grid_byte << 7) | field7  -> 8 signed bytes (grid * sign), exact.
// Per group: 4 keys (scalar) + 4 u64 loads + pinsrq/inserti combine; NO sign
// LUT, NO vpsignb, NO parity unpack. Table 32K x 8B = 256KB (LLC-resident).
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include <stdint.h>
#include <string.h>
#include <x86intrin.h>

extern const int8_t spec_keven[1024];

static inline float spec_hsum8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static uint64_t pair_lut[32768];
static int pair_lut_ready = 0;

void spec_xxs_v6_init(void) {
    if (pair_lut_ready) return;
    for (int gb = 0; gb < 256; gb++) {
        uint64_t g = iq2xxs_grid[gb];
        int8_t * gp = (int8_t *)&g;
        for (int f = 0; f < 128; f++) {
            int8_t * sp = (int8_t *)&spec_keven[f * 8];
            uint64_t v = 0;
            for (int j = 0; j < 8; j++) {
                int8_t sv = sp[j] < 0 ? (int8_t)-gp[j] : gp[j];
                v |= ((uint64_t)(uint8_t)sv) << (8 * j);
            }
            pair_lut[(gb << 7) | f] = v;
        }
    }
    pair_lut_ready = 1;
}

// NOTE: signed grid values in [-43,43] fit int8; maddubs takes UNSIGNED first
// operand, so we cannot feed signed values to maddubs directly. Instead apply
// the pair-LUT result to q8 via sign trick: q8s = q8 * sign, computed as
// XOR+SUB with a mask. But the LUT gives signed MAGNITUDES, not masks...
// => v6 as designed needs (magnitudes, signmask) separately. Two 4-bit...
// Redesign: LUT stores (mag_u8[8] packed in u64 LOW-PRECISION?) no.
// Correct approach: LUT key -> u64 where byte = mag, PLUS separate sign via
// keven anyway = no saving. v6 INVALID as stated.
//
// Salvage (v6b): LUT stores sign MASK bytes (0x00/0xFF) per pair: mask8 =
// f(field7) only (grid-independent!) => that's just keven reorganized: 128
// entries, NOT 32K. No win vs baseline (same 4 loads).
// => KILL v6 without benchmarking: the sign application fundamentally needs
// the mask, and mag*sign signed bytes can't use maddubs (u8 x i8).
// This file kept as a record; no GEMV emitted.
