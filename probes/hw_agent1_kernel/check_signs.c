// Check iqp-style GFNI sign expansion bit order vs keven_signs_q2xs LUT.
#include <stdint.h>
#include <stdio.h>
#include <x86intrin.h>

static inline uint8_t unpack7(uint32_t v) {
    uint32_t p = v ^ (v >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    return (uint8_t)(v ^ ((p & 1) << 7));
}

static const int8_t keven[1024] = {
#include "keven.inc"
};

int main(void) {
    const __m256i bcast = _mm256_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
                                           2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i sel = _mm256_set1_epi64x((int64_t)0x8040201008040201ULL);
    const __m256i one = _mm256_set1_epi8(1);
    int bad = 0;
    for (int q = 0; q < 128; q++) {
        uint8_t sb = unpack7((uint32_t)q);
        uint32_t signs = (uint32_t)sb * 0x01010101u;
        __m256i sv = _mm256_shuffle_epi8(_mm256_set1_epi32((int32_t)signs), bcast);
        __m256i m = _mm256_gf2p8affine_epi64_epi8(sel, sv, 0);
        __m256i m1 = _mm256_or_si256(m, one); // 0xFF->-1, 0x00->+1
        int8_t out[32];
        _mm256_storeu_si256((__m256i *)out, m1);
        for (int j = 0; j < 8; j++) {
            if (out[j] != keven[q * 8 + j]) {
                if (bad < 5) printf("q=%d j=%d got=%d want=%d sb=%02x\n", q, j, out[j], keven[q*8+j], sb);
                bad++;
            }
        }
    }
    printf(bad ? "MISMATCH x%d\n" : "MATCH all 128 sign entries\n", bad);
    return bad != 0;
}
