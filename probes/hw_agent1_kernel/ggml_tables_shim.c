// Provides ggml runtime tables needed by x86/quants.c (normally in ggml-cpu.c).
// fp16->fp32 is bit-exact; init once at startup.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_ue4m3[1 << 8];

// Exact copy of ggml-quants.c quantize_row_q8_K_ref (llama.cpp @3057bb6) so the
// x86 TU's quantize_row_q8_K forwarder links standalone.
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } block_q8_K_shim;
static inline int nearest_int_shim(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
void quantize_row_q8_K_ref(const float * x, void * vy, int64_t k) {
    assert(k % 256 == 0);
    const int64_t nb = k / 256;
    block_q8_K_shim * y = (block_q8_K_shim *)vy;
    for (int i = 0; i < nb; i++) {
        float max = 0, amax = 0;
        for (int j = 0; j < 256; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, 256);
            x += 256;
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < 256; ++j) {
            int v = nearest_int_shim(iscale * x[j]);
            y[i].qs[j] = v < 127 ? (int8_t)v : 127;
        }
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1 / iscale;
        x += 256;
    }
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) { f = s; }
        else {
            e = 1;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3FFu;
            f = s | ((e + 112u) << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + 112u) << 23) | (m << 13);
    }
    float r;
    __builtin_memcpy(&r, &f, 4);
    return r;
}

__attribute__((constructor)) static void init_tables(void) {
    for (int i = 0; i < (1 << 16); i++) ggml_table_f32_f16[i] = fp16_to_fp32((uint16_t)i);
    for (int i = 0; i < (1 << 8); i++) ggml_table_f32_e8m0_half[i] = 0.0f; // unused by IQ2/Q8_K path
}
