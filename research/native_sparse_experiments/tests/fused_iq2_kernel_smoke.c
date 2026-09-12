#include "ggml-common.h"
#include "ggml-cpu/quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The production runtime initializes this lookup table during startup. */
float ggml_table_f32_f16[1 << 16];

void ggml_vec_dot_iq2_xxs_q8_K_4x1(int n, float * s, size_t row_stride,
                                   const void * vx, const void * vy);

static uint32_t next_value(uint32_t * state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

int main(void) {
    const int n = 512;
    const int nb = n / QK_K;
    const size_t row_stride = (size_t) nb * sizeof(block_iq2_xxs);
    block_iq2_xxs * rows = aligned_alloc(64, 4 * row_stride);
    block_q8_K * activation = aligned_alloc(64, (size_t) nb * sizeof(block_q8_K));
    if (rows == NULL || activation == NULL) {
        return 2;
    }

    uint32_t state = 7;
    for (int r = 0; r < 4; ++r) {
        for (int i = 0; i < nb; ++i) {
            block_iq2_xxs * block = (block_iq2_xxs *) ((char *) rows + r * row_stride) + i;
            block->d = (ggml_half) (0x3800u + (r + i) * 37u);
            for (size_t q = 0; q < sizeof(block->qs) / sizeof(block->qs[0]); ++q) {
                block->qs[q] = (uint16_t) next_value(&state);
            }
        }
    }
    for (int i = 0; i < nb; ++i) {
        activation[i].d = 0.01f + 0.001f * (float) i;
        for (int q = 0; q < QK_K; ++q) {
            activation[i].qs[q] = (int8_t) (next_value(&state) & 0x7f);
        }
    }

    float reference[4] = {0};
    float fused[4] = {0};
    for (int r = 0; r < 4; ++r) {
        ggml_vec_dot_iq2_xxs_q8_K(n, &reference[r], 0,
            (const char *) rows + r * row_stride, 0, activation, 0, 1);
    }
    ggml_vec_dot_iq2_xxs_q8_K_4x1(n, fused, row_stride, rows, activation);
    for (int r = 0; r < 4; ++r) {
        if (memcmp(&reference[r], &fused[r], sizeof(float)) != 0 &&
                fabsf(reference[r] - fused[r]) > 1e-6f) {
            fprintf(stderr, "row %d mismatch: %.9g %.9g\n", r, reference[r], fused[r]);
            free(rows);
            free(activation);
            return 1;
        }
    }
    free(rows);
    free(activation);
    puts("fused IQ2_XXS kernel smoke: PASS");
    return 0;
}
