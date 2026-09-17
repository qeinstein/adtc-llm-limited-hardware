/* ggml_mini: minimal implementations of the 4 ggml entry points kreplay
 * needs, avoiding all of ggml.o/ggml-cpu.o (backend, threading, tables).
 * row sizes verified against ggml-common.h static_asserts:
 *   q8_K = 4+256+32 = 292; q4_K = 4+12+128 = 144;
 *   q5_K = 4+12+128+32 = 176; q6_K = 2+16+192 = 210 (all per 256 vals).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "ggml.h"
#include "ggml-quants.h"

void ggml_cpu_init(void) { /* kreplay uses no SILU/GELU tables */ }

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    (void) fmt;
    fprintf(stderr, "ggml_abort at %s:%d\n", file, line);
    abort();
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    size_t bs = 0;
    switch (type) {
        case GGML_TYPE_Q8_K: bs = 292; break;
        case GGML_TYPE_Q4_K: bs = 144; break;
        case GGML_TYPE_Q5_K: bs = 176; break;
        case GGML_TYPE_Q6_K: bs = 210; break;
        default: fprintf(stderr, "ggml_row_size: unsupported type %d\n", type); abort();
    }
    return (size_t)(ne / 256) * bs;
}

void ggml_quantize_init(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M: iq2xs_init_impl(type); break;
        default: break;
    }
}

size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_Q8_K: return 292;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_IQ2_XXS: return 66;
        default: fprintf(stderr, "ggml_type_size: unsupported type %d\n", type); abort();
    }
}

const char * ggml_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "f32";
        case GGML_TYPE_Q8_K: return "q8_K";
        case GGML_TYPE_Q4_K: return "q4_K";
        case GGML_TYPE_Q5_K: return "q5_K";
        case GGML_TYPE_Q6_K: return "q6_K";
        case GGML_TYPE_IQ2_XXS: return "iq2_xxs";
        case GGML_TYPE_IQ2_S: return "iq2_s";
        default: return "?";
    }
}
