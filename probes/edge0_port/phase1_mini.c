/* phase1_mini: minimal ggml runtime for the Phase-1 shootout.
 * Derived from probes/hw_profile/ggml_mini.c (same box/pin), extended with
 * Q4_0/Q8_0/IQ2_S row sizes needed for the INT4 arms.
 * Row sizes (bytes per block, values per block):
 *   q4_0: 18/32; q8_0: 34/32; q4_K: 144/256; q8_K: 292/256;
 *   iq2_xxs: 66/256; iq2_s: 82/256; q2_K: 84/256; q3_K: 110/256.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "ggml.h"
#include "ggml-quants.h"

void ggml_cpu_init(void) { /* no SILU/GELU tables on this path */ }

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    (void) fmt;
    fprintf(stderr, "ggml_abort at %s:%d\n", file, line);
    abort();
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    size_t bs = 0, vpb = 0;
    switch (type) {
        case GGML_TYPE_Q4_0:   bs = 18;  vpb = 32;  break;
        case GGML_TYPE_Q8_0:   bs = 34;  vpb = 32;  break;
        case GGML_TYPE_Q4_K:   bs = 144; vpb = 256; break;
        case GGML_TYPE_Q8_K:   bs = 292; vpb = 256; break;
        case GGML_TYPE_IQ2_XXS: bs = 66; vpb = 256; break;
        case GGML_TYPE_IQ2_S:  bs = 82;  vpb = 256; break;
        case GGML_TYPE_Q2_K:   bs = 84;  vpb = 256; break;
        case GGML_TYPE_Q3_K:   bs = 110; vpb = 256; break;
        default: fprintf(stderr, "ggml_row_size: unsupported type %d\n", type); abort();
    }
    return (size_t)(ne / (int64_t)vpb) * bs;
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
        case GGML_TYPE_Q4_0: return 18;
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q8_K: return 292;
        case GGML_TYPE_IQ2_XXS: return 66;
        case GGML_TYPE_IQ2_S: return 82;
        case GGML_TYPE_Q2_K: return 84;
        case GGML_TYPE_Q3_K: return 110;
        default: fprintf(stderr, "ggml_type_size: unsupported type %d\n", type); abort();
    }
}

const char * ggml_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "f32";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q4_K: return "q4_K";
        case GGML_TYPE_Q8_K: return "q8_K";
        case GGML_TYPE_IQ2_XXS: return "iq2_xxs";
        case GGML_TYPE_IQ2_S: return "iq2_s";
        case GGML_TYPE_Q2_K: return "q2_K";
        case GGML_TYPE_Q3_K: return "q3_K";
        default: return "?";
    }
}
