// Expert-GEMV format shootout v2: interleaved trials, min-of-N, FTZ/DAZ.
// Same shapes/methodology as v1; robust to background load.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>
#include <pmmintrin.h>

#include "ggml-cpu/quants.h"
#include "ggml.h"

typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*quant_fn)(const float *, void *, int64_t);

struct fmt {
    const char * name;
    enum ggml_type wtype;
    enum ggml_type atype;
    vec_dot_fn dot;
    quant_fn quant_act;
    double bpw;
    uint8_t * pool;
    uint8_t * act;
    size_t nrows[2];
    size_t wrow[2];
};

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void fill_rand(void * p, size_t n) {
    uint64_t s = 0x12345678abcdefULL;
    uint8_t * b = p;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        b[i] = (uint8_t)(s >> 33);
    }
}

int main(void) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    struct fmt fmts[] = {
        {"iq2_xxs x q8_K", GGML_TYPE_IQ2_XXS, GGML_TYPE_Q8_K, ggml_vec_dot_iq2_xxs_q8_K, quantize_row_q8_K, 2.0625, 0,0,{0},{0}},
        {"q8_0 x q8_0   ", GGML_TYPE_Q8_0,    GGML_TYPE_Q8_0, ggml_vec_dot_q8_0_q8_0,     quantize_row_q8_0, 8.5, 0,0,{0},{0}},
        {"q4_0 x q8_0   ", GGML_TYPE_Q4_0,    GGML_TYPE_Q8_0, ggml_vec_dot_q4_0_q8_0,     quantize_row_q8_0, 4.5, 0,0,{0},{0}},
        {"q2_K x q8_K   ", GGML_TYPE_Q2_K,    GGML_TYPE_Q8_K, ggml_vec_dot_q2_K_q8_K,     quantize_row_q8_K, 2.625, 0,0,{0},{0}},
        {"iq3_xxs x q8_K", GGML_TYPE_IQ3_XXS, GGML_TYPE_Q8_K, ggml_vec_dot_iq3_xxs_q8_K, quantize_row_q8_K, 3.4375, 0,0,{0},{0}},
        {"mxfp4 x q8_0  ", GGML_TYPE_MXFP4,   GGML_TYPE_Q8_0, ggml_vec_dot_mxfp4_q8_0,   quantize_row_q8_0, 4.5, 0,0,{0},{0}},
        {"q4_K x q8_K   ", GGML_TYPE_Q4_K,    GGML_TYPE_Q8_K, ggml_vec_dot_q4_K_q8_K,     quantize_row_q8_K, 4.5, 0,0,{0},{0}},
        {"q5_K x q8_K   ", GGML_TYPE_Q5_K,    GGML_TYPE_Q8_K, ggml_vec_dot_q5_K_q8_K,     quantize_row_q8_K, 5.5, 0,0,{0},{0}},
    };
    const int nf = sizeof(fmts) / sizeof(fmts[0]);
    const int Ks[2] = {2048, 512};
    const size_t POOL = 64 << 20;
    float * fact = aligned_alloc(64, 2048 * sizeof(float));
    for (int i = 0; i < 2048; i++) fact[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    for (int f = 0; f < nf; f++) {
        fmts[f].pool = aligned_alloc(64, POOL + 4096);
        fill_rand(fmts[f].pool, POOL + 4096);
        fmts[f].act = aligned_alloc(64, ggml_row_size(fmts[f].atype, 2048));
        for (int s = 0; s < 2; s++) {
            fmts[f].wrow[s] = ggml_row_size(fmts[f].wtype, Ks[s]);
            fmts[f].nrows[s] = (POOL - fmts[f].wrow[s]) / fmts[f].wrow[s];
        }
    }

    double best[8][2];
    for (int f = 0; f < nf; f++) for (int s = 0; s < 2; s++) best[f][s] = 1e18;
    float sum = 0;
    const int TRIALS = 7;
    for (int t = 0; t < TRIALS; t++) {
        for (int f = 0; f < nf; f++) {
            for (int s = 0; s < 2; s++) {
                int K = Ks[s];
                size_t wr = fmts[f].wrow[s], nr = fmts[f].nrows[s];
                fmts[f].quant_act(fact, fmts[f].act, K);
                uint8_t * pool = fmts[f].pool;
                uint64_t t0 = now_ns();
                for (size_t r = 0; r < nr; r++)
                    fmts[f].dot(K, &sum, 0, pool + r * wr, 0, fmts[f].act, 0, 1);
                double nsr = (double)(now_ns() - t0) / nr;
                if (nsr < best[f][s]) best[f][s] = nsr;
            }
        }
        printf("trial %d done sum=%.4f\n", t, sum); fflush(stdout);
    }
    printf("\nmin-of-%d ns/row:\n", TRIALS);
    for (int f = 0; f < nf; f++) {
        double ms = (163840.0 * best[f][0] * 2 + 655360.0 * best[f][1]) / 1e6;
        printf("%-14s bpw=%5.2f gate2048=%8.1fns down512=%7.1fns ST-ms=%7.1f /4ideal=%6.1f\n",
               fmts[f].name, fmts[f].bpw, best[f][0], best[f][1], ms, ms / 4.0);
    }
    return 0;
}
