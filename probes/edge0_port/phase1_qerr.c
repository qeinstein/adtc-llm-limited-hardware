// Phase-1 quant-error tool: compares IQ2 / Q2_K / Q4_0 / Q4_K against bf16
// source for one expert (L20 E0/E1), weight-level and GEMV-output-level.
// Also quantifies the double-quant penalty (Q* from IQ2-dequant vs from bf16)
// carried by the speed-test pools. All stats in-process; no big dumps.
// Usage: phase1_qerr <E>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
#include "ggml.h"

#define BF16_DIR "/tmp/edge0_phase1/bf16"
#define RAW_DIR "/tmp/agent1_raw"

static void die(const char *m) { fprintf(stderr, "ERR %s\n", m); exit(1); }

static void read_full(const char *path, void *buf, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERR open %s\n", path); exit(1); }
    size_t r = fread(buf, 1, n, f);
    fclose(f);
    if (r != n) { fprintf(stderr, "ERR short %s %zu/%zu\n", path, r, n); exit(1); }
}

static void bf16_to_f32(const uint16_t *s, float *d, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t u = (uint32_t)s[i] << 16;
        memcpy(d + i, &u, 4);
    }
}

static void stats(const char *tag, const float *v, const float *ref, size_t n) {
    double se = 0, sr = 0, mx = 0, dot = 0, nv = 0;
    for (size_t i = 0; i < n; i++) {
        double e = (double)v[i] - ref[i];
        se += e * e; sr += (double)ref[i] * ref[i];
        double a = fabs(e);
        if (a > mx) mx = a;
        dot += (double)v[i] * ref[i]; nv += (double)v[i] * v[i];
    }
    double rel = sqrt(se / sr);
    double cos = dot / sqrt(nv * sr + 1e-30);
    double snr = -20.0 * log10(rel + 1e-30);
    printf("%-28s relL2=%.5f SNR_dB=%6.2f maxabs=%.4g cos=%.6f\n",
           tag, rel, snr, mx, cos);
}

// y = W x, W row-major rows x K
static void gemv(const float *W, const float *x, float *y, int rows, int K) {
    for (int r = 0; r < rows; r++) {
        double s = 0;
        const float *w = W + (size_t)r * K;
        for (int j = 0; j < K; j++) s += (double)w[j] * x[j];
        y[r] = (float)s;
    }
}

static void fill_randn(float *x, int n) {
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    double sum = 0;
    for (int i = 0; i < n; i++) {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        double u1 = ((s * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        double u2 = ((s * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
        if (u1 < 1e-12) u1 = 1e-12;
        double g = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
        x[i] = (float)g; sum += g * g;
    }
    double f = 1.0 / sqrt(sum / n);
    for (int i = 0; i < n; i++) x[i] = (float)(x[i] * f);
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: phase1_qerr <E>");
    int E = atoi(argv[1]);
    ggml_quantize_init(GGML_TYPE_IQ2_XXS);
    ggml_quantize_init(GGML_TYPE_IQ2_S);

    // bf16 source: gate_up [1024,2048], down [2048,512]
    char path[256];
    snprintf(path, sizeof(path), "%s/L20_E%03d_gate_up_proj.bf16", BF16_DIR, E);
    uint16_t *gu = malloc(1024 * 2048 * 2);
    read_full(path, gu, 1024 * 2048 * 2);
    snprintf(path, sizeof(path), "%s/L20_E%03d_down_proj.bf16", BF16_DIR, E);
    uint16_t *dn = malloc(2048 * 512 * 2);
    read_full(path, dn, 2048 * 512 * 2);
    float *gu_f = malloc(1024 * 2048 * 4);
    float *dn_f = malloc(2048 * 512 * 4);
    bf16_to_f32(gu, gu_f, 1024 * 2048);
    bf16_to_f32(dn, dn_f, 2048 * 512);
    free(gu); free(dn);
    float *halfA = gu_f;                    // rows 0..511
    float *halfB = gu_f + 512 * 2048;       // rows 512..1023

    // IQ2 raw -> f32
    snprintf(path, sizeof(path), "%s/L20_E%03d_gate.iq2xxs", RAW_DIR, E);
    size_t gsz = 512 * ggml_row_size(GGML_TYPE_IQ2_XXS, 2048);
    uint8_t *graw = malloc(gsz);
    read_full(path, graw, gsz);
    snprintf(path, sizeof(path), "%s/L20_E%03d_up.iq2xxs", RAW_DIR, E);
    uint8_t *uraw = malloc(gsz);
    read_full(path, uraw, gsz);
    snprintf(path, sizeof(path), "%s/L20_E%03d_down.iq2s", RAW_DIR, E);
    size_t dsz = 2048 * ggml_row_size(GGML_TYPE_IQ2_S, 512);
    uint8_t *draw = malloc(dsz);
    read_full(path, draw, dsz);
    float *g_iq2 = malloc(512 * 2048 * 4);
    float *u_iq2 = malloc(512 * 2048 * 4);
    float *d_iq2 = malloc(2048 * 512 * 4);
    size_t gwr = ggml_row_size(GGML_TYPE_IQ2_XXS, 2048);
    size_t dwr = ggml_row_size(GGML_TYPE_IQ2_S, 512);
    for (int r = 0; r < 512; r++) {
        dequantize_row_iq2_xxs((const void *)(graw + r * gwr), g_iq2 + r * 2048, 2048);
        dequantize_row_iq2_xxs((const void *)(uraw + r * gwr), u_iq2 + r * 2048, 2048);
    }
    for (int r = 0; r < 2048; r++)
        dequantize_row_iq2_s((const void *)(draw + r * dwr), d_iq2 + r * 512, 512);

    // Establish gate_up split: try both half assignments, keep min-error.
    // (Unsloth UD quant came from this bf16; correct pairing has ~2% err,
    // wrong pairing ~140%.)
    printf("== E%d split identification ==\n", E);
    stats("halfA-vs-iq2gate", halfA, g_iq2, 512 * 2048);
    stats("halfA-vs-iq2up  ", halfA, u_iq2, 512 * 2048);
    stats("halfB-vs-iq2gate", halfB, g_iq2, 512 * 2048);
    stats("halfB-vs-iq2up  ", halfB, u_iq2, 512 * 2048);

    // Decide mapping by min relL2 (recompute cheaply via stats above is
    // visual; here pick programmatically with a coarse norm check).
    double eAG = 0, eAU = 0, eBG = 0, eBU = 0, nA = 0, nB = 0;
    for (int i = 0; i < 512 * 2048; i++) {
        double a = halfA[i], b = halfB[i];
        nA += a * a; nB += b * b;
        double d;
        d = a - g_iq2[i]; eAG += d * d;
        d = a - u_iq2[i]; eAU += d * d;
        d = b - g_iq2[i]; eBG += d * d;
        d = b - u_iq2[i]; eBU += d * d;
    }
    float *src_gate, *src_up;
    if (eAG / nA + eBU / nB < eAU / nA + eBG / nB) {
        src_gate = halfA; src_up = halfB;
        printf("mapping: halfA=gate halfB=up\n");
    } else {
        src_gate = halfB; src_up = halfA;
        printf("mapping: halfA=up halfB=gate\n");
    }
    float *src_down = dn_f;

    // Requantize-from-bf16 variants + double-quant variants.
    struct { const char *nm; enum ggml_type t; } fmts[] = {
        {"q2k", GGML_TYPE_Q2_K}, {"q40", GGML_TYPE_Q4_0}, {"q4k", GGML_TYPE_Q4_K},
    };
    float *x2048 = malloc(2048 * 4), *x512 = malloc(512 * 4);
    fill_randn(x2048, 2048); fill_randn(x512, 512);
    float *y_src = malloc(2048 * 4), *y_q = malloc(2048 * 4);

    struct { const char *pn; float *src, *iq2; int rows, K; } projs[] = {
        {"gate", src_gate, g_iq2, 512, 2048},
        {"up", src_up, u_iq2, 512, 2048},
        {"down", src_down, d_iq2, 2048, 512},
    };
    for (int p = 0; p < 3; p++) {
        const char *pn = projs[p].pn;
        float *src = projs[p].src, *iq2 = projs[p].iq2;
        int rows = projs[p].rows, K = projs[p].K;
        size_t n = (size_t)rows * K;
        printf("== E%d %s (rows=%d K=%d) weight-level vs bf16 ==\n", E, pn, rows, K);
        char tag[64];
        snprintf(tag, sizeof(tag), "%s.iq2", pn);
        stats(tag, iq2, src, n);
        for (int f = 0; f < 3; f++) {
            enum ggml_type t = fmts[f].t;
            size_t wr = ggml_row_size(t, K);
            uint8_t *q = malloc((size_t)rows * wr);
            float *dq = malloc(n * 4);
            float *dq2 = malloc(n * 4);
            for (int r = 0; r < rows; r++) {
                if (t == GGML_TYPE_Q2_K) {
                    quantize_row_q2_K(src + (size_t)r * K, q + (size_t)r * wr, K);
                    dequantize_row_q2_K((const void *)(q + (size_t)r * wr), dq + (size_t)r * K, K);
                    quantize_row_q2_K(iq2 + (size_t)r * K, q + (size_t)r * wr, K);
                    dequantize_row_q2_K((const void *)(q + (size_t)r * wr), dq2 + (size_t)r * K, K);
                } else if (t == GGML_TYPE_Q4_0) {
                    quantize_row_q4_0(src + (size_t)r * K, q + (size_t)r * wr, K);
                    dequantize_row_q4_0((const void *)(q + (size_t)r * wr), dq + (size_t)r * K, K);
                    quantize_row_q4_0(iq2 + (size_t)r * K, q + (size_t)r * wr, K);
                    dequantize_row_q4_0((const void *)(q + (size_t)r * wr), dq2 + (size_t)r * K, K);
                } else {
                    quantize_row_q4_K(src + (size_t)r * K, q + (size_t)r * wr, K);
                    dequantize_row_q4_K((const void *)(q + (size_t)r * wr), dq + (size_t)r * K, K);
                    quantize_row_q4_K(iq2 + (size_t)r * K, q + (size_t)r * wr, K);
                    dequantize_row_q4_K((const void *)(q + (size_t)r * wr), dq2 + (size_t)r * K, K);
                }
            }
            snprintf(tag, sizeof(tag), "%s.%s-from-bf16", pn, fmts[f].nm);
            stats(tag, dq, src, n);
            snprintf(tag, sizeof(tag), "%s.%s-from-iq2dq", pn, fmts[f].nm);
            stats(tag, dq2, src, n);
            // GEMV-output-level (x RMS 1.0)
            float *x = (K == 2048) ? x2048 : x512;
            gemv(src, x, y_src, rows, K);
            gemv(iq2, x, y_q, rows, K);
            snprintf(tag, sizeof(tag), "%s.y-iq2", pn);
            stats(tag, y_q, y_src, rows);
            gemv(dq, x, y_q, rows, K);
            snprintf(tag, sizeof(tag), "%s.y-%s", pn, fmts[f].nm);
            stats(tag, y_q, y_src, rows);
            free(q); free(dq); free(dq2);
        }
    }
    return 0;
}
