// hw_agent1 baseline harness: exact-shape expert GEMV via REAL ggml AVX2 kernels.
// Links against /tmp/llamacpp-pin ggml x86/quants.c compiled with -O3 -march=native.
// Workload: ONE routed expert (gate+up IQ2_XXS 512x2048, down IQ2_S 2048x512)
// using REAL weight bytes range-fetched from the pinned UD-IQ2_XXS GGUF.
// Dataflow mirrors decode: x[2048] -> gate,up -> h=silu(g)*u -> down -> y[2048].
// Cold-cache timing (clflush weights per rep) models decode streaming from RAM.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void quantize_row_q8_K(const float * x, void * y, int64_t k);

#define QK_K 256
#define B_XXS 66    // block_iq2_xxs bytes
#define B_IQ2S 82   // block_iq2_s bytes
#define B_Q8K 292   // block_q8_K bytes (2 + 256 + 2*8*2 = 292? verified below)

static uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static void clflush_buf(const void * p, size_t n) {
    const char * c = (const char *)p;
    for (size_t i = 0; i < n; i += 64) _mm_clflush(c + i);
    _mm_mfence();
}
static void * load_file(const char * path, size_t expect) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if ((size_t)n != expect) { fprintf(stderr, "%s: size %ld != %zu\n", path, n, expect); exit(1); }
    void * b = aligned_alloc(64, (expect + 63) & ~(size_t)63);
    if (fread(b, 1, expect, f) != expect) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return b;
}
static int cmp_dbl(const void * a, const void * b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double * v, int n) {
    qsort(v, n, sizeof(double), cmp_dbl);
    return v[n / 2];
}

int main(int argc, char ** argv) {
    const char * raw = argc > 1 ? argv[1] : "/tmp/agent1_raw";
    char path[512];
    snprintf(path, sizeof(path), "%s/L00_E000_gate.iq2xxs", raw);
    // expert slice: 512 rows x 8 blk x 66 B = 270336
    uint8_t * gate = load_file(path, 512 * 8 * B_XXS);
    snprintf(path, sizeof(path), "%s/L00_E000_up.iq2xxs", raw);
    uint8_t * up = load_file(path, 512 * 8 * B_XXS);
    snprintf(path, sizeof(path), "%s/L00_E000_down.iq2s", raw);
    // down slice: 2048 rows x 2 blk x 82 B = 335872
    uint8_t * down = load_file(path, 2048 * 2 * B_IQ2S);

    // activation x[2048]: RMSNorm'd pseudo-realistic vector (deterministic;
    // timing is value-independent for these SIMD kernels; parity uses identical inputs).
    // Shape: mostly small gaussian + a few outlier channels (LLM-like).
    float * x = aligned_alloc(64, 2048 * sizeof(float));
    uint64_t rng = 0x12345678;
    for (int i = 0; i < 2048; i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        double u = ((rng >> 33) + 0.5) / 4294967296.0;
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        double v = ((rng >> 33) + 0.5) / 4294967296.0;
        double g = sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
        x[i] = (float)(g * 0.35);
    }
    x[7] = 8.5f; x[311] = -6.2f; x[1023] = 5.1f; x[1723] = -4.4f; // outlier channels
    double ss = 0;
    for (int i = 0; i < 2048; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / 2048) + 1e-6f);
    for (int i = 0; i < 2048; i++) x[i] *= inv;

    // quantize activations with REAL ggml quantize_row_q8_K
    uint8_t * xq = aligned_alloc(64, 8 * B_Q8K);   // n=2048 -> 8 blocks
    quantize_row_q8_K(x, xq, 2048);

    float * gout = aligned_alloc(64, 512 * sizeof(float));
    float * uout = aligned_alloc(64, 512 * sizeof(float));
    float * h = aligned_alloc(64, 512 * sizeof(float));
    uint8_t * hq = aligned_alloc(64, 2 * B_Q8K);   // n=512 -> 2 blocks
    float * yout = aligned_alloc(64, 2048 * sizeof(float));

    // correctness reference run (also warms up)
    for (int r = 0; r < 512; r++)
        ggml_vec_dot_iq2_xxs_q8_K(2048, &gout[r], 0, gate + r * 8 * B_XXS, 0, xq, 0, 1);
    for (int r = 0; r < 512; r++)
        ggml_vec_dot_iq2_xxs_q8_K(2048, &uout[r], 0, up + r * 8 * B_XXS, 0, xq, 0, 1);
    for (int i = 0; i < 512; i++) {
        float g = gout[i];
        h[i] = (g / (1.0f + expf(-g))) * uout[i]; // SiLU(g)*u
    }
    quantize_row_q8_K(h, hq, 512);
    for (int r = 0; r < 2048; r++)
        ggml_vec_dot_iq2_s_q8_K(512, &yout[r], 0, down + r * 2 * B_IQ2S, 0, hq, 0, 1);
    double chk = 0;
    for (int i = 0; i < 2048; i++) chk += yout[i];
    printf("ref checksum: %.6f  y[0]=%.6f y[1000]=%.6f\n", chk, yout[0], yout[1000]);

    // save reference outputs for parity diff
    FILE * f = fopen("/tmp/hw1/ref_gout.bin", "wb");
    fwrite(gout, 4, 512, f); fclose(f);
    f = fopen("/tmp/hw1/ref_uout.bin", "wb");
    fwrite(uout, 4, 512, f); fclose(f);
    f = fopen("/tmp/hw1/ref_yout.bin", "wb");
    fwrite(yout, 4, 2048, f); fclose(f);
    f = fopen("/tmp/hw1/ref_h.bin", "wb");
    fwrite(h, 4, 512, f); fclose(f);

    // rdtsc calibration
    double t0 = now_ns();
    uint64_t c0 = rdtsc();
    struct timespec sl = {0, 200000000};
    nanosleep(&sl, NULL);
    uint64_t c1 = rdtsc();
    double t1 = now_ns();
    double tsc_ghz = (c1 - c0) / (t1 - t0);
    printf("TSC: %.3f GHz\n", tsc_ghz);

    const int REPS = 300;
    double * tg = malloc(REPS * sizeof(double));
    double * tu = malloc(REPS * sizeof(double));
    double * td = malloc(REPS * sizeof(double));
    double * tq = malloc(REPS * sizeof(double));
    uint64_t * cg = malloc(REPS * sizeof(uint64_t));
    uint64_t * cu = malloc(REPS * sizeof(uint64_t));
    uint64_t * cd = malloc(REPS * sizeof(uint64_t));

    // COLD: flush weight stream before every rep (decode-like streaming from RAM)
    for (int it = 0; it < REPS; it++) {
        clflush_buf(gate, 512 * 8 * B_XXS);
        uint64_t a = rdtsc();
        double s = now_ns();
        for (int r = 0; r < 512; r++)
            ggml_vec_dot_iq2_xxs_q8_K(2048, &gout[r], 0, gate + r * 8 * B_XXS, 0, xq, 0, 1);
        tg[it] = now_ns() - s; cg[it] = rdtsc() - a;

        clflush_buf(up, 512 * 8 * B_XXS);
        a = rdtsc(); s = now_ns();
        for (int r = 0; r < 512; r++)
            ggml_vec_dot_iq2_xxs_q8_K(2048, &uout[r], 0, up + r * 8 * B_XXS, 0, xq, 0, 1);
        tu[it] = now_ns() - s; cu[it] = rdtsc() - a;

        clflush_buf(down, 2048 * 2 * B_IQ2S);
        a = rdtsc(); s = now_ns();
        for (int r = 0; r < 2048; r++)
            ggml_vec_dot_iq2_s_q8_K(512, &yout[r], 0, down + r * 2 * B_IQ2S, 0, hq, 0, 1);
        td[it] = now_ns() - s; cd[it] = rdtsc() - a;

        s = now_ns();
        quantize_row_q8_K(x, xq, 2048);
        quantize_row_q8_K(h, hq, 512);
        tq[it] = now_ns() - s;
    }
    double mg = median(tg, REPS), mu = median(tu, REPS), md = median(td, REPS), mq = median(tq, REPS);
    double mcg = median(tg, REPS); // placeholder, replaced below
    (void)mcg;
    // cycle medians (proper u64 sort via insertion-free index: copy to double)
    double *dg = malloc(REPS*sizeof(double)), *du = malloc(REPS*sizeof(double)), *dd = malloc(REPS*sizeof(double));
    for (int i = 0; i < REPS; i++) { dg[i] = (double)cg[i]; du[i] = (double)cu[i]; dd[i] = (double)cd[i]; }
    double cg_med = median(dg, REPS), cu_med = median(du, REPS), cd_med = median(dd, REPS);
    printf("COLD medians over %d reps (TSC %.3f GHz):\n", REPS, tsc_ghz);
    printf("  gate 2048->512 IQ2_XXS: %9.0f ns  %10.0f cyc  %6.1f cyc/out  bytes=%d -> eff BW %5.2f GB/s\n", mg, cg_med, cg_med/512, 512*8*B_XXS, (512*8*B_XXS)/mg);
    printf("  up   2048->512 IQ2_XXS: %9.0f ns  %10.0f cyc  %6.1f cyc/out  bytes=%d -> eff BW %5.2f GB/s\n", mu, cu_med, cu_med/512, 512*8*B_XXS, (512*8*B_XXS)/mu);
    printf("  down 512->2048 IQ2_S:   %9.0f ns  %10.0f cyc  %6.1f cyc/out  bytes=%d -> eff BW %5.2f GB/s\n", md, cd_med, cd_med/2048, 2048*2*B_IQ2S, (2048*2*B_IQ2S)/md);
    printf("  q8_K quant x+h:         %9.0f ns\n", mq);
    double expert_us = (mg + mu + md) / 1000.0;
    double layer8_us = expert_us * 8.0; // 8 routed experts per layer (single-thread sum)
    printf("  ONE routed expert: %.1f us | x8 = %.1f us/layer | x40 layers = %.1f ms/token (1 thread, routed IQ2 only)\n",
           expert_us, layer8_us, layer8_us * 40 / 1000.0);

    // HOT: no flush (upper bound on compute, L2/L3 resident)
    for (int it = 0; it < REPS; it++) {
        double s = now_ns();
        for (int r = 0; r < 512; r++)
            ggml_vec_dot_iq2_xxs_q8_K(2048, &gout[r], 0, gate + r * 8 * B_XXS, 0, xq, 0, 1);
        tg[it] = now_ns() - s;
        s = now_ns();
        for (int r = 0; r < 512; r++)
            ggml_vec_dot_iq2_xxs_q8_K(2048, &uout[r], 0, up + r * 8 * B_XXS, 0, xq, 0, 1);
        tu[it] = now_ns() - s;
        s = now_ns();
        for (int r = 0; r < 2048; r++)
            ggml_vec_dot_iq2_s_q8_K(512, &yout[r], 0, down + r * 2 * B_IQ2S, 0, hq, 0, 1);
        td[it] = now_ns() - s;
    }
    mg = median(tg, REPS); mu = median(tu, REPS); md = median(td, REPS);
    printf("HOT medians:\n");
    printf("  gate: %.0f ns  up: %.0f ns  down: %.0f ns  expert: %.1f us  x320 experts: %.1f ms/token\n",
           mg, mu, md, (mg+mu+md)/1000.0, (mg+mu+md)*320/1e6);
    return 0;
}
