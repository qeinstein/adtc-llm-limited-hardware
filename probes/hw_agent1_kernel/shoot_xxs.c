// Head-to-head: ggml baseline vec_dot loop vs specialized GEMV kernels.
// Regime: weights stream from 64MB pool (DRAM, TLB-cold-ish like decode);
// activation hot. Interleaved A/B trials, FTZ/DAZ. Parity vs baseline first.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>
#include <pmmintrin.h>

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void quantize_row_q8_K(const float * x, void * y, int64_t k);
void spec_xxs_v1_gemv(float * y, const void * w, const void * act, int rows, int nb);

#define B_XXS 66
#define ROWS 512
#define NB 8
#define ROWB (NB * B_XXS)
#define EXPB (ROWS * ROWB)  // 270336

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static int cmp_dbl(const void * a, const void * b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(void) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    FILE * f = fopen("/tmp/agent1_raw/L00_E000_gate.iq2xxs", "rb");
    if (!f) { fprintf(stderr, "missing weights\n"); return 1; }
    static uint8_t exp0[EXPB];
    if (fread(exp0, 1, EXPB, f) != EXPB) { fprintf(stderr, "short\n"); return 1; }
    fclose(f);

    // 64MB streaming pool of expert copies (242 copies)
    const size_t POOL = 64 << 20;
    uint8_t * pool = aligned_alloc(64, POOL);
    size_t ncopy = POOL / EXPB;
    for (size_t c = 0; c < ncopy; c++) memcpy(pool + c * EXPB, exp0, EXPB);

    float * x = aligned_alloc(64, 2048 * sizeof(float));
    uint64_t rng = 0x12345678;
    for (int i = 0; i < 2048; i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        double u = ((rng >> 33) + 0.5) / 4294967296.0;
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        double v = ((rng >> 33) + 0.5) / 4294967296.0;
        x[i] = (float)(sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v) * 0.35);
    }
    x[7] = 8.5f; x[311] = -6.2f; x[1023] = 5.1f; x[1723] = -4.4f;
    double ss = 0;
    for (int i = 0; i < 2048; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / 2048) + 1e-6f);
    for (int i = 0; i < 2048; i++) x[i] *= inv;
    uint8_t * xq = aligned_alloc(64, 8 * 292);
    quantize_row_q8_K(x, xq, 2048);

    float * yb = aligned_alloc(64, ROWS * sizeof(float));
    float * yv = aligned_alloc(64, ROWS * sizeof(float));

    // ---- parity ----
    for (int r = 0; r < ROWS; r++)
        ggml_vec_dot_iq2_xxs_q8_K(2048, &yb[r], 0, exp0 + r * ROWB, 0, xq, 0, 1);
    spec_xxs_v1_gemv(yv, exp0, xq, ROWS, NB);
    double maxd = 0;
    long nbit = 0;
    for (int r = 0; r < ROWS; r++) {
        double d = fabs((double)yb[r] - (double)yv[r]);
        if (d > maxd) maxd = d;
        if (memcmp(&yb[r], &yv[r], 4) == 0) nbit++;
    }
    printf("v1 parity: max_abs_diff=%.3e bit_exact=%ld/%d\n", maxd, nbit, ROWS);

    // ---- timing: interleaved ----
    const int TRIALS = 40;
    double tb[40], tv[40];
    for (int t = 0; t < TRIALS; t++) {
        uint8_t * w = pool + (t % ncopy) * EXPB;
        double s = now_ns();
        for (int r = 0; r < ROWS; r++)
            ggml_vec_dot_iq2_xxs_q8_K(2048, &yb[r], 0, w + r * ROWB, 0, xq, 0, 1);
        tb[t] = now_ns() - s;
        s = now_ns();
        spec_xxs_v1_gemv(yv, w, xq, ROWS, NB);
        tv[t] = now_ns() - s;
    }
    qsort(tb, TRIALS, sizeof(double), cmp_dbl);
    qsort(tv, TRIALS, sizeof(double), cmp_dbl);
    printf("baseline gate-GEMV: min=%.0fns med=%.0fns (%.1f ns/row)\n", tb[0], tb[TRIALS/2], tb[TRIALS/2]/ROWS);
    printf("v1       gate-GEMV: min=%.0fns med=%.0fns (%.1f ns/row)  speedup=%.3f\n",
           tv[0], tv[TRIALS/2], tv[TRIALS/2]/ROWS, tb[TRIALS/2]/tv[TRIALS/2]);
    return 0;
}
