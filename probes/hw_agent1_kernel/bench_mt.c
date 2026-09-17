// MT scaling with persistent threads (ggml-style) + core-freq calibration.
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>
#include <pmmintrin.h>

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void quantize_row_q8_K(const float * x, void * y, int64_t k);

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static int cmp_dbl(const void * a, const void * b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// core-freq probe: dependent IMUL chain, 3c/iter on Gracemont (documented 3c latency)
static double probe_ghz(void) {
    const long IT = 2000000;
    uint64_t x = 0x123456789abcdefULL;
    double s = now_ns();
    for (long i = 0; i < IT; i++) { x = x * 0x9e3779b97f4a7c15ULL + (uint64_t)i; }
    double e = now_ns() - s;
    volatile uint64_t sink = x;
    (void)sink;
    // chain: imul(3c) + add(1c, dependent) ≈ 4c/iter
    return (IT * 4.0) / e;
}

#define MAXT 4
static pthread_barrier_t bar0, bar1;
static _Atomic int work_kind; // 0 gate, 1 down, -1 exit
static const uint8_t * volatile W;
static const uint8_t * XQ;
static float * Y;
static int ROWB, NROWS, NB_K;

static void do_rows(int ith, int nth) {
    int r0 = NROWS * ith / nth, r1 = NROWS * (ith + 1) / nth;
    const uint8_t * w = W;
    if (work_kind == 0)
        for (int r = r0; r < r1; r++)
            ggml_vec_dot_iq2_xxs_q8_K(2048, &Y[r], 0, w + r * ROWB, 0, XQ, 0, 1);
    else
        for (int r = r0; r < r1; r++)
            ggml_vec_dot_iq2_s_q8_K(512, &Y[r], 0, w + r * ROWB, 0, XQ, 0, 1);
}

static void * worker(void * v) {
    int ith = (int)(intptr_t)v;
    extern int g_nth;
    for (;;) {
        pthread_barrier_wait(&bar0);
        if (work_kind < 0) break;
        do_rows(ith, g_nth);
        pthread_barrier_wait(&bar1);
    }
    return NULL;
}
int g_nth = 1;

int main(void) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    printf("core freq probe: %.2f GHz\n", probe_ghz());

    FILE * f = fopen("/tmp/agent1_raw/L00_E000_gate.iq2xxs", "rb");
    static uint8_t gate[270336], up[270336], down[335872];
    if (fread(gate, 1, 270336, f) != 270336) return 1;
    fclose(f);
    f = fopen("/tmp/agent1_raw/L00_E000_down.iq2s", "rb");
    if (fread(down, 1, 335872, f) != 335872) return 1;
    fclose(f);

    const size_t POOL = 48 << 20;
    uint8_t * pg = aligned_alloc(64, POOL), *pd = aligned_alloc(64, POOL);
    size_t ng = POOL / 270336, nd = POOL / 335872;
    for (size_t c = 0; c < ng; c++) memcpy(pg + c * 270336, gate, 270336);
    for (size_t c = 0; c < nd; c++) memcpy(pd + c * 335872, down, 335872);

    float * x = aligned_alloc(64, 2048 * sizeof(float));
    uint64_t rng = 99;
    for (int i = 0; i < 2048; i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        x[i] = (float)(((rng >> 33) / 4294967296.0) - 0.5);
    }
    uint8_t * xq = aligned_alloc(64, 8 * 292);
    quantize_row_q8_K(x, xq, 2048);
    // h from quick gate pass
    float * gout = aligned_alloc(64, 512 * sizeof(float));
    float * h = aligned_alloc(64, 512 * sizeof(float));
    for (int r = 0; r < 512; r++)
        ggml_vec_dot_iq2_xxs_q8_K(2048, &gout[r], 0, gate + r * 528, 0, xq, 0, 1);
    for (int i = 0; i < 512; i++) h[i] = gout[i] / (1.0f + expf(-gout[i]));
    uint8_t * hq = aligned_alloc(64, 2 * 292);
    quantize_row_q8_K(h, hq, 512);
    float * yb = aligned_alloc(64, 2048 * sizeof(float));

    pthread_barrier_init(&bar0, NULL, MAXT);
    pthread_barrier_init(&bar1, NULL, MAXT);
    pthread_t th[MAXT];
    for (int i = 1; i < MAXT; i++) pthread_create(&th[i], NULL, worker, (void *)(intptr_t)i);

    for (int nth = 1; nth <= 4; nth++) {
        g_nth = nth;
        double tm[30];
        for (int t = 0; t < 30; t++) {
            double s = now_ns();
            // gate GEMV
            W = pg + (t % ng) * 270336; XQ = xq; Y = gout; ROWB = 528; NROWS = 512;
            work_kind = 0;
            pthread_barrier_wait(&bar0);
            if (nth == 1) do_rows(0, 1);
            else do_rows(0, nth);
            pthread_barrier_wait(&bar1);
            // down GEMV
            W = pd + (t % nd) * 335872; XQ = hq; Y = yb; ROWB = 164; NROWS = 2048;
            work_kind = 1;
            pthread_barrier_wait(&bar0);
            do_rows(0, nth);
            pthread_barrier_wait(&bar1);
            tm[t] = now_ns() - s;
        }
        // NOTE: barrier has MAXT parties but only nth work; idle threads still
        // wait at barriers (they check work_kind and skip rows when ith>=nth? NO -
        // worker() always runs do_rows with g_nth... r0==r1 for ith>=nth gives empty range. OK.)
        qsort(tm, 30, sizeof(double), cmp_dbl);
        printf("nth=%d gate+down med=%.0f ns (speedup vs ST=%.2f)\n", nth, tm[15], 0.0);
        // store ST ref on first iter
        static double st = 0;
        if (nth == 1) st = tm[15];
        printf("   -> speedup=%.3f\n", st / tm[15]);
    }
    printf("core freq probe: %.2f GHz\n", probe_ghz());
    work_kind = -1;
    pthread_barrier_wait(&bar0);
    for (int i = 1; i < MAXT; i++) pthread_join(th[i], NULL);
    return 0;
}
