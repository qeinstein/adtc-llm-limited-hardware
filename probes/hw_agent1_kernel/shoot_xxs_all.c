// Head-to-head shootout: baseline vs spec variants (parity + interleaved timing).
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
void spec_xxs_v2_gemv(float * y, const void * w, const void * act, int rows, int nb);
void spec_xxs_v3_gemv(float * y, const void * w, const void * act, int rows, int nb);
void spec_xxs_v2b_gemv(float * y, const void * w, const void * act, int rows, int nb);
void spec_xxs_v3b_gemv(float * y, const void * w, const void * act, int rows, int nb);

#define B_XXS 66
#define ROWS 512
#define NB 8
#define ROWB (NB * B_XXS)
#define EXPB (ROWS * ROWB)

typedef void (*gemv_fn)(float *, const void *, const void *, int, int);
static uint8_t * g_pool;
static size_t g_ncopy;
static uint8_t * g_xq;
static float * g_yb;

static void base_gemv(float * y, const void * w, const void * act, int rows, int nb) {
    (void)nb;
    for (int r = 0; r < rows; r++)
        ggml_vec_dot_iq2_xxs_q8_K(2048, &y[r], 0, (const char *)w + r * ROWB, 0, act, 0, 1);
}

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
    const size_t POOL = 64 << 20;
    g_pool = aligned_alloc(64, POOL);
    g_ncopy = POOL / EXPB;
    for (size_t c = 0; c < g_ncopy; c++) memcpy(g_pool + c * EXPB, exp0, EXPB);

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
    g_xq = aligned_alloc(64, 8 * 292);
    quantize_row_q8_K(x, g_xq, 2048);

    struct { const char * name; gemv_fn fn; } vs[] = {
        {"baseline", base_gemv},
        {"v1-2row ", base_gemv}, // placeholder replaced below (v1 kept for reference)
        {"v2-gfni ", NULL},
        {"v3-gather", NULL},
        {"v2b-nogfni", NULL},
        {"v3b-gathbl", NULL},
    };
    vs[1].fn = spec_xxs_v1_gemv;
    vs[2].fn = spec_xxs_v2_gemv;
    vs[3].fn = spec_xxs_v3_gemv;
    vs[4].fn = spec_xxs_v2b_gemv;
    vs[5].fn = spec_xxs_v3b_gemv;
    const int NV = 6;

    g_yb = aligned_alloc(64, ROWS * sizeof(float));
    float * yv = aligned_alloc(64, ROWS * sizeof(float));
    base_gemv(g_yb, exp0, g_xq, ROWS, NB);

    for (int v = 1; v < NV; v++) {
        vs[v].fn(yv, exp0, g_xq, ROWS, NB);
        double maxd = 0;
        long nbit = 0;
        for (int r = 0; r < ROWS; r++) {
            double d = fabs((double)g_yb[r] - (double)yv[r]);
            if (d > maxd) maxd = d;
            if (memcmp(&g_yb[r], &yv[r], 4) == 0) nbit++;
        }
        printf("%s parity: max_abs_diff=%.3e bit_exact=%ld/%d\n", vs[v].name, maxd, nbit, ROWS);
    }

    const int TRIALS = 48;
    double t[6][48];
    for (int rep = 0; rep < TRIALS; rep++) {
        for (int v = 0; v < NV; v++) {
            uint8_t * w = g_pool + ((rep * NV + v) % g_ncopy) * EXPB;
            double s = now_ns();
            vs[v].fn(yv, w, g_xq, ROWS, NB);
            t[v][rep] = now_ns() - s;
        }
    }
    double med[6], mn[6];
    for (int v = 0; v < NV; v++) {
        qsort(t[v], TRIALS, sizeof(double), cmp_dbl);
        med[v] = t[v][TRIALS / 2];
        mn[v] = t[v][0];
    }
    for (int v = 0; v < NV; v++)
        printf("%s: min=%8.0fns med=%8.0fns (%6.1f ns/row) speedup=%.3f\n",
               vs[v].name, mn[v], med[v], med[v] / ROWS, med[0] / med[v]);
    return 0;
}
