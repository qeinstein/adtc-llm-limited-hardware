// llc_pollute: does the streaming expert-weight traffic evict useful state
// from caches, and can non-temporal hints preserve it?
//
// Method: a "hot" buffer (activation-like state) of H bytes isChecksummed by
// an AVX2 reduction (timed). Between two timed reductions we run:
//   arm A (control):   no stream  -> hot re-access time = unpolluted baseline
//   arm B (stream):    32 MB expert-pool traversal via the REAL
//                      ggml_vec_dot_iq2_xxs_q8_K kernel (normal loads)
//   arm C (stream+nta): same traversal but with _MM_HINT_NTA prefetch ahead
//                      of the kernel's reads (tests whether NTA preserves hot)
//   arm D (memcpy+nta): plain byte-stream with NTA prefetch (mechanism check:
//                      does NTA demonstrably preserve hot state at all here?)
// H in {32K (L1d), 512K (L2), 4M (L3 6MB)}.
// Reports: hot time per arm (median of trials), stream time, slowdown vs A.
//
// Usage: llc_pollute [trials=30]
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <pmmintrin.h>

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
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

#define POOL (32u << 20)
#define GATEB 270336
#define ROWB_GU 528
#define DGATE 512
#define DIN 2048

static uint8_t *pool;
static size_t ncp;
static uint8_t *xq;
static float *out512;

// AVX2 reduction over hot buffer (timed victim)
static double hot_pass(float * h, size_t nf, volatile float * sink) {
    double s = now_ns();
    __m256 acc = _mm256_setzero_ps();
    for (size_t i = 0; i < nf; i += 8) {
        __m256 x = _mm256_loadu_ps(h + i);
        acc = _mm256_fmadd_ps(x, x, acc);
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    *sink = _mm_cvtss_f32(lo);
    return now_ns() - s;
}

// full-pool expert traversal with real kernel; nta Dist>0 => NTA-prefetch ahead
static double stream_pass(int nta_dist_rows, volatile float * sink) {
    double s = now_ns();
    float acc = 0;
    for (size_t c = 0; c < ncp; c++) {
        const uint8_t * w = pool + c * GATEB;
        for (int r = 0; r < DGATE; r++) {
            if (nta_dist_rows > 0 && r + nta_dist_rows < DGATE)
                _mm_prefetch((const char *)(w + (r + nta_dist_rows) * ROWB_GU), _MM_HINT_NTA);
            float v;
            ggml_vec_dot_iq2_xxs_q8_K(DIN, &v, 0, w + r * ROWB_GU, 0, xq, 0, 1);
            acc += v;
        }
    }
    *sink += acc;
    return now_ns() - s;
}

// plain streaming read with/without NTA (mechanism check)
static double memcpy_pass(int nta, volatile uint64_t * sink) {
    double s = now_ns();
    uint64_t acc = 0;
    const uint64_t * p = (const uint64_t *)pool;
    size_t n64 = POOL / 8;
    for (size_t i = 0; i < n64; i += 8) {
        if (nta) _mm_prefetch((const char *)(p + i + 64), _MM_HINT_NTA);
        acc += p[i] + p[i + 1] + p[i + 2] + p[i + 3] + p[i + 4] + p[i + 5] + p[i + 6] + p[i + 7];
    }
    *sink += acc;
    return now_ns() - s;
}

int main(int argc, char ** argv) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    int trials = (argc >= 2) ? atoi(argv[1]) : 30;

    FILE * f = fopen("/tmp/agent1_raw/L00_E000_gate.iq2xxs", "rb");
    static uint8_t gate[GATEB];
    if (!f || fread(gate, 1, GATEB, f) != GATEB) { fprintf(stderr, "weight load failed\n"); return 1; }
    fclose(f);
    pool = aligned_alloc(64, POOL);
    ncp = POOL / GATEB;
    for (size_t c = 0; c < ncp; c++) memcpy(pool + c * GATEB, gate, GATEB);

    float * x = aligned_alloc(64, DIN * sizeof(float));
    uint64_t rng = 7;
    for (int i = 0; i < DIN; i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        x[i] = (float)(((rng >> 33) / 4294967296.0) - 0.5);
    }
    xq = aligned_alloc(64, 8 * 292);
    quantize_row_q8_K(x, xq, DIN);
    out512 = aligned_alloc(64, DGATE * sizeof(float));

    size_t hots[] = {32 << 10, 512 << 10, 4 << 20};
    volatile float sinkf = 0;
    volatile uint64_t sink64 = 0;
    printf("hot_bytes,arm,med_ns,p10_ns,p90_ns,stream_med_ns\n");
    for (int h = 0; h < 3; h++) {
        size_t HB = hots[h], nf = HB / sizeof(float);
        float * hot = aligned_alloc(64, HB);
        for (size_t i = 0; i < nf; i++) hot[i] = (float)i * 0.001f;
        // arms: 0=control(none), 1=kernel stream, 2=kernel+NTA(4 rows ahead),
        // 3=kernel+NTA(16), 4=memcpy, 5=memcpy+NTA
        for (int arm = 0; arm < 6; arm++) {
            double * tm = malloc(trials * sizeof(double));
            double * st = malloc(trials * sizeof(double));
            hot_pass(hot, nf, &sinkf); // warm
            for (int t = 0; t < trials; t++) {
                hot_pass(hot, nf, &sinkf); // (re)establish hot
                double s = 0;
                if (arm == 1) s = stream_pass(0, &sinkf);
                else if (arm == 2) s = stream_pass(4, &sinkf);
                else if (arm == 3) s = stream_pass(16, &sinkf);
                else if (arm == 4) s = memcpy_pass(0, &sink64);
                else if (arm == 5) s = memcpy_pass(1, &sink64);
                st[t] = s;
                tm[t] = hot_pass(hot, nf, &sinkf);
            }
            qsort(tm, trials, sizeof(double), cmp_dbl);
            qsort(st, trials, sizeof(double), cmp_dbl);
            const char * nm = (arm == 0) ? "control" : (arm == 1) ? "kernel" : (arm == 2) ? "kernel+nta4"
                : (arm == 3) ? "kernel+nta16" : (arm == 4) ? "memcpy" : "memcpy+nta";
            printf("%zu,%s,%.0f,%.0f,%.0f,%.0f\n", HB, nm, tm[trials / 2], tm[trials / 10],
                tm[(trials * 9) / 10], st[trials / 2]);
            fflush(stdout);
            free(tm); free(st);
        }
        free(hot);
    }
    fprintf(stderr, "sink=%f %lu\n", sinkf, (unsigned long)sink64);
    return 0;
}
