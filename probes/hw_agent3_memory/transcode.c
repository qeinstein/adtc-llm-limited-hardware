// transcode: transient execution-cache probe. Compact IQ2 backing ->
// transcode ONLY imminent experts (8-16) -> Q8_0 CPU-friendly rep ->
// GEMV via ggml_vec_dot_q8_0_q8_0. Measures: transcode ms vs GEMV ms
// (overlap feasibility), Q8_0 GEMV speedup vs IQ2 decode-GEMV, output
// maxdiff (parity proxy; token parity needs full model), cache bytes.
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void quantize_row_q8_K(const float * x, void * y, int64_t k);
void quantize_row_q8_0(const float * x, void * y, int64_t k);
void dequantize_row_iq2_xxs(const void * x, float * y, int64_t k);
void dequantize_row_iq2_s(const void * x, float * y, int64_t k);

/* link stubs for standalone ggml-quants.c.o (never called on our paths) */
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    (void)file; (void)line; (void)fmt; abort();
}
size_t ggml_row_size(int type, int64_t ne) { (void)type; (void)ne; abort(); }
const char * ggml_type_name(int type) { (void)type; abort(); }
size_t ggml_type_size(int type) { (void)type; abort(); }

#define DIN 2048
#define DFF 512
#define TOPK 8
#define GATEB 270336
#define DOWNB 335872
#define ROWB_GU 528
#define ROWB_DN 164
#define Q8ROW_GU (2048/32*34)  /* 2176 */
#define Q8ROW_DN (512/32*34)   /* 544 */
#define Q8GATE (512*Q8ROW_GU)  /* 1114112 */
#define Q8DOWN (2048*Q8ROW_DN) /* 1114112 */
#define Q8EXP (Q8GATE*2+Q8DOWN) /* 3342336 */

static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static int cmp_dbl(const void * a, const void * b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static void * load_file(const char * path, size_t want) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    void * p = malloc(want ? want : 1);
    size_t off = 0;
    while (off < want) {
        ssize_t r = read(fd, (char *)p + off, want - off);
        if (r <= 0) { close(fd); free(p); return NULL; }
        off += r;
    }
    close(fd);
    return p;
}

static uint8_t * W[TOPK][3];   /* IQ2 backing */
static uint8_t * Q8[TOPK];     /* transient exec rep, Q8EXP bytes each */
static float X[DIN];
static uint8_t XQK[DIN * 2], XQ0[DIN / 32 * 34 + 64];

static void transcode_one(int e) {
    static float fbuf[DIN];
    uint8_t * q = Q8[e];
    for (int r = 0; r < DFF; r++) {
        dequantize_row_iq2_xxs(W[e][0] + r * ROWB_GU, fbuf, DIN);
        quantize_row_q8_0(fbuf, q + r * Q8ROW_GU, DIN);
    }
    q += Q8GATE;
    for (int r = 0; r < DFF; r++) {
        dequantize_row_iq2_xxs(W[e][1] + r * ROWB_GU, fbuf, DIN);
        quantize_row_q8_0(fbuf, q + r * Q8ROW_GU, DIN);
    }
    q += Q8GATE;
    for (int r = 0; r < DIN; r++) {
        dequantize_row_iq2_s(W[e][2] + r * ROWB_DN, fbuf, DFF);
        quantize_row_q8_0(fbuf, q + r * Q8ROW_DN, DFF);
    }
}

static float gout[DFF], uout[DFF], h[DFF], dout[DIN];
static uint8_t hqK[DFF * 2], hq0[DFF / 32 * 34 + 64];

static void expert_iq2(int e, float * y) {
    for (int r = 0; r < DFF; r++)
        ggml_vec_dot_iq2_xxs_q8_K(DIN, &gout[r], 0, W[e][0] + r * ROWB_GU, 0, XQK, 0, 1);
    for (int r = 0; r < DFF; r++)
        ggml_vec_dot_iq2_xxs_q8_K(DIN, &uout[r], 0, W[e][1] + r * ROWB_GU, 0, XQK, 0, 1);
    for (int r = 0; r < DFF; r++) { float g = gout[r]; h[r] = (g / (1.0f + expf(-g))) * uout[r]; }
    quantize_row_q8_K(h, hqK, DFF);
    for (int r = 0; r < DIN; r++)
        ggml_vec_dot_iq2_s_q8_K(DFF, &dout[r], 0, W[e][2] + r * ROWB_DN, 0, hqK, 0, 1);
    for (int r = 0; r < DIN; r++) y[r] += dout[r] / TOPK;
}

static void expert_q8(int e, float * y) {
    uint8_t * q = Q8[e];
    for (int r = 0; r < DFF; r++)
        ggml_vec_dot_q8_0_q8_0(DIN, &gout[r], 0, q + r * Q8ROW_GU, 0, XQ0, 0, 1);
    for (int r = 0; r < DFF; r++)
        ggml_vec_dot_q8_0_q8_0(DIN, &uout[r], 0, q + Q8GATE + r * Q8ROW_GU, 0, XQ0, 0, 1);
    for (int r = 0; r < DFF; r++) { float g = gout[r]; h[r] = (g / (1.0f + expf(-g))) * uout[r]; }
    quantize_row_q8_0(h, hq0, DFF);
    for (int r = 0; r < DIN; r++)
        ggml_vec_dot_q8_0_q8_0(DFF, &dout[r], 0, q + 2 * Q8GATE + r * Q8ROW_DN, 0, hq0, 0, 1);
    for (int r = 0; r < DIN; r++) y[r] += dout[r] / TOPK;
}

struct bg { int e0, e1; double ms; };
static void * bg_worker(void * v) {
    struct bg * b = v;
    double t0 = now_ns();
    for (int e = b->e0; e < b->e1; e++) transcode_one(e);
    b->ms = (now_ns() - t0) / 1e6;
    return NULL;
}

int main(void) {
    char path[512];
    int ids[TOPK] = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int k = 0; k < TOPK; k++) {
        snprintf(path, sizeof path, "/tmp/agent1_raw/L00_E%03d_gate.iq2xxs", ids[k]);
        W[k][0] = load_file(path, GATEB);
        snprintf(path, sizeof path, "/tmp/agent1_raw/L00_E%03d_up.iq2xxs", ids[k]);
        W[k][1] = load_file(path, GATEB);
        snprintf(path, sizeof path, "/tmp/agent1_raw/L00_E%03d_down.iq2s", ids[k]);
        W[k][2] = load_file(path, DOWNB);
        if (!W[k][0] || !W[k][1] || !W[k][2]) { fprintf(stderr, "load expert %d failed\n", ids[k]); return 1; }
        Q8[k] = malloc(Q8EXP);
    }
    for (int i = 0; i < DIN; i++) X[i] = sinf(i * 0.37f) * 0.6f + cosf(i * 0.11f) * 0.4f;
    quantize_row_q8_K(X, XQK, DIN);
    quantize_row_q8_0(X, XQ0, DIN);
    printf("q8expert_bytes=%d q8cache_8exp_MB=%.2f q8cache_16exp_MB=%.2f\n",
        Q8EXP, 8.0 * Q8EXP / (1 << 20), 16.0 * Q8EXP / (1 << 20));

    /* warmup */
    transcode_one(0);
    float yref[DIN], yq8[DIN];
    memset(yref, 0, sizeof yref);
    for (int k = 0; k < TOPK; k++) expert_iq2(k, yref);

#define TRIALS 15
    double t_tc[TRIALS], t_iq2[TRIALS], t_q8[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        double t0 = now_ns();
        for (int k = 0; k < TOPK; k++) transcode_one(k);
        t_tc[t] = now_ns() - t0;
        memset(yref, 0, sizeof yref);
        t0 = now_ns();
        for (int k = 0; k < TOPK; k++) expert_iq2(k, yref);
        t_iq2[t] = now_ns() - t0;
        memset(yq8, 0, sizeof yq8);
        t0 = now_ns();
        for (int k = 0; k < TOPK; k++) expert_q8(k, yq8);
        t_q8[t] = now_ns() - t0;
    }
    qsort(t_tc, TRIALS, sizeof(double), cmp_dbl);
    qsort(t_iq2, TRIALS, sizeof(double), cmp_dbl);
    qsort(t_q8, TRIALS, sizeof(double), cmp_dbl);
    printf("transcode8_ms_med=%.3f iq2_gemv8_ms_med=%.3f q8_gemv8_ms_med=%.3f\n",
        t_tc[TRIALS/2]/1e6, t_iq2[TRIALS/2]/1e6, t_q8[TRIALS/2]/1e6);
    printf("q8gemv_vs_iq2gemv=%.3fx transcode_vs_iq2gemv=%.2fx\n",
        t_iq2[TRIALS/2]/t_q8[TRIALS/2], t_tc[TRIALS/2]/t_iq2[TRIALS/2]);

    double maxad = 0, maxref = 0;
    for (int i = 0; i < DIN; i++) {
        double ad = fabs(yref[i] - yq8[i]);
        if (ad > maxad) maxad = ad;
        double d = fabs(yref[i]) > 1e-6 ? ad / fabs(yref[i]) : 0;
        if (d > maxref) maxref = d;
    }
    /* checksum both */
    uint64_t hr = 1469598103934665603ULL, hq = 1469598103934665603ULL;
    for (size_t i = 0; i < sizeof yref; i++) {
        hr ^= ((uint8_t*)yref)[i]; hr *= 1099511628211ULL;
        hq ^= ((uint8_t*)yq8)[i]; hq *= 1099511628211ULL;
    }
    printf("maxabsdiff=%.6f maxreldiff=%.6f ck_iq2=%016llx ck_q8=%016llx bitwise=%s\n",
        maxad, maxref, (unsigned long long)hr, (unsigned long long)hq,
        hr == hq ? "IDENTICAL" : "DIFFER");

    /* overlap: bg thread transcodes 8 while main runs IQ2 GEMV on 8 */
    struct bg b = {0, 8, 0};
    pthread_t th;
    double t0 = now_ns();
    pthread_create(&th, NULL, bg_worker, &b);
    memset(yref, 0, sizeof yref);
    for (int k = 0; k < TOPK; k++) expert_iq2(k, yref);
    double fg_ms = (now_ns() - t0) / 1e6;
    pthread_join(th, NULL);
    printf("overlap: fg_gemv_ms=%.3f bg_transcode_ms=%.3f hidden=%s\n",
        fg_ms, b.ms, b.ms <= fg_ms ? "YES" : "NO");
    return 0;
}
