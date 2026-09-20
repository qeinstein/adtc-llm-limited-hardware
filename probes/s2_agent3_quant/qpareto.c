// qpareto: quant-format Pareto for MoE experts under the noDN staged path.
// Method: dequantize REAL L00 IQ2 expert bytes to f32 (real ggml dequant),
// requantize to Q2_K/Q3_K/Q4_K (real ggml quantizers), time the REAL per-format
// AVX2 GEMV kernels + staging-miss refill. Single-threaded (4T scaling is the
// known ~3.5x); medians over trials. Caveat (see REPORT): requant-from-IQ2 is
// NOT quant-from-fp16, so the quality gate for any promoted format stays open.
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q2_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void quantize_row_q8_K(const float * x, void * y, int64_t k);
void quantize_row_q2_K_ref(const float * x, void * y, int64_t k);
void quantize_row_q3_K_ref(const float * x, void * y, int64_t k);
void quantize_row_q4_K_ref(const float * x, void * y, int64_t k);
void dequantize_row_iq2_xxs(const void * x, float * y, int64_t k);
void dequantize_row_iq2_s(const void * x, float * y, int64_t k);

#define DIN 2048
#define DFF 512
#define NEXP 8
#define GATEB_IQ2 270336
#define DOWNB_IQ2 335872
#define ROWB_GU_IQ2 528
#define ROWB_DN_IQ2 164
// K-quant super-block 256: q2=84B, q3=110B, q4=144B (ggml-common.h asserts)
#define RB(nc, bs) ((size_t)(nc) / 256 * (bs))

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
    if (fd < 0) { fprintf(stderr, "open %s\n", path); exit(1); }
    void * p = malloc(want ? want : 1);
    size_t off = 0;
    while (off < want) {
        ssize_t r = read(fd, (char *)p + off, want - off);
        if (r <= 0) { fprintf(stderr, "read %s\n", path); exit(1); }
        off += r;
    }
    close(fd);
    return p;
}

typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*quant_fn)(const float *, void *, int64_t);

struct fmt {
    const char * name;
    vec_dot_fn gu, dn;
    quant_fn q;
    size_t rb_gu, rb_dn; // row bytes at DIN/DFF cols
};

int main(void) {
    struct fmt F[] = {
        { "IQ2", ggml_vec_dot_iq2_xxs_q8_K, ggml_vec_dot_iq2_s_q8_K, NULL, ROWB_GU_IQ2, ROWB_DN_IQ2 },
        { "Q2_K", ggml_vec_dot_q2_K_q8_K, ggml_vec_dot_q2_K_q8_K, quantize_row_q2_K_ref, RB(DIN, 84), RB(DFF, 84) },
        { "Q3_K", ggml_vec_dot_q3_K_q8_K, ggml_vec_dot_q3_K_q8_K, quantize_row_q3_K_ref, RB(DIN, 110), RB(DFF, 110) },
        { "Q4_K", ggml_vec_dot_q4_K_q8_K, ggml_vec_dot_q4_K_q8_K, quantize_row_q4_K_ref, RB(DIN, 144), RB(DFF, 144) },
    };
    // load REAL IQ2 backing: 8 L00 experts
    static uint8_t * W[NEXP][3];
    char path[512];
    for (int k = 0; k < NEXP; k++) {
        snprintf(path, sizeof path, "/tmp/agent1_raw/L00_E%03d_gate.iq2xxs", k);
        W[k][0] = load_file(path, GATEB_IQ2);
        snprintf(path, sizeof path, "/tmp/agent1_raw/L00_E%03d_up.iq2xxs", k);
        W[k][1] = load_file(path, GATEB_IQ2);
        snprintf(path, sizeof path, "/tmp/agent1_raw/L00_E%03d_down.iq2s", k);
        W[k][2] = load_file(path, DOWNB_IQ2);
    }
    static float X[DIN];
    for (int i = 0; i < DIN; i++) X[i] = sinf(i * 0.37f) * 0.6f + cosf(i * 0.11f) * 0.4f;
    static uint8_t XQK[DIN * 2], HQK[DFF * 2];
    quantize_row_q8_K(X, XQK, DIN);

    static float gout[DFF], uout[DFF], h[DFF], dout[DIN];
    float ref_y[DIN];
    // requant staging buffers (max = Q4_K sizes)
    size_t maxslot = RB(DIN, 144) * DFF * 2 + RB(DFF, 144) * DIN;
    uint8_t * Q = malloc(maxslot * NEXP);
    float * fbuf = malloc(DIN * sizeof(float));

    for (int f = 0; f < 4; f++) {
        size_t gateB = F[f].rb_gu * DFF, downB = F[f].rb_dn * DIN;
        size_t expB = gateB * 2 + downB;
        // requant once (one-time cost, timed separately)
        double t0 = now_ns();
        if (F[f].q) {
            for (int k = 0; k < NEXP; k++) {
                uint8_t * q = Q + (size_t)k * maxslot;
                for (int r = 0; r < DFF; r++) {
                    dequantize_row_iq2_xxs(W[k][0] + r * ROWB_GU_IQ2, fbuf, DIN);
                    F[f].q(fbuf, q + r * F[f].rb_gu, DIN);
                }
                q += gateB;
                for (int r = 0; r < DFF; r++) {
                    dequantize_row_iq2_xxs(W[k][1] + r * ROWB_GU_IQ2, fbuf, DIN);
                    F[f].q(fbuf, q + r * F[f].rb_gu, DIN);
                }
                q += gateB;
                for (int r = 0; r < DIN; r++) {
                    dequantize_row_iq2_s(W[k][2] + r * ROWB_DN_IQ2, fbuf, DFF);
                    F[f].q(fbuf, q + r * F[f].rb_dn, DFF);
                }
            }
        }
        double requant_ms = (now_ns() - t0) / 1e6;
        // GEMV trials (decode-like: weights streamed, act hot)
#define TRIALS 15
        double t_g[TRIALS];
        float y[DIN];
        for (int t = 0; t < TRIALS; t++) {
            memset(y, 0, sizeof y);
            t0 = now_ns();
            for (int k = 0; k < NEXP; k++) {
                const uint8_t * base = F[f].q ? Q + (size_t)k * maxslot : NULL;
                const char * g = F[f].q ? (const char *)base : (const char *)W[k][0];
                const char * u = F[f].q ? (const char *)base + gateB : (const char *)W[k][1];
                const char * d = F[f].q ? (const char *)base + 2 * gateB : (const char *)W[k][2];
                for (int r = 0; r < DFF; r++)
                    F[f].gu(DIN, &gout[r], 0, g + r * F[f].rb_gu, 0, XQK, 0, 1);
                for (int r = 0; r < DFF; r++)
                    F[f].gu(DIN, &uout[r], 0, u + r * F[f].rb_gu, 0, XQK, 0, 1);
                for (int r = 0; r < DFF; r++) {
                    float gg = gout[r];
                    h[r] = (gg / (1.0f + expf(-gg))) * uout[r];
                }
                quantize_row_q8_K(h, HQK, DFF);
                for (int r = 0; r < DIN; r++)
                    F[f].dn(DFF, &dout[r], 0, d + r * F[f].rb_dn, 0, HQK, 0, 1);
                for (int r = 0; r < DIN; r++) y[r] += dout[r] / NEXP;
            }
            t_g[t] = now_ns() - t0;
        }
        qsort(t_g, TRIALS, sizeof(double), cmp_dbl);
        double gemv_med = t_g[TRIALS / 2] / 1e6;
        // staging-miss refill cost: memcpy of one expert bundle, PTE-present
        uint8_t * slot = malloc(expB);
        const uint8_t * src = F[f].q ? Q : W[0][0];
        // for IQ2 the 3 planes are discontiguous; time plane-by-plane like stage_load
        double t_m[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            t0 = now_ns();
            if (F[f].q) {
                memcpy(slot, src, expB);
            } else {
                memcpy(slot, W[0][0], GATEB_IQ2);
                memcpy(slot + GATEB_IQ2, W[0][1], GATEB_IQ2);
                memcpy(slot + 2 * GATEB_IQ2, W[0][2], DOWNB_IQ2);
            }
            t_m[t] = now_ns() - t0;
        }
        qsort(t_m, TRIALS, sizeof(double), cmp_dbl);
        // quality proxy vs IQ2 reference (first format run = IQ2)
        double maxad = 0;
        if (f == 0) memcpy(ref_y, y, sizeof y);
        else {
            for (int i = 0; i < DIN; i++) {
                double ad = fabs(y[i] - ref_y[i]);
                if (ad > maxad) maxad = ad;
            }
        }
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        printf("%-4s bytes_exp=%zu (%.2fx IQ2) requant8_ms=%.1f(1x) gemv8_ms_med=%.3f miss_refill_us=%.1f maxabsdiff_vs_IQ2=%.6f minflt=%ld\n",
            F[f].name, expB, (double)expB / 876544.0, requant_ms, gemv_med,
            t_m[TRIALS / 2] / 1e3, maxad, ru.ru_minflt);
        free(slot);
    }
    return 0;
}
