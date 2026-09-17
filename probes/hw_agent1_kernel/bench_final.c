// Final accounting: full expert-region single-thread GEMV + MT scaling.
// All kernels = REAL ggml AVX2 from pinned llama.cpp 3057bb66, -O3 -march=native.
// Weights = REAL bytes from pinned UD-IQ2_XXS GGUF. Pool-streaming regime.
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
void ggml_vec_dot_q5_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
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
static void * load_file(const char * p, size_t n) {
    FILE * f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p); exit(1); }
    void * b = aligned_alloc(64, (n + 63) & ~(size_t)63);
    if (fread(b, 1, n, f) != n) { fprintf(stderr, "short %s\n", p); exit(1); }
    fclose(f);
    return b;
}
static double read_mhz(void) {
    FILE * f = fopen("/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq", "r");
    if (!f) return 0;
    double khz = 0;
    if (fscanf(f, "%lf", &khz) != 1) khz = 0;
    fclose(f);
    return khz / 1000.0;
}

struct mt_args {
    int kind; // 0=gatexxs 1=downiq2s
    const uint8_t * wpool;
    size_t stride;
    const uint8_t * xq;
    float * y;
    int r0, r1;
    int rowb;
};

static void * mt_worker(void * v) {
    struct mt_args * a = v;
    if (a->kind == 0)
        for (int r = a->r0; r < a->r1; r++)
            ggml_vec_dot_iq2_xxs_q8_K(2048, &a->y[r], 0, a->wpool + r * a->rowb, 0, a->xq, 0, 1);
    else
        for (int r = a->r0; r < a->r1; r++)
            ggml_vec_dot_iq2_s_q8_K(512, &a->y[r], 0, a->wpool + r * a->rowb, 0, a->xq, 0, 1);
    return NULL;
}

int main(void) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    uint8_t * gate = load_file("/tmp/agent1_raw/L00_E000_gate.iq2xxs", 270336);
    uint8_t * up = load_file("/tmp/agent1_raw/L00_E000_up.iq2xxs", 270336);
    uint8_t * down = load_file("/tmp/agent1_raw/L00_E000_down.iq2s", 335872);
    uint8_t * sg = load_file("/tmp/agent1_raw/L00_shexp_gate.q5k", 720896);
    uint8_t * su = load_file("/tmp/agent1_raw/L00_shexp_up.q5k", 720896);
    uint8_t * sd = load_file("/tmp/agent1_raw/L00_shexp_down.q6k", 860160);

    // streaming pools (32MB each to fit 2GB box alongside others)
    const size_t POOL = 32 << 20;
    uint8_t * p_gate, *p_up, *p_down, *p_sg, *p_su, *p_sd;
    size_t n_gate, n_up, n_down, n_sg, n_su, n_sd;
#define MAKEPOOL(dst, n, src, sz) do { \
    dst = aligned_alloc(64, POOL); n = POOL / (sz); \
    for (size_t c = 0; c < n; c++) memcpy(dst + c * (sz), src, sz); } while (0)
    MAKEPOOL(p_gate, n_gate, gate, 270336);
    MAKEPOOL(p_up, n_up, up, 270336);
    MAKEPOOL(p_down, n_down, down, 335872);
    MAKEPOOL(p_sg, n_sg, sg, 720896);
    MAKEPOOL(p_su, n_su, su, 720896);
    MAKEPOOL(p_sd, n_sd, sd, 860160);

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
    float * gout = aligned_alloc(64, 512 * sizeof(float));
    float * uout = aligned_alloc(64, 512 * sizeof(float));
    float * h = aligned_alloc(64, 512 * sizeof(float));
    for (int r = 0; r < 512; r++)
        ggml_vec_dot_iq2_xxs_q8_K(2048, &gout[r], 0, gate + r * 528, 0, xq, 0, 1);
    for (int r = 0; r < 512; r++)
        ggml_vec_dot_iq2_xxs_q8_K(2048, &uout[r], 0, up + r * 528, 0, xq, 0, 1);
    for (int i = 0; i < 512; i++) { float g = gout[i]; h[i] = (g / (1.0f + expf(-g))) * uout[i]; }
    uint8_t * hq = aligned_alloc(64, 2 * 292);
    quantize_row_q8_K(h, hq, 512);
    float * yout = aligned_alloc(64, 2048 * sizeof(float));

    printf("cpu2 freq: %.0f MHz (sampled)\n", read_mhz());

    const int T = 36;
    double t_gate[36], t_up[36], t_down[36], t_sg[36], t_su[36], t_sd[36], t_q[36];
    for (int t = 0; t < T; t++) {
        double s;
        s = now_ns();
        { uint8_t * w = p_gate + (t % n_gate) * 270336;
          for (int r = 0; r < 512; r++) ggml_vec_dot_iq2_xxs_q8_K(2048, &gout[r], 0, w + r * 528, 0, xq, 0, 1); }
        t_gate[t] = now_ns() - s;
        s = now_ns();
        { uint8_t * w = p_up + (t % n_up) * 270336;
          for (int r = 0; r < 512; r++) ggml_vec_dot_iq2_xxs_q8_K(2048, &uout[r], 0, w + r * 528, 0, xq, 0, 1); }
        t_up[t] = now_ns() - s;
        s = now_ns();
        { uint8_t * w = p_down + (t % n_down) * 335872;
          for (int r = 0; r < 2048; r++) ggml_vec_dot_iq2_s_q8_K(512, &yout[r], 0, w + r * 164, 0, hq, 0, 1); }
        t_down[t] = now_ns() - s;
        s = now_ns();
        { uint8_t * w = p_sg + (t % n_sg) * 720896;
          for (int r = 0; r < 512; r++) ggml_vec_dot_q5_K_q8_K(2048, &gout[r], 0, w + r * 1408, 0, xq, 0, 1); }
        t_sg[t] = now_ns() - s;
        s = now_ns();
        { uint8_t * w = p_su + (t % n_su) * 720896;
          for (int r = 0; r < 512; r++) ggml_vec_dot_q5_K_q8_K(2048, &uout[r], 0, w + r * 1408, 0, xq, 0, 1); }
        t_su[t] = now_ns() - s;
        s = now_ns();
        { uint8_t * w = p_sd + (t % n_sd) * 860160;
          for (int r = 0; r < 2048; r++) ggml_vec_dot_q6_K_q8_K(512, &yout[r], 0, w + r * 420, 0, hq, 0, 1); }
        t_sd[t] = now_ns() - s;
        s = now_ns();
        quantize_row_q8_K(x, xq, 2048);
        quantize_row_q8_K(h, hq, 512);
        t_q[t] = now_ns() - s;
    }
    qsort(t_gate, T, sizeof(double), cmp_dbl);
    qsort(t_up, T, sizeof(double), cmp_dbl);
    qsort(t_down, T, sizeof(double), cmp_dbl);
    qsort(t_sg, T, sizeof(double), cmp_dbl);
    qsort(t_su, T, sizeof(double), cmp_dbl);
    qsort(t_sd, T, sizeof(double), cmp_dbl);
    qsort(t_q, T, sizeof(double), cmp_dbl);
    double mg = t_gate[T/2], mu = t_up[T/2], md = t_down[T/2];
    double ms_g = t_sg[T/2], ms_u = t_su[T/2], ms_d = t_sd[T/2], mq = t_q[T/2];
    printf("ST medians (pool-streaming, pinned):\n");
    printf("  routed gate  XXS 512x2048: %9.0f ns (%6.1f ns/row) BW=%5.2f GB/s\n", mg, mg/512, 270336/mg);
    printf("  routed up    XXS 512x2048: %9.0f ns (%6.1f ns/row) BW=%5.2f GB/s\n", mu, mu/512, 270336/mu);
    printf("  routed down  IQ2S 2048x512: %9.0f ns (%6.1f ns/row) BW=%5.2f GB/s\n", md, md/2048, 335872/md);
    printf("  shared gate  Q5K 512x2048: %9.0f ns (%6.1f ns/row) BW=%5.2f GB/s\n", ms_g, ms_g/512, 720896/ms_g);
    printf("  shared up    Q5K 512x2048: %9.0f ns (%6.1f ns/row) BW=%5.2f GB/s\n", ms_u, ms_u/512, 720896/ms_u);
    printf("  shared down  Q6K 2048x512: %9.0f ns (%6.1f ns/row) BW=%5.2f GB/s\n", ms_d, ms_d/2048, 860160/ms_d);
    printf("  q8_K quant x+h:            %9.0f ns\n", mq);
    double routed = mg + mu + md;
    double shared = ms_g + ms_u + ms_d;
    double per_layer_1t = routed * 8 + shared;
    printf("  ONE routed expert: %.1f us; shared expert: %.1f us\n", routed/1e3, shared/1e3);
    printf("  ST per-layer experts: %.3f ms; ST x40 = %.1f ms/token (experts only, no dispatch)\n",
           per_layer_1t/1e6, per_layer_1t*40/1e6);
    printf("  /4-ideal x40 = %.1f ms/token\n", per_layer_1t*40/1e6/4);
    printf("cpu2 freq: %.0f MHz (sampled)\n", read_mhz());

    // ---- MT scaling on ONE routed expert (gate+down), row-split ----
    for (int nth = 1; nth <= 4; nth *= 2) {
        double tm[24];
        for (int t = 0; t < 24; t++) {
            uint8_t * wg = p_gate + (t % n_gate) * 270336;
            uint8_t * wd = p_down + (t % n_down) * 335872;
            pthread_t th[4];
            struct mt_args args[4];
            double s = now_ns();
            for (int i = 0; i < nth; i++) {
                args[i].kind = 0; args[i].wpool = wg; args[i].xq = xq; args[i].y = gout;
                args[i].r0 = 512 * i / nth; args[i].r1 = 512 * (i + 1) / nth; args[i].rowb = 528;
                if (i + 1 < nth) pthread_create(&th[i], NULL, mt_worker, &args[i]);
                else mt_worker(&args[i]);
            }
            for (int i = 0; i + 1 < nth; i++) pthread_join(th[i], NULL);
            for (int i = 0; i < nth; i++) {
                args[i].kind = 1; args[i].wpool = wd; args[i].xq = hq; args[i].y = yout;
                args[i].r0 = 2048 * i / nth; args[i].r1 = 2048 * (i + 1) / nth; args[i].rowb = 164;
                if (i + 1 < nth) pthread_create(&th[i], NULL, mt_worker, &args[i]);
                else mt_worker(&args[i]);
            }
            for (int i = 0; i + 1 < nth; i++) pthread_join(th[i], NULL);
            tm[t] = now_ns() - s;
        }
        qsort(tm, 24, sizeof(double), cmp_dbl);
        printf("  MT gate+down expert-part with %d threads: med=%.0f ns\n", nth, tm[12]);
    }
    return 0;
}
