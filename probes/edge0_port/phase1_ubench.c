// Phase 1 shootout: CURRENT K=8+IQ2 vs EDGE0-STYLE K=4+INT4 on exact Qwen
// expert shapes with REAL weight bytes and REAL ggml kernels (pin 3057bb66).
//
// Arms:
//   k8_iq2 : K=8 gate/up IQ2_XXS + down IQ2_S, act q8_K  (current production)
//   k4_q40 : K=4 gate/up/down Q4_0, act q8_0             (Edge0-style INT4)
//   k4_q4k : K=4 gate/up/down Q4_K, act q8_K             (INT4 super-block ref)
//   k8_q40 : K=8 Q4_0 (diagnostic: isolates format vs K effects)
//   k8_q2k / k4_q2k : Q2_K (2.625 bpw, plain-decode 2-bit candidate)
//   k8_q3k / k4_q3k : Q3_K (3.4375 bpw, plain-decode 3-bit candidate)
//
// Weights: real L20 experts (256/expert-full layer) from /tmp/agent1_raw
// (IQ2 bytes) and /tmp/agent1_f32 (f32 dequants -> requantized at init with
// the REAL ggml quantize_row_*). Pools are 32 MB each (>> 6 MB LLC) of
// DISTINCT real rows; activations are RMS-normalized randn (vec-dot time
// is value-independent; FTZ/DAZ on).
//
// Usage: phase1_ubench st|mt4|ratio [arm|all] [trials]
//   st    : single-thread streaming passes, interleaved + rotated arms
//   mt4   : 4 persistent pthreads (sem-dispatched, yield-joined)
//   ratio : adjacent ST/MT pairs per (arm, proj); drift-cancelling scaling
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <x86intrin.h>
#include <pmmintrin.h>

#include "ggml-cpu/quants.h"
#include "ggml.h"

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
// Self-counters: perf_event_open on SELF (pid 0), bracketed tightly around
// the trial loop so pool-build/quantize init cannot pollute. Run single-arm
// (./ubench st <arm> N) for exact per-arm steady-state counts.
// Disable with HWP_SELF=0.
static int hwp_open(uint32_t type, uint64_t config) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof(a));
    a.size = sizeof(a);
    a.type = type;
    a.config = config;
    a.disabled = 0;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.exclude_idle = 1;
    return syscall(__NR_perf_event_open, &a, 0, -1, -1, 0);
}
static uint64_t hwp_read(int fd) {
    uint64_t v = 0;
    if (fd >= 0) {
        ssize_t r = read(fd, &v, sizeof(v));
        (void)r;
    }
    return v;
}
struct hwp { int cyc, ins, cref, cmiss, br, brm; int ok; };
static void hwp_start(struct hwp *h) {
    const char *e = getenv("HWP_SELF");
    h->ok = 0;
    if (e && !strcmp(e, "0")) return;
    h->cyc = hwp_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    h->ins = hwp_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    h->cref = hwp_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    h->cmiss = hwp_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    h->br = hwp_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    h->brm = hwp_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
    h->ok = (h->cyc >= 0 && h->ins >= 0);
    if (!h->ok) {
        int fds[6] = {h->cyc, h->ins, h->cref, h->cmiss, h->br, h->brm};
        for (int i = 0; i < 6; i++) if (fds[i] >= 0) close(fds[i]);
    }
}
#endif

#define POOL_BYTES (32u << 20)
#define MAX_EXPERTS 256
#define RAW_DIR "/tmp/agent1_raw"
#define F32_DIR "/tmp/agent1_f32"

typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t,
                           const void *, size_t, int);
typedef void (*quant_fn)(const float *, void *, int64_t);

struct arm {
    const char *name;
    int K;
    enum ggml_type wgu, wd, atype;
    vec_dot_fn dot_gu, dot_d;
    quant_fn qact;
    double bpw_gu, bpw_d;
};

static struct arm ARMS[] = {
    {"k8_iq2", 8, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_Q8_K,
     ggml_vec_dot_iq2_xxs_q8_K, ggml_vec_dot_iq2_s_q8_K, quantize_row_q8_K,
     2.0625, 2.5},
    {"k4_q40", 4, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
     ggml_vec_dot_q4_0_q8_0, ggml_vec_dot_q4_0_q8_0, quantize_row_q8_0,
     4.5, 4.5},
    {"k4_q4k", 4, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q8_K,
     ggml_vec_dot_q4_K_q8_K, ggml_vec_dot_q4_K_q8_K, quantize_row_q8_K,
     4.5, 4.5},
    {"k8_q40", 8, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0,
     ggml_vec_dot_q4_0_q8_0, ggml_vec_dot_q4_0_q8_0, quantize_row_q8_0,
     4.5, 4.5},
    {"k8_q2k", 8, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q8_K,
     ggml_vec_dot_q2_K_q8_K, ggml_vec_dot_q2_K_q8_K, quantize_row_q8_K,
     2.625, 2.625},
    {"k4_q2k", 4, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q8_K,
     ggml_vec_dot_q2_K_q8_K, ggml_vec_dot_q2_K_q8_K, quantize_row_q8_K,
     2.625, 2.625},
    {"k8_q3k", 8, GGML_TYPE_Q3_K, GGML_TYPE_Q3_K, GGML_TYPE_Q8_K,
     ggml_vec_dot_q3_K_q8_K, ggml_vec_dot_q3_K_q8_K, quantize_row_q8_K,
     3.4375, 3.4375},
    {"k4_q3k", 4, GGML_TYPE_Q3_K, GGML_TYPE_Q3_K, GGML_TYPE_Q8_K,
     ggml_vec_dot_q3_K_q8_K, ggml_vec_dot_q3_K_q8_K, quantize_row_q8_K,
     3.4375, 3.4375},
};
#define NARMS (sizeof(ARMS) / sizeof(ARMS[0]))

struct pool {
    uint8_t *buf;
    size_t wrow;      // bytes per weight row
    size_t nrows;     // rows in pool
    int nexperts;     // distinct real experts loaded
    int K;            // activation width
    size_t actsz;     // quantized activation bytes
    uint8_t *act;     // quantized activation (hot)
    vec_dot_fn dot;
    quant_fn qact;
    const char *tag;
};

// gate/up: 512 rows x K=2048 ; down: 2048 rows x K=512
static const char *PROJS[3] = {"gate", "up", "down"};
static const int PK[3] = {2048, 2048, 512};
static const int PROWS[3] = {512, 512, 2048};

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t fnv1a(const void *p, size_t n) {
    const uint8_t *b = p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static void pin_core(int core) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(core, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

static void fill_randn(float *x, int n, double rms) {
    uint64_t s = 0x243F6A8885A308D3ULL;
    double sum = 0;
    for (int i = 0; i < n; i++) {
        // xorshift64* uniform -> Box-Muller
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        double u1 = ((s * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        double u2 = ((s * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
        if (u1 < 1e-12) u1 = 1e-12;
        double g = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
        x[i] = (float)g;
        sum += g * g;
    }
    double f = rms / sqrt(sum / n);
    for (int i = 0; i < n; i++) x[i] = (float)(x[i] * f);
}

static size_t read_file(const char *path, void *buf, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(buf, 1, cap, f);
    fclose(f);
    return n;
}

// Load one pool of DISTINCT real rows for (wtype, proj). Returns rows loaded.
static size_t load_pool(struct pool *p, enum ggml_type wt, int proj,
                        quant_fn qweight /*NULL for direct IQ2 copy*/) {
    char path[256];
    const char *pn = PROJS[proj];
    int K = PK[proj], nrows_exp = PROWS[proj];
    size_t wrow = ggml_row_size(wt, K);
    size_t f32sz = (size_t)nrows_exp * K * 4;
    float *f32 = NULL;
    uint8_t *raw = NULL;
    if (qweight) {
        f32 = aligned_alloc(64, f32sz);
        if (!f32) { fprintf(stderr, "oom f32\n"); exit(1); }
    } else {
        // raw IQ2 file size unknown upfront; allocate f32-equivalent cap
        raw = malloc(f32sz);
        if (!raw) { fprintf(stderr, "oom raw\n"); exit(1); }
    }
    size_t off = 0;
    int nexp = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        size_t got = 0;
        if (qweight) {
            snprintf(path, sizeof(path), "%s/L20_E%03d_%s.f32", F32_DIR, e, pn);
            got = read_file(path, f32, f32sz);
            if (got != f32sz) continue; // missing/short: skip expert
        } else {
            const char *ext = (proj == 2) ? "iq2s" : "iq2xxs";
            snprintf(path, sizeof(path), "%s/L20_E%03d_%s.%s", RAW_DIR, e, pn, ext);
            got = read_file(path, raw, f32sz);
            size_t expect = wrow * (size_t)nrows_exp;
            if (got != expect) continue;
        }
        if (off + (size_t)nrows_exp * wrow > POOL_BYTES) break; // pool full
        if (qweight) {
            for (int r = 0; r < nrows_exp; r++)
                qweight(f32 + (size_t)r * K, p->buf + off + (size_t)r * wrow, K);
        } else {
            memcpy(p->buf + off, raw, (size_t)nrows_exp * wrow);
        }
        off += (size_t)nrows_exp * wrow;
        nexp++;
    }
    free(f32);
    free(raw);
    p->wrow = wrow;
    p->nrows = off / wrow;
    p->nexperts = nexp;
    p->K = K;
    return p->nrows;
}

// ---- MT4 worker machinery (persistent threads, sem dispatch) ----
static struct {
    vec_dot_fn dot;
    uint8_t *pool;
    size_t wrow, nrows;
    uint8_t *act;
    int K;
    float *out;
    sem_t wake;          // workers sleep here when idle (no spin contention
                         // against ST passes sharing the cores)
    atomic_int done;     // workers ack
    int nth;
    _Atomic int stop;
} MT;

static void *worker_fn(void *arg) {
    int tid = (int)(intptr_t)arg;
    pin_core(tid);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    while (1) {
        sem_wait(&MT.wake);
        if (atomic_load(&MT.stop)) break;
        size_t chunk = (MT.nrows + MT.nth - 1) / MT.nth;
        size_t lo = (size_t)tid * chunk;
        size_t hi = lo + chunk;
        if (hi > MT.nrows) hi = MT.nrows;
        for (size_t r = lo; r < hi; r++)
            MT.dot(MT.K, &MT.out[r], 0, MT.pool + r * MT.wrow, 0, MT.act, 0, 1);
        atomic_fetch_add(&MT.done, 1);
    }
    return NULL;
}

static uint64_t run_mt4(struct pool *p, float *out) {
    MT.dot = p->dot;
    MT.pool = p->buf;
    MT.wrow = p->wrow;
    MT.nrows = p->nrows;
    MT.act = p->act;
    MT.K = p->K;
    MT.out = out;
    atomic_store(&MT.done, 0);
    uint64_t t0 = now_ns();
    for (int i = 0; i < MT.nth; i++) sem_post(&MT.wake);
    // Main YIELD-joins (it shares core 0 with worker 0; pause-spinning
    // would steal that core).
    while (atomic_load(&MT.done) < MT.nth) {
        sched_yield();
    }
    return now_ns() - t0;
}

static uint64_t run_st(struct pool *p, float *out) {
    uint64_t t0 = now_ns();
    for (size_t r = 0; r < p->nrows; r++)
        p->dot(p->K, &out[r], 0, p->buf + r * p->wrow, 0, p->act, 0, 1);
    return now_ns() - t0;
}

int main(int argc, char **argv) {
    int mt4 = (argc > 1 && !strcmp(argv[1], "mt4"));
    int ratio = (argc > 1 && !strcmp(argv[1], "ratio"));
    if (ratio) mt4 = 1; // ratio mode needs the worker threads too
    const char *only = (argc > 2) ? argv[2] : "all";
    int trials = (argc > 3) ? atoi(argv[3]) : 7;
    if (trials > 32) trials = 32;
    if (trials < 1) trials = 1;
    if (!mt4 && (argc < 2 || strcmp(argv[1], "st"))) {
        fprintf(stderr, "usage: %s st|mt4|ratio [arm|all] [trials]\n", argv[0]);
        return 1;
    }
    pin_core(mt4 ? 0 : 2);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    ggml_quantize_init(GGML_TYPE_IQ2_XXS);
    ggml_quantize_init(GGML_TYPE_IQ2_S);

    // activation float vectors (RMS 1.0, realistic post-norm scale)
    float *fact2048 = aligned_alloc(64, 2048 * sizeof(float));
    float *fact512 = aligned_alloc(64, 512 * sizeof(float));
    fill_randn(fact2048, 2048, 1.0);
    fill_randn(fact512, 512, 1.0);

    // Build pools: per arm, 3 projs. With `only`, build just that arm
    // (fast init for counter runs).
    struct pool P[NARMS][3];
    memset(P, 0, sizeof(P));
    size_t maxrows = 0;
    for (int a = 0; a < (int)NARMS; a++) {
        if (strcmp(only, "all") && strcmp(only, ARMS[a].name)) continue;
        for (int pr = 0; pr < 3; pr++) {
            struct pool *p = &P[a][pr];
            enum ggml_type wt = (pr == 2) ? ARMS[a].wd : ARMS[a].wgu;
            p->buf = aligned_alloc(64, POOL_BYTES + 4096);
            p->dot = (pr == 2) ? ARMS[a].dot_d : ARMS[a].dot_gu;
            p->qact = ARMS[a].qact;
            p->tag = PROJS[pr];
            quant_fn qw = NULL;
            if (wt == GGML_TYPE_Q4_0) qw = quantize_row_q4_0;
            else if (wt == GGML_TYPE_Q4_K) qw = quantize_row_q4_K;
            else if (wt == GGML_TYPE_Q2_K) qw = quantize_row_q2_K;
            else if (wt == GGML_TYPE_Q3_K) qw = quantize_row_q3_K;
            size_t nr = load_pool(p, wt, pr, qw);
            p->actsz = ggml_row_size(ARMS[a].atype, p->K);
            p->act = aligned_alloc(64, p->actsz + 64);
            p->qact((p->K == 2048) ? fact2048 : fact512, p->act, p->K);
            if (nr > maxrows) maxrows = nr;
            uint64_t ph = fnv1a(p->buf, p->nrows * p->wrow);
            printf("pool %-6s %-4s wrow=%4zu nrows=%6zu nexperts=%3d poolhash=%016llx\n",
                   ARMS[a].name, p->tag, p->wrow, p->nrows, p->nexperts,
                   (unsigned long long)ph);
        }
    }
    fflush(stdout);
    float *out = aligned_alloc(64, maxrows * sizeof(float) + 64);

    // Activation-quantize micro-cost (per call; production does 40x(q2048+q512))
    for (int a = 0; a < (int)NARMS; a++) {
        if (!P[a][0].buf) continue;
        for (int pr = 0; pr < 3; pr += 2) { // 2048-wide and 512-wide
            struct pool *p = &P[a][pr];
            uint64_t t0 = now_ns();
            const int NQ = 200;
            for (int i = 0; i < NQ; i++)
                p->qact((p->K == 2048) ? fact2048 : fact512, p->act, p->K);
            double ns = (double)(now_ns() - t0) / NQ;
            printf("qact %-6s K=%d ns/call=%.1f\n", ARMS[a].name, p->K, ns);
        }
    }
    fflush(stdout);

    // Warmup: one full pass per pool, discarded (defeats freq-ramp /
    // cold-iTLB ordering bias; early arms measured ~35% slow without it).
    for (int a = 0; a < (int)NARMS; a++) {
        if (!P[a][0].buf) continue;
        for (int pr = 0; pr < 3; pr++)
            run_st(&P[a][pr], out);
    }
    printf("warmup done\n");
    fflush(stdout);

    pthread_t th[4];
    if (mt4) {
        MT.nth = 4;
        sem_init(&MT.wake, 0, 0);
        atomic_init(&MT.done, 0);
        atomic_init(&MT.stop, 0);
        for (int t = 0; t < 4; t++)
            pthread_create(&th[t], NULL, worker_fn, (void *)(intptr_t)t);
        // warmup dispatch
        run_mt4(&P[0][0], out);
    }

    double res[NARMS][3][32]; // ns/row per trial (trials<=32)
    uint64_t chk[NARMS][3];
    memset(chk, 0, sizeof(chk));
    int order[NARMS];
    for (int a = 0; a < (int)NARMS; a++) order[a] = a;

    if (ratio) {
        // Adjacent ST/MT pairs per (arm, proj): window drift cancels, the
        // ratio is the robust signal on a noisy host. Checksums verify both
        // paths compute identically.
        double rat[NARMS][3][32];
        for (int t = 0; t < trials; t++) {
            for (int a = 0; a < (int)NARMS; a++) {
                int ai = order[(a + t) % NARMS];
                if (strcmp(only, "all") && strcmp(only, ARMS[ai].name)) continue;
                for (int pr = 0; pr < 3; pr++) {
                    struct pool *p = &P[ai][pr];
                    uint64_t dt_st = run_st(p, out);
                    uint64_t h_st = fnv1a(out, p->nrows * sizeof(float));
                    uint64_t dt_mt = run_mt4(p, out);
                    uint64_t h_mt = fnv1a(out, p->nrows * sizeof(float));
                    double r = (double)dt_st / (double)dt_mt;
                    rat[ai][pr][t] = r;
                    printf("trial %d %-6s %-4s st_ns=%8.1f mt_ns=%8.1f "
                           "ratio=%5.2f %s\n",
                           t, ARMS[ai].name, p->tag,
                           (double)dt_st / p->nrows, (double)dt_mt / p->nrows, r,
                           (h_st == h_mt) ? "st=mt" : "MISMATCH!");
                }
            }
            fflush(stdout);
        }
        printf("\nratio summary (%d trials, max/median scaling):\n", trials);
        for (int a = 0; a < (int)NARMS; a++) {
            if (strcmp(only, "all") && strcmp(only, ARMS[a].name)) continue;
            printf("%-6s", ARMS[a].name);
            for (int pr = 0; pr < 3; pr++) {
                double tmp[32];
                for (int t = 0; t < trials; t++) tmp[t] = rat[a][pr][t];
                for (int i = 1; i < trials; i++) {
                    double k = tmp[i];
                    int j = i - 1;
                    while (j >= 0 && tmp[j] > k) { tmp[j + 1] = tmp[j]; j--; }
                    tmp[j + 1] = k;
                }
                printf(" %s=%5.2f/%5.2f", PROJS[pr], tmp[trials - 1],
                       tmp[trials / 2]);
            }
            printf("\n");
        }
        atomic_store(&MT.stop, 1);
        for (int t = 0; t < 4; t++) sem_post(&MT.wake);
        for (int t = 0; t < 4; t++) pthread_join(th[t], NULL);
        return 0;
    }

    struct hwp hw;
    uint64_t b_cyc = 0, b_ins = 0, b_cref = 0, b_cmiss = 0, b_br = 0, b_brm = 0;
#ifdef __linux__
    hwp_start(&hw);
    if (hw.ok) {
        b_cyc = hwp_read(hw.cyc); b_ins = hwp_read(hw.ins);
        b_cref = hwp_read(hw.cref); b_cmiss = hwp_read(hw.cmiss);
        b_br = hwp_read(hw.br); b_brm = hwp_read(hw.brm);
    }
#endif
    uint64_t total_rows = 0;
    uint64_t total_ns = 0;

    for (int t = 0; t < trials; t++) {
        // rotate arm order per trial to defeat drift
        for (int a = 0; a < (int)NARMS; a++) {
            int ai = order[(a + t) % NARMS];
            if (strcmp(only, "all") && strcmp(only, ARMS[ai].name)) continue;
            for (int pr = 0; pr < 3; pr++) {
                struct pool *p = &P[ai][pr];
                uint64_t dt = mt4 ? run_mt4(p, out) : run_st(p, out);
                total_rows += p->nrows;
                total_ns += dt;
                double nsr = (double)dt / (double)p->nrows;
                res[ai][pr][t] = nsr;
                double sum = 0;
                for (size_t r = 0; r < p->nrows; r++) sum += out[r];
                uint64_t h = fnv1a(out, p->nrows * sizeof(float));
                if (t == 0) chk[ai][pr] = h;
                printf("trial %d %-6s %-4s ns/row=%8.1f sum=%12.4f %s\n",
                       t, ARMS[ai].name, p->tag, nsr, sum,
                       (h == chk[ai][pr]) ? "chk=STABLE" : "chk=DIFF!");
            }
        }
        fflush(stdout);
    }

#ifdef __linux__
    if (hw.ok) {
        double d_cyc = (double)(hwp_read(hw.cyc) - b_cyc);
        double d_ins = (double)(hwp_read(hw.ins) - b_ins);
        double d_cref = (double)(hwp_read(hw.cref) - b_cref);
        double d_cmiss = (double)(hwp_read(hw.cmiss) - b_cmiss);
        double d_br = (double)(hwp_read(hw.br) - b_br);
        double d_brm = (double)(hwp_read(hw.brm) - b_brm);
        double R = (double)total_rows;
        printf("SELF rows=%.0f ns_total=%.0f cyc/row=%.0f ins/row=%.0f "
               "IPC=%.3f miss/row=%.1f miss_pct=%.1f br_miss_pct=%.2f\n",
               R, (double)total_ns, d_cyc / R, d_ins / R, d_ins / d_cyc,
               d_cmiss / R, 100.0 * d_cmiss / (d_cref + 1),
               100.0 * d_brm / (d_br + 1));
    } else {
        printf("SELF unavailable (perf_event_open failed or HWP_SELF=0)\n");
    }
#endif

    if (mt4) {
        atomic_store(&MT.stop, 1);
        for (int t = 0; t < 4; t++) sem_post(&MT.wake);
        for (int t = 0; t < 4; t++) pthread_join(th[t], NULL);
    }

    // Summary: min + median per arm/proj, derived ms/token
    printf("\nsummary (%s, %d trials):\n", mt4 ? "mt4" : "st", trials);
    for (int a = 0; a < (int)NARMS; a++) {
        if (strcmp(only, "all") && strcmp(only, ARMS[a].name)) continue;
        if (!P[a][0].buf) continue;
        double mn[3], md[3];
        for (int pr = 0; pr < 3; pr++) {
            double tmp[32];
            for (int t = 0; t < trials; t++) tmp[t] = res[a][pr][t];
            // insertion sort for median
            for (int i = 1; i < trials; i++) {
                double k = tmp[i];
                int j = i - 1;
                while (j >= 0 && tmp[j] > k) { tmp[j + 1] = tmp[j]; j--; }
                tmp[j + 1] = k;
            }
            mn[pr] = tmp[0];
            md[pr] = tmp[trials / 2];
        }
        // rows/token: gate/up 512*K*40, down 2048*K*40
        double rg = 512.0 * ARMS[a].K * 40, rd = 2048.0 * ARMS[a].K * 40;
        double ms_min = (rg * mn[0] + rg * mn[1] + rd * mn[2]) / 1e6;
        double ms_med = (rg * md[0] + rg * md[1] + rd * md[2]) / 1e6;
        printf("%-6s K=%d gate=%7.1f/%7.1f up=%7.1f/%7.1f down=%7.1f/%7.1f "
               "ms/tok(min/med)=%7.1f/%7.1f\n",
               ARMS[a].name, ARMS[a].K, mn[0], md[0], mn[1], md[1], mn[2], md[2],
               ms_min, ms_med);
    }
    return 0;
}
