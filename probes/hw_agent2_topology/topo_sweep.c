// topo_sweep: thread/affinity/partition sweep on the REAL expert workload.
// Workload = one full MoE layer: 8 experts x (gate 512x2048 IQ2_XXS +
// up 512x2048 IQ2_XXS + SiLU/mul + down 2048x512 IQ2_S), real weight bytes
// from unsloth/Qwen3.5-35B-A3B-UD-IQ2_XXS (L00 E000..E007), real ggml kernels
// from pinned llama.cpp 3057bb66, 32 MB streaming pools (decode-like DRAM
// regime, same as agent1's validated setup), persistent pthreads + barriers
// (ggml threadpool style).
//
// Usage: topo_sweep <nth 1..8> <row|expert> <free|pin> [trials=40] [refbase] [hot|stream=stream] [futex|spin=futex]
//   refbase != "" enables parity check of every trial vs <refbase>.bin.
//   hot:    same pool offsets every trial (L3-resident best case)
//   stream: non-overlapping pool offsets per trial (DRAM-streaming decode case)
//   futex:  pthread_barrier (blocks; like ggml poll=0 wait path)
//   spin:   sense-reversing spinning barrier (like ggml --poll busy-wait)
// Usage: topo_sweep --dump-ref <refbase>   (ST row/free reference outputs)
//
// Stdout: one CSV line per run. Stderr: human-readable detail.
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <immintrin.h>
#include <pmmintrin.h>

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void quantize_row_q8_K(const float * x, void * y, int64_t k);

#define NEXP 8
#define DGATE 512
#define DIN 2048
#define GATEB 270336
#define DOWNB 335872
#define ROWB_GU 528
#define ROWB_DN 164

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static int cmp_dbl(const void * a, const void * b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

// ---- pools (streaming copies of real bytes) ----
static uint8_t *pool_g, *pool_u, *pool_d; // 32 MB each
static size_t ng_copies, nd_copies;
#define POOL (32u << 20)

static float *g_x;            // [DIN] per-trial activation
static uint8_t *g_xq;         // q8_K(DIN)
static float *g_out;          // [DIN] accumulated MoE output
static float *g_gout, *g_uout, *g_h;   // [NEXP][DGATE] scratch
static uint8_t *g_hq;         // [NEXP] q8_K(DGATE)
static float *g_dout;         // [NEXP][DIN] per-expert down outputs
static float g_w[NEXP];       // fixed router weights (uniform; same all cfgs)

static pthread_barrier_t bar;
static int g_nth = 1;
// sense-reversing spinning barrier (ggml-poll analog)
static atomic_int spin_cnt;
static atomic_int spin_sense;
static _Thread_local int tl_sense = 0;
static int g_spin = 0;
static inline void BAR(void) {
    if (!g_spin) { pthread_barrier_wait(&bar); return; }
    int s = tl_sense;
    if (atomic_fetch_add(&spin_cnt, 1) == g_nth - 1) {
        atomic_store(&spin_cnt, 0);
        atomic_store_explicit(&spin_sense, 1 - s, memory_order_release);
    } else {
        while (atomic_load_explicit(&spin_sense, memory_order_acquire) == s)
            _mm_pause();
    }
    tl_sense = 1 - s;
}
static int g_part_row = 1; // 1=row-split, 0=expert-split
static int g_hot = 0; // 1=same pool offsets every trial (L3-resident)
static inline size_t poff(long trial, int e, size_t ncp) {
    if (g_hot) return (size_t)e % ncp;
    return (size_t)((trial * NEXP + e) % (long)ncp);
}
static double *g_busy;       // per-worker busy ns accumulator
static int *g_cpu0;           // per-worker first cpu seen
static long *g_mig;           // per-worker cpu transitions
static int *g_lastcpu;

static void do_gate_up_rows(int ith, int nth, int e, const uint8_t *xq, float *out, const uint8_t *pool, size_t ncp, size_t msz, long trial) {
    int r0 = DGATE * ith / nth, r1 = DGATE * (ith + 1) / nth;
    const uint8_t *w = pool + poff(trial, e, ncp) * msz;
    for (int r = r0; r < r1; r++)
        ggml_vec_dot_iq2_xxs_q8_K(DIN, &out[e * DGATE + r], 0, w + r * ROWB_GU, 0, xq, 0, 1);
}
static void do_down_rows(int ith, int nth, int e, const uint8_t *hq, float *out, long trial) {
    int r0 = DIN * ith / nth, r1 = DIN * (ith + 1) / nth;
    const uint8_t *w = pool_d + poff(trial, e, nd_copies) * DOWNB;
    for (int r = r0; r < r1; r++)
        ggml_vec_dot_iq2_s_q8_K(DGATE, &out[e * DIN + r], 0, w + r * ROWB_DN, 0, hq, 0, 1);
}

// one worker's share of one trial; returns busy ns (excludes barrier wait)
static double run_share(int ith, int nth, long trial) {
    double busy = 0, s;
    if (g_part_row) {
        for (int e = 0; e < NEXP; e++) {
            s = now_ns();
            do_gate_up_rows(ith, nth, e, g_xq, g_gout, pool_g, ng_copies, GATEB, trial);
            busy += now_ns() - s;
            BAR();
            // up GEMV rows (same partition)
            s = now_ns();
            { int r0 = DGATE * ith / nth, r1 = DGATE * (ith + 1) / nth;
              const uint8_t *w = pool_u + poff(trial, e, ng_copies) * GATEB;
              for (int r = r0; r < r1; r++)
                  ggml_vec_dot_iq2_xxs_q8_K(DIN, &g_uout[e * DGATE + r], 0, w + r * ROWB_GU, 0, g_xq, 0, 1); }
            busy += now_ns() - s;
            BAR();
            // SiLU+mul on my row share (all threads participate)
            { int r0 = DGATE * ith / nth, r1 = DGATE * (ith + 1) / nth;
              for (int r = r0; r < r1; r++) {
                  float g = g_gout[e * DGATE + r];
                  g_h[e * DGATE + r] = (g / (1.0f + expf(-g))) * g_uout[e * DGATE + r];
              } }
            BAR();
            if (ith == 0) quantize_row_q8_K(g_h + e * DGATE, g_hq + (size_t)e * 2 * 292, DGATE);
            BAR();
            s = now_ns();
            do_down_rows(ith, nth, e, g_hq + (size_t)e * 2 * 292, g_dout, trial);
            busy += now_ns() - s;
            BAR();
        }
    } else {
        // expert-split: whole experts per worker, one barrier at trial end
        int e0 = NEXP * ith / nth, e1 = NEXP * (ith + 1) / nth;
        s = now_ns();
        for (int e = e0; e < e1; e++) {
            const uint8_t *wg = pool_g + poff(trial, e, ng_copies) * GATEB;
            const uint8_t *wu = pool_u + poff(trial, e, ng_copies) * GATEB;
            for (int r = 0; r < DGATE; r++) {
                ggml_vec_dot_iq2_xxs_q8_K(DIN, &g_gout[e * DGATE + r], 0, wg + r * ROWB_GU, 0, g_xq, 0, 1);
                ggml_vec_dot_iq2_xxs_q8_K(DIN, &g_uout[e * DGATE + r], 0, wu + r * ROWB_GU, 0, g_xq, 0, 1);
            }
            for (int r = 0; r < DGATE; r++) {
                float g = g_gout[e * DGATE + r];
                g_h[e * DGATE + r] = (g / (1.0f + expf(-g))) * g_uout[e * DGATE + r];
            }
            quantize_row_q8_K(g_h + e * DGATE, g_hq + (size_t)e * 2 * 292, DGATE);
            const uint8_t *wd = pool_d + poff(trial, e, nd_copies) * DOWNB;
            for (int r = 0; r < DIN; r++)
                ggml_vec_dot_iq2_s_q8_K(DGATE, &g_dout[e * DIN + r], 0, wd + r * ROWB_DN, 0, g_hq + (size_t)e * 2 * 292, 0, 1);
        }
        busy += now_ns() - s;
        BAR();
    }
    return busy;
}

static void sample_cpu(int ith) {
    int c = sched_getcpu();
    if (g_lastcpu[ith] < 0) { g_lastcpu[ith] = c; g_cpu0[ith] = c; }
    else if (c != g_lastcpu[ith]) { g_mig[ith]++; g_lastcpu[ith] = c; }
}

static _Atomic long g_iter;
static _Atomic int g_exit;

struct warg { int ith, nth; };
static void * worker(void * v) {
    struct warg * a = v;
    int ith = a->ith, nth = a->nth;
    for (;;) {
        BAR(); // ready
        if (g_exit) break;
        sample_cpu(ith);
        g_busy[ith] += run_share(ith, nth, g_iter);
        BAR(); // done
    }
    return NULL;
}

static long g_vmhwm_kb = -1;
static void read_ctxt(long * vol, long * invol) {
    *vol = *invol = -1;
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "voluntary_ctxt_switches:", 24)) *vol = atol(line + 24);
        else if (!strncmp(line, "nonvoluntary_ctxt_switches:", 27)) *invol = atol(line + 27);
        else if (!strncmp(line, "VmHWM:", 6)) g_vmhwm_kb = atol(line + 6);
    }
    fclose(f);
}

static int load_file(const char * path, uint8_t * dst, size_t n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return 0; }
    size_t r = fread(dst, 1, n, f);
    fclose(f);
    return r == n;
}

static void make_activation(long trial) {
    uint64_t rng = 1000 + (uint64_t)trial * 7919ULL;
    for (int i = 0; i < DIN; i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        g_x[i] = (float)(((rng >> 33) / 4294967296.0) - 0.5);
    }
    quantize_row_q8_K(g_x, g_xq, DIN);
}

static void reduce_out(void) {
    for (int i = 0; i < DIN; i++) {
        double acc = 0;
        for (int e = 0; e < NEXP; e++) acc += (double)g_dout[e * DIN + i] * g_w[e];
        g_out[i] = (float)acc;
    }
}

static int g_dump_ref = 0;
static const char * g_refbase = NULL;

int main(int argc, char ** argv) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    int nth = 4, trials = 40, pin = 0;
    if (argc >= 2 && !strcmp(argv[1], "--dump-ref")) {
        if (argc < 3) { fprintf(stderr, "usage: %s --dump-ref <refbase>\n", argv[0]); return 2; }
        g_dump_ref = 1; g_refbase = argv[2];
        nth = 1; g_part_row = 1; trials = 40;
        if (argc >= 4 && !strcmp(argv[3], "hot")) g_hot = 1;
        if (argc >= 5 && !strcmp(argv[4], "spin")) g_spin = 1;
    } else {
        if (argc < 4) { fprintf(stderr, "usage: %s <nth 1..8> <row|expert> <free|pin> [trials=40] [refbase]\n", argv[0]); return 2; }
        nth = atoi(argv[1]);
        if (!strcmp(argv[2], "row")) g_part_row = 1;
        else if (!strcmp(argv[2], "expert")) g_part_row = 0;
        else { fprintf(stderr, "bad part\n"); return 2; }
        if (!strcmp(argv[3], "pin")) pin = 1;
        else if (!strcmp(argv[3], "free")) pin = 0;
        else { fprintf(stderr, "bad aff\n"); return 2; }
        if (argc >= 5) trials = atoi(argv[4]);
        if (argc >= 6) g_refbase = argv[5];
        if (argc >= 7) {
            if (!strcmp(argv[6], "hot")) g_hot = 1;
            else if (!strcmp(argv[6], "stream")) g_hot = 0;
            else { fprintf(stderr, "bad poolmode\n"); return 2; }
        }
        if (argc >= 8) {
            if (!strcmp(argv[7], "spin")) g_spin = 1;
            else if (!strcmp(argv[7], "futex")) g_spin = 0;
            else { fprintf(stderr, "bad barrier\n"); return 2; }
        }
        if (nth < 1 || nth > 8 || trials < 1) return 2;
    }
    g_nth = nth;

    // usable cpu list (respects outer taskset)
    cpu_set_t avail;
    CPU_ZERO(&avail);
    sched_getaffinity(0, sizeof avail, &avail);
    int cpus[64]; int ncpu = 0;
    for (int c = 0; c < 64 && ncpu < 64; c++) if (CPU_ISSET(c, &avail)) cpus[ncpu++] = c;
    if (ncpu == 0) return 1;

    // load real expert bytes
    static uint8_t eg[NEXP][GATEB], eu[NEXP][GATEB], ed[NEXP][DOWNB];
    char path[256];
    for (int e = 0; e < NEXP; e++) {
        snprintf(path, sizeof path, "/tmp/agent1_raw/L00_E%03d_gate.iq2xxs", e);
        if (!load_file(path, eg[e], GATEB)) return 1;
        snprintf(path, sizeof path, "/tmp/agent1_raw/L00_E%03d_up.iq2xxs", e);
        if (!load_file(path, eu[e], GATEB)) return 1;
        snprintf(path, sizeof path, "/tmp/agent1_raw/L00_E%03d_down.iq2s", e);
        if (!load_file(path, ed[e], DOWNB)) return 1;
    }
    pool_g = aligned_alloc(64, POOL); pool_u = aligned_alloc(64, POOL); pool_d = aligned_alloc(64, POOL);
    ng_copies = POOL / GATEB; nd_copies = POOL / DOWNB;
    for (size_t c = 0; c < ng_copies; c++) {
        memcpy(pool_g + c * GATEB, eg[c % NEXP], GATEB);
        memcpy(pool_u + c * GATEB, eu[c % NEXP], GATEB);
    }
    for (size_t c = 0; c < nd_copies; c++) memcpy(pool_d + c * DOWNB, ed[c % NEXP], DOWNB);

    g_x = aligned_alloc(64, DIN * sizeof(float));
    g_xq = aligned_alloc(64, 8 * 292);
    g_out = aligned_alloc(64, DIN * sizeof(float));
    g_gout = aligned_alloc(64, (size_t)NEXP * DGATE * sizeof(float));
    g_uout = aligned_alloc(64, (size_t)NEXP * DGATE * sizeof(float));
    g_h = aligned_alloc(64, (size_t)NEXP * DGATE * sizeof(float));
    g_hq = aligned_alloc(64, (size_t)NEXP * 2 * 292);
    g_dout = aligned_alloc(64, (size_t)NEXP * DIN * sizeof(float));
    for (int e = 0; e < NEXP; e++) g_w[e] = 1.0f / NEXP;
    g_busy = calloc(nth, sizeof(double));
    g_cpu0 = malloc(nth * sizeof(int));
    g_mig = calloc(nth, sizeof(long));
    g_lastcpu = malloc(nth * sizeof(int));
    for (int i = 0; i < nth; i++) g_lastcpu[i] = -1;

    pthread_barrier_init(&bar, NULL, nth);
    atomic_init(&spin_cnt, 0);
    atomic_init(&spin_sense, 0);
    pthread_t * th = malloc(nth * sizeof(pthread_t));
    struct warg * wa = malloc(nth * sizeof(struct warg));
    for (int i = 1; i < nth; i++) {
        wa[i].ith = i; wa[i].nth = nth;
        pthread_create(&th[i], NULL, worker, &wa[i]);
    }
    // pin AFTER create (main = worker 0)
    if (pin) {
        for (int i = 0; i < nth; i++) {
            cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpus[i % ncpu], &s);
            pthread_t t = (i == 0) ? pthread_self() : th[i];
            int rv = pthread_setaffinity_np(t, sizeof s, &s);
            if (rv) fprintf(stderr, "pin worker %d -> cpu %d failed: %s\n", i, cpus[i % ncpu], strerror(rv));
        }
    }

    // reference outputs for parity
    float * ref = NULL;
    if (g_dump_ref || g_refbase) {
        ref = aligned_alloc(64, (size_t)trials * DIN * sizeof(float));
        if (!g_dump_ref && g_refbase) {
            char pb[300]; snprintf(pb, sizeof pb, "%s.bin", g_refbase);
            FILE * f = fopen(pb, "rb");
            if (!f) { fprintf(stderr, "ref %s missing\n", pb); return 1; }
            if (fread(ref, sizeof(float), (size_t)trials * DIN, f) != (size_t)trials * DIN) {
                fprintf(stderr, "ref short\n"); return 1;
            }
            fclose(f);
        }
    }

    // warmup
    for (int w = 0; w < 5; w++) {
        make_activation(100000 + w);
        g_iter = 100000 + w;
        BAR();
        run_share(0, nth, g_iter);
        BAR();
        reduce_out();
    }
    memset(g_busy, 0, nth * sizeof(double));
    for (int i = 0; i < nth; i++) { g_lastcpu[i] = -1; g_mig[i] = 0; }

    double * tm = malloc(trials * sizeof(double));
    uint64_t * hh = malloc(trials * sizeof(uint64_t));
    long v0, i0, v1, i1;
    struct rusage ru0, ru1;
    getrusage(RUSAGE_SELF, &ru0);
    read_ctxt(&v0, &i0);
    double maxdiff = 0;
    int nbad = 0;
    for (int t = 0; t < trials; t++) {
        make_activation(t);
        g_iter = t;
        BAR(); // ready
        sample_cpu(0);
        double s = now_ns();
        g_busy[0] += run_share(0, nth, t);
        BAR(); // done (includes stragglers)
        double e = now_ns();
        tm[t] = e - s;
        reduce_out();
        hh[t] = fnv1a(g_out, (size_t)DIN * sizeof(float));
        if (g_dump_ref) memcpy(ref + (size_t)t * DIN, g_out, (size_t)DIN * sizeof(float));
        else if (ref) {
            for (int i = 0; i < DIN; i++) {
                double d = fabs((double)g_out[i] - ref[(size_t)t * DIN + i]);
                if (d > maxdiff) maxdiff = d;
                if (d != 0.0) nbad++;
            }
        }
    }
    read_ctxt(&v1, &i1);
    getrusage(RUSAGE_SELF, &ru1);

    g_exit = 1;
    BAR();
    for (int i = 1; i < nth; i++) pthread_join(th[i], NULL);

    if (g_dump_ref) {
        char pb[300], ph[300];
        snprintf(pb, sizeof pb, "%s.bin", g_refbase);
        snprintf(ph, sizeof ph, "%s.u64", g_refbase);
        FILE * f = fopen(pb, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", pb); return 1; }
        fwrite(ref, sizeof(float), (size_t)trials * DIN, f); fclose(f);
        f = fopen(ph, "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", ph); return 1; }
        for (int t = 0; t < trials; t++) fprintf(f, "%016lx\n", (unsigned long)hh[t]);
        fclose(f);
        fprintf(stderr, "dumped %d trials ref -> %s.{bin,u64}\n", trials, g_refbase);
    }

    double * ts = malloc(trials * sizeof(double));
    memcpy(ts, tm, trials * sizeof(double));
    qsort(ts, trials, sizeof(double), cmp_dbl);
    double med = ts[trials / 2], p10 = ts[trials / 10], p90 = ts[(trials * 9) / 10], mn = ts[0];
    double mean = 0; for (int t = 0; t < trials; t++) mean += tm[t]; mean /= trials;
    double sd = 0; for (int t = 0; t < trials; t++) sd += (tm[t] - mean) * (tm[t] - mean);
    sd = sqrt(sd / trials);
    double bmax = 0, bmin = 1e30, bsum = 0;
    for (int i = 0; i < nth; i++) {
        if (g_busy[i] > bmax) bmax = g_busy[i];
        if (g_busy[i] < bmin) bmin = g_busy[i];
        bsum += g_busy[i];
    }
    long mig = 0; for (int i = 0; i < nth; i++) mig += g_mig[i];
    // CSV to stdout
    printf("%d,%s,%s,%s,%s,%d,%.0f,%.0f,%.0f,%.0f,%.0f,%.4f,%.3f,%ld,%ld,%ld,%ld,%ld,%.6f,%d\n",
        nth, g_part_row ? "row" : "expert", pin ? "pin" : "free",
        g_hot ? "hot" : "stream", g_spin ? "spin" : "futex", trials,
        med, p10, p90, mn, mean, sd / mean, bmax / (bmin > 0 ? bmin : 1),
        mig, v1 - v0, i1 - i0,
        ru1.ru_minflt - ru0.ru_minflt, ru1.ru_majflt - ru0.ru_majflt,
        maxdiff, nbad);
    fflush(stdout);
    fprintf(stderr, "nth=%d part=%s aff=%s pool=%s bar=%s ncpu=%d trials=%d med=%.3fms p10=%.3f p90=%.3f min=%.3f cv=%.3f bal=%.3f mig=%ld ctxt=%ld/%ld minflt=%ld maxdiff=%.3g nbad=%d y0=%.6f ylast=%.6f\n",
        nth, g_part_row ? "row" : "expert", pin ? "pin" : "free",
        g_hot ? "hot" : "stream", g_spin ? "spin" : "futex", ncpu, trials,
        med / 1e6, p10 / 1e6, p90 / 1e6, mn / 1e6, sd / mean, bmax / (bmin > 0 ? bmin : 1),
        mig, v1 - v0, i1 - i0, ru1.ru_minflt - ru0.ru_minflt, maxdiff, nbad, g_out[0], g_out[DIN - 1]);
    fprintf(stderr, "vmhwm_kb=%ld\n", g_vmhwm_kb);
    return 0;
}

