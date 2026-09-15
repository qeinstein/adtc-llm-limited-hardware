/* kreplay: exact-kernel steady-state decode replay for Qwen3.5-35B-A3B-UD-IQ2_XXS.
 *
 * What it is: calls the SAME ggml-cpu kernel entry points the bounded
 * staged executor (v10) executes at decode, with REAL packed weights
 * (range-fetched from the pinned GGUF) and REAL activation rows:
 *   gate/up : ggml_vec_dot_iq2_xxs_q8_K   (AVX2, arch/x86/quants.c)
 *   down    : ggml_vec_dot_iq2_s_q8_K
 *   dense   : ggml_vec_dot_q4/q5/q6_K_q8_K, ggml_vec_dot_f32 (router)
 *   act     : quantize_row_q8_K (same sharded flow as mul_mat_id)
 * Threading mirrors ggml_compute_forward_mul_mat_id at cne1==1, nth=4:
 *   rows<=512: static 4-way shard; rows>512: atomic 64-row chunks.
 * Build: same source (pinned llama.cpp @3057bb66) and flags as the
 * bounded build (-O3 -march=native). Link libggml-cpu.a + libggml-base.a.
 *
 * Modes:
 *   expert <routes.txt> <slicedir> <act.npy>   40 layers x top8 replay
 *   dense  <densedir>   <act.npy>               per-token dense GEMVs
 *   mach   <slotfile>                             bounded-cache machinery
 * Env: KREPLAY_THREADS (default 4), KREPLAY_EVICT (default 1: evict L3
 * between layers), KREPLAY_REPS (default 1).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/resource.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "quants.h"
#include "vec.h"

#define N_LAYER 40
#define N_TOPK 8
#define N_EMBD 2048
#define N_FF 512

static int NTH = 4;
static int EVICT = 1;

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void * load_file(const char * path, size_t * n) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s failed\n", path); exit(1); }
    struct stat st; fstat(fd, &st);
    void * p = malloc(st.st_size ? st.st_size : 1);
    size_t off = 0;
    while (off < (size_t) st.st_size) {
        ssize_t r = read(fd, (char *) p + off, st.st_size - off);
        if (r <= 0) { fprintf(stderr, "read %s failed\n", path); exit(1); }
        off += r;
    }
    close(fd);
    if (n) *n = st.st_size;
    return p;
}

/* ---- L3 eviction buffer: dense trunk (~1.6GB) + expert churn evict everything ---- */
static float * evict_buf = NULL;
static size_t evict_n = 48 << 20; /* floats: 192MB stream >> 6MB L3 */
static volatile double evict_sink = 0;
static void evict_l3(void) {
    if (!EVICT) return;
    double s = 0;
    for (size_t i = 0; i < evict_n; i += 16) s += evict_buf[i];
    evict_sink = s;
}

/* ---- minimal persistent thread pool (mirrors ggml persistent threadpool) ---- */
typedef void (*shard_fn)(int ith, void * ctx);

struct pool {
    pthread_t * th;
    int * ids;
    pthread_barrier_t bar;
    shard_fn fn;
    void * ctx;
    _Atomic int stop;
    int n;
};

struct warg { struct pool * p; int ith; };

static void * pool_worker(void * arg) {
    struct warg * wa = arg;
    struct pool * p = wa->p;
    int ith = wa->ith;
    for (;;) {
        pthread_barrier_wait(&p->bar); /* wait for job */
        if (atomic_load(&p->stop)) break;
        p->fn(ith, p->ctx);
        pthread_barrier_wait(&p->bar); /* job done */
    }
    return NULL;
}

static struct pool * pool_create(int n) {
    struct pool * p = calloc(1, sizeof(*p));
    p->n = n - 1;
    p->th = calloc(p->n, sizeof(pthread_t));
    p->ids = calloc(p->n, sizeof(int));
    static struct warg wa[64];
    pthread_barrier_init(&p->bar, NULL, n);
    for (int i = 0; i < p->n; i++) {
        wa[i].p = p; wa[i].ith = i + 1;
        pthread_create(&p->th[i], NULL, pool_worker, &wa[i]);
    }
    return p;
}

static void pool_run(struct pool * p, shard_fn fn, void * ctx) {
    p->fn = fn; p->ctx = ctx;
    /* release workers; main is ith=0 */
    pthread_barrier_wait(&p->bar);
    fn(0, ctx);
    pthread_barrier_wait(&p->bar);
}

static void pool_free(struct pool * p) {
    atomic_store(&p->stop, 1);
    pthread_barrier_wait(&p->bar);
    for (int i = 0; i < p->n; i++) pthread_join(p->th[i], NULL);
    pthread_barrier_destroy(&p->bar);
    free(p->th); free(p);
}

/* ---- GEMV job ---- */
typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

struct gemv_job {
    vec_dot_fn fn;
    const char * w;      /* packed weights base */
    const void * xq;     /* quantized activation row */
    float * y;           /* output rows */
    int64_t ncols;
    int64_t nrows;
    size_t rowbytes;
    _Atomic int chunk;
    int nchunk;
    int dynamic;
};

static void gemv_shard(int ith, void * vctx) {
    struct gemv_job * j = vctx;
    if (!j->dynamic) {
        int64_t r0 = (j->nrows * ith) / NTH;
        int64_t r1 = (j->nrows * (ith + 1)) / NTH;
        for (int64_t r = r0; r < r1; r++)
            j->fn(j->ncols, &j->y[r], 0, j->w + r * j->rowbytes, 0, j->xq, 0, 1);
    } else {
        for (;;) {
            int c = atomic_fetch_add(&j->chunk, 1);
            if (c >= j->nchunk) break;
            int64_t r0 = c * 64, r1 = r0 + 64;
            if (r1 > j->nrows) r1 = j->nrows;
            for (int64_t r = r0; r < r1; r++)
                j->fn(j->ncols, &j->y[r], 0, j->w + r * j->rowbytes, 0, j->xq, 0, 1);
        }
    }
}

static void run_gemv(struct pool * p, vec_dot_fn fn, const void * w, const void * xq,
                     float * y, int64_t ncols, int64_t nrows, size_t rowbytes) {
    struct gemv_job j = { fn, w, xq, y, ncols, nrows, rowbytes, 0, 0, 0 };
    /* mirror ggml chunking policy at cne1==1, nth=4 */
    int64_t nc = (nrows + 63) / 64;
    if (nc < NTH * 4) { j.dynamic = 0; }
    else { j.dynamic = 1; j.nchunk = (int) nc; }
    pool_run(p, gemv_shard, &j);
}

/* ---- quantize job (sharded over Q8_K blocks like ggml) ---- */
struct q_job { const float * x; char * y; int64_t n; size_t bs; };

static void quant_shard(int ith, void * vctx) {
    struct q_job * j = vctx;
    int64_t nb = j->n / 256, b0 = (nb * ith) / NTH, b1 = (nb * (ith + 1)) / NTH;
    if (b1 > b0)
        quantize_row_q8_K(j->x + b0 * 256, j->y + b0 * j->bs, (b1 - b0) * 256);
}

static void run_quant(struct pool * p, const float * x, void * y, int64_t n) {
    size_t bs = ggml_row_size(GGML_TYPE_Q8_K, 256);
    struct q_job j = { x, y, n, bs };
    pool_run(p, quant_shard, &j);
}

/* ---- elementwise: silu(g)*u and weighted accumulate, 4-way split ---- */
struct ew_job { const float * g, * u; float * z; int64_t n; };
static void swiglu_shard(int ith, void * vctx) {
    struct ew_job * j = vctx;
    int64_t r0 = (j->n * ith) / NTH, r1 = (j->n * (ith + 1)) / NTH;
    for (int64_t i = r0; i < r1; i++) {
        float g = j->g[i];
        j->z[i] = (g / (1.0f + expf(-g))) * j->u[i];
    }
}

struct acc_job { float * y; const float * d; float w; int64_t n; };
static void acc_shard(int ith, void * vctx) {
    struct acc_job * j = vctx;
    int64_t r0 = (j->n * ith) / NTH, r1 = (j->n * (ith + 1)) / NTH;
    for (int64_t i = r0; i < r1; i++) j->y[i] += j->w * j->d[i];
}

static uint64_t fnv1a(const float * p, int64_t n) {
    uint64_t h = 1469598103934665603ULL;
    const unsigned char * b = (const unsigned char *) p;
    for (int64_t i = 0; i < n * 4; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static double sumf(const float * p, int64_t n) {
    double s = 0; for (int64_t i = 0; i < n; i++) s += p[i]; return s;
}

/* steady-state sync: after init+preload, print READY and block until
 * hwpstat (HWP_SYNC=1) attaches counters and releases us. */
static void hwp_sync(void) {
    const char * fd = getenv("HWP_SYNC_FD");
    if (!fd) return;
    printf("READY\n"); fflush(stdout);
    char b = 0;
    if (read(atoi(fd), &b, 1) != 1) { fprintf(stderr, "sync read failed\n"); exit(1); }
    close(atoi(fd));
}

/* npy loader for float32 C-order 2D */
static float * load_npy(const char * path, int64_t * r, int64_t * c) {
    size_t n; unsigned char * b = load_file(path, &n);
    /* 10-byte header + dict; find shape tuple (memmem: magic has NUL) */
    char * sh = memmem(b, n < 256 ? n : 256, "'shape'", 7);
    if (!sh) { fprintf(stderr, "npy shape? %s\n", path); exit(1); }
    long long a = 0, d = 0;
    sscanf(sh, "'shape': (%lld, %lld", &a, &d);
    size_t off = 10 + (unsigned char) b[8] + ((unsigned char) b[9] << 8);
    float * out = malloc(a * d * 4);
    memcpy(out, b + off, a * d * 4);
    free(b);
    *r = a; *c = d;
    return out;
}

/* routes: 40 lines x 8 ints */
static int routes[N_LAYER][N_TOPK];
static void load_routes(const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) { fprintf(stderr, "open %s\n", path); exit(1); }
    for (int l = 0; l < N_LAYER; l++)
        for (int k = 0; k < N_TOPK; k++)
            if (fscanf(f, "%d", &routes[l][k]) != 1) { fprintf(stderr, "routes parse\n"); exit(1); }
    fclose(f);
}

static void * load_slice(const char * dir, int L, int E, const char * kind, const char * ext, size_t expect) {
    char p[512];
    snprintf(p, sizeof(p), "%s/L%02d_E%03d_%s.%s", dir, L, E, kind, ext);
    size_t n; void * b = load_file(p, &n);
    if (n != expect) { fprintf(stderr, "%s: got %zu want %zu\n", p, n, expect); exit(1); }
    return b;
}

/* ================= EXPERT MODE ================= */
static int run_expert(const char * routes_path, const char * slicedir, const char * act_path) {
    load_routes(routes_path);
    int64_t AR, AC;
    float * act = load_npy(act_path, &AR, &AC);
    if (AC != N_EMBD) { fprintf(stderr, "act cols %lld\n", (long long) AC); return 1; }

    struct pool * pool = pool_create(NTH);
    size_t q8_2048 = ggml_row_size(GGML_TYPE_Q8_K, N_EMBD);
    size_t q8_512 = ggml_row_size(GGML_TYPE_Q8_K, N_FF);
    void * xq = malloc(q8_2048), * zq = malloc(q8_512);
    float * g = malloc(N_FF * 4), * u = malloc(N_FF * 4), * z = malloc(N_FF * 4);
    float * d = malloc(N_EMBD * 4), * y = malloc(N_EMBD * 4);

    /* preload all slices (page-cache warm, CPU-cache cold after evict) */
    double t_pre = now_s();
    void * W[N_LAYER][N_TOPK][3];
    for (int L = 0; L < N_LAYER; L++)
        for (int k = 0; k < N_TOPK; k++) {
            int E = routes[L][k];
            W[L][k][0] = load_slice(slicedir, L, E, "gate", "iq2xxs", 270336);
            W[L][k][1] = load_slice(slicedir, L, E, "up", "iq2xxs", 270336);
            W[L][k][2] = load_slice(slicedir, L, E, "down", "iq2s", 335872);
        }
    fprintf(stderr, "expert: preloaded 40x8 bundles (%.1f MB) in %.1f ms\n",
        40.0 * 8 * 876544 / 1048576.0, (now_s() - t_pre) * 1000);

    double t_q = 0, t_gate = 0, t_up = 0, t_ew = 0, t_qz = 0, t_down = 0, t_acc = 0, t_evict = 0;
    int reps = getenv("KREPLAY_REPS") ? atoi(getenv("KREPLAY_REPS")) : 1;
    uint64_t ck = 0;
    hwp_sync();
    struct rusage ru0, ru1;
    getrusage(RUSAGE_SELF, &ru0);
    double wall0 = now_s();
    for (int rep = 0; rep < reps; rep++) {
        for (int L = 0; L < N_LAYER; L++) {
            double te = now_s();
            evict_l3();
            t_evict += now_s() - te;
            void * (*slot)[3] = W[L];
            memset(y, 0, N_EMBD * 4);
            const float * x = act + ((L * N_TOPK) % AR) * N_EMBD;
            { /* x is L1-hot in reality (just produced by attention): touch, untimed */
                volatile float s = 0;
                for (int i = 0; i < N_EMBD; i += 16) s += x[i];
            }
            for (int k = 0; k < N_TOPK; k++) {
                double s;
                s = now_s(); run_quant(pool, x, xq, N_EMBD); t_q += now_s() - s;
                s = now_s(); run_gemv(pool, ggml_vec_dot_iq2_xxs_q8_K, slot[k][0], xq, g, N_EMBD, N_FF, 528); t_gate += now_s() - s;
                s = now_s(); run_quant(pool, x, xq, N_EMBD); t_q += now_s() - s;
                s = now_s(); run_gemv(pool, ggml_vec_dot_iq2_xxs_q8_K, slot[k][1], xq, u, N_EMBD, N_FF, 528); t_up += now_s() - s;
                struct ew_job ew = { g, u, z, N_FF };
                s = now_s(); pool_run(pool, swiglu_shard, &ew); t_ew += now_s() - s;
                s = now_s(); run_quant(pool, z, zq, N_FF); t_qz += now_s() - s;
                s = now_s(); run_gemv(pool, ggml_vec_dot_iq2_s_q8_K, slot[k][2], zq, d, N_FF, N_EMBD, 164); t_down += now_s() - s;
                struct acc_job ac = { y, d, 1.0f / N_TOPK, N_EMBD };
                s = now_s(); pool_run(pool, acc_shard, &ac); t_acc += now_s() - s;
            }
            ck ^= fnv1a(y, N_EMBD);
        }
    }
    double wall = now_s() - wall0;
    getrusage(RUSAGE_SELF, &ru1);
    double cpu = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) +
        (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec) / 1e6 +
        (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) +
        (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1e6;
    printf("rusage_cpu=%.3f wall=%.3f cpus=%.2f minflt=%ld majflt=%ld nvcsw=%ld nivcsw=%ld\n",
        cpu, wall, cpu / wall, ru1.ru_minflt - ru0.ru_minflt, ru1.ru_majflt - ru0.ru_majflt,
        ru1.ru_nvcsw - ru0.ru_nvcsw, ru1.ru_nivcsw - ru0.ru_nivcsw);
    double kern = t_q + t_gate + t_up + t_ew + t_qz + t_down + t_acc;
    double ysum = sumf(y, N_EMBD);
    printf("mode=expert reps=%d threads=%d evict=%d actrows=%lld\n", reps, NTH, EVICT, (long long) AR);
    printf("per_token_ms: quant_x=%.3f gate=%.3f up=%.3f silu_mul=%.3f quant_z=%.3f down=%.3f wacc=%.3f kernel_sum=%.3f evict=%.3f wall=%.3f\n",
        t_q / reps * 1000, t_gate / reps * 1000, t_up / reps * 1000,
        t_ew / reps * 1000, t_qz / reps * 1000, t_down / reps * 1000, t_acc / reps * 1000,
        kern / reps * 1000, t_evict / reps * 1000, wall / reps * 1000);
    printf("note: quant_x covers gate+up quants (2x2048/layer-expert); quant_z covers down (512)\n");
    printf("checksum=%016llx ysum_last=%.6f\n", (unsigned long long) ck, ysum);
    pool_free(pool);
    return 0;
}

/* ================= DENSE MODE ================= */
static void * load_dense(const char * dir, const char * name, size_t expect) {
    char p[512];
    snprintf(p, sizeof(p), "%s/%s.bin", dir, name);
    size_t n; void * b = load_file(p, &n);
    if (n != expect) { fprintf(stderr, "%s: got %zu want %zu\n", p, n, expect); exit(1); }
    return b;
}

struct dop {
    const char * name; const char * file; vec_dot_fn fn; enum ggml_type ty;
    int64_t ncols, nrows; int count; size_t bytes;
};

static int run_dense(const char * densedir, const char * act_path) {
    int64_t AR, AC;
    float * act = load_npy(act_path, &AR, &AC);
    struct pool * pool = pool_create(NTH);
    /* 4096-wide activations: concat of two real rows */
    float * act4096 = malloc(AR / 2 * 4096 * 4);
    for (int64_t i = 0; i < AR / 2; i++) {
        memcpy(act4096 + i * 4096, act + (2 * i) * 2048, 2048 * 4);
        memcpy(act4096 + i * 4096 + 2048, act + (2 * i + 1) * 2048, 2048 * 4);
    }
    struct dop ops[] = {
        { "attn_q",  "blk_3_attn_q_weight",  ggml_vec_dot_q5_K_q8_K, GGML_TYPE_Q5_K, 2048, 8192, 10, 11534336 },
        { "attn_k",  "blk_3_attn_k_weight",  ggml_vec_dot_q5_K_q8_K, GGML_TYPE_Q5_K, 2048, 512, 10, 720896 },
        { "attn_v",  "blk_3_attn_v_weight",  ggml_vec_dot_q5_K_q8_K, GGML_TYPE_Q5_K, 2048, 512, 10, 720896 },
        { "attn_o",  "blk_3_attn_output_weight", ggml_vec_dot_q5_K_q8_K, GGML_TYPE_Q5_K, 4096, 2048, 10, 5767168 },
        { "gdn_qkv", "blk_0_attn_qkv_weight", ggml_vec_dot_q5_K_q8_K, GGML_TYPE_Q5_K, 2048, 8192, 30, 11534336 },
        { "gdn_gate","blk_0_attn_gate_weight", ggml_vec_dot_q5_K_q8_K, GGML_TYPE_Q5_K, 2048, 4096, 30, 5767168 },
        { "ssm_out", "blk_0_ssm_out_weight", ggml_vec_dot_q6_K_q8_K, GGML_TYPE_Q6_K, 4096, 2048, 30, 6881280 },
        { "sh_gate", "blk_0_ffn_gate_shexp_weight", ggml_vec_dot_q5_K_q8_K, GGML_TYPE_Q5_K, 2048, 512, 40, 720896 },
        { "sh_up",   "blk_0_ffn_up_shexp_weight", ggml_vec_dot_q5_K_q8_K, GGML_TYPE_Q5_K, 2048, 512, 40, 720896 },
        { "sh_down", "blk_0_ffn_down_shexp_weight", ggml_vec_dot_q6_K_q8_K, GGML_TYPE_Q6_K, 512, 2048, 40, 860160 },
        { "head",    "output_weight", ggml_vec_dot_q4_K_q8_K, GGML_TYPE_Q4_K, 2048, 248320, 1, 286064640 },
    };
    int nops = sizeof(ops) / sizeof(ops[0]);
    void * W[16];
    int have_head = 1;
    for (int i = 0; i < nops; i++) {
        char p[512]; snprintf(p, sizeof(p), "%s/%s.bin", densedir, ops[i].file);
        int fd = open(p, O_RDONLY);
        if (fd < 0 && !strcmp(ops[i].name, "head")) { have_head = 0; W[i] = NULL; continue; }
        if (fd < 0) { fprintf(stderr, "missing %s\n", p); return 1; }
        close(fd);
        W[i] = load_dense(densedir, ops[i].file, ops[i].bytes);
    }
    float * router = load_dense(densedir, "blk_0_ffn_gate_inp_weight", 2097152);

    size_t q8max = ggml_row_size(GGML_TYPE_Q8_K, 4096);
    void * xq = malloc(q8max);
    float * y = malloc(248320 * 4);
    float * g512 = malloc(512 * 4), * u512 = malloc(512 * 4), * z512 = malloc(512 * 4);
    int reps = getenv("KREPLAY_REPS") ? atoi(getenv("KREPLAY_REPS")) : 1;
    uint64_t ck = 0;
    double t_router = 0, t_shewp = 0;
    printf("mode=dense reps=%d threads=%d evict=%d\n", reps, NTH, EVICT);
    hwp_sync();
    double wall0 = now_s();
    for (int rep = 0; rep < reps; rep++) {
        for (int i = 0; i < nops; i++) {
            if (!strcmp(ops[i].name, "head") && !have_head) continue;
            size_t rb = ggml_row_size(ops[i].ty, ops[i].ncols);
            if (rb * (size_t) ops[i].nrows != ops[i].bytes) {
                fprintf(stderr, "%s rowbytes mismatch %zu\n", ops[i].name, rb); return 1;
            }
            double s = now_s();
            for (int c = 0; c < ops[i].count; c++) {
                evict_l3();
                const float * x = ops[i].ncols == 4096
                    ? act4096 + (c % (AR / 2)) * 4096 : act + (c % AR) * 2048;
                run_quant(pool, x, xq, ops[i].ncols);
                /* shared-down input is 512-wide silu product, not x: emulate */
                if (!strcmp(ops[i].name, "sh_down")) {
                    struct ew_job ew = { g512, u512, z512, 512 };
                    /* reuse previous gate/up outputs is overkill; use act slice */
                    memcpy(g512, act + (c % AR) * 2048, 512 * 4);
                    memcpy(u512, act + ((c + 7) % AR) * 2048, 512 * 4);
                    pool_run(pool, swiglu_shard, &ew);
                    run_quant(pool, z512, xq, 512);
                }
                run_gemv(pool, ops[i].fn, W[i], xq, y, ops[i].ncols, ops[i].nrows, rb);
            }
            double dt = (now_s() - s) / reps;
            double gb = (double) ops[i].bytes * ops[i].count / 1e9;
            printf("  %-8s count=%2d rows=%6lld cols=%4lld ms=%.3f weight_GB=%.3f GB/s=%.2f\n",
                ops[i].name, ops[i].count, (long long) ops[i].nrows, (long long) ops[i].ncols,
                dt * 1000, gb, gb / dt);
            ck ^= fnv1a(y, ops[i].nrows > 8192 ? 8192 : ops[i].nrows);
        }
        /* router: 40 x F32 256x2048 */
        {
            double s = now_s();
            for (int c = 0; c < 40; c++) {
                evict_l3();
                const float * x = act + (c % AR) * 2048;
                struct gemv_job j;
                /* F32-expert rows: plain dot via ggml_vec_dot_f32, static shard */
                for (int t = 0; t < NTH; t++) { /* serial over shards; timed jointly */
                    int64_t r0 = (256 * t) / NTH, r1 = (256 * (t + 1)) / NTH;
                    for (int64_t r = r0; r < r1; r++)
                        ggml_vec_dot_f32(2048, &y[r], 0, (const float *) router + r * 2048, 0, x, 0, 1);
                }
                (void) j;
            }
            t_router += now_s() - s;
            ck ^= fnv1a(y, 256);
        }
    }
    double wall = now_s() - wall0;
    printf("  router   count=40 rows=   256 cols=2048 ms=%.3f (F32, serial-shard approx)\n",
        t_router / reps * 1000);
    printf("  shexp_ew (in sh_down) ms=%.3f\n", t_shewp);
    printf("checksum=%016llx wall_per_token_ms=%.3f\n", (unsigned long long) ck, wall / reps * 1000);
    pool_free(pool);
    return 0;
}

/* ================= MACH MODE: bounded-cache machinery ================= */
static int run_mach(const char * sidecar) {
    int fd = open(sidecar, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s\n", sidecar); return 1; }
    struct stat st; fstat(fd, &st);
    size_t SB = 876544;
    int nrec = st.st_size / SB;
    void * slot = malloc(SB);
    printf("mode=mach sidecar=%s recs=%d recbytes=%zu\n", sidecar, nrec, SB);
    /* warm pread */
    double s = now_s();
    for (int i = 0; i < nrec; i++) {
        size_t off = (size_t) i * SB, got = 0;
        while (got < SB) {
            ssize_t r = pread(fd, (char *) slot + got, SB - got, off + got);
            if (r <= 0) { fprintf(stderr, "pread\n"); return 1; }
            got += r;
        }
    }
    double warm = now_s() - s;
    printf("  pread_warm: %d x %zu B = %.1f MB in %.3f ms -> %.2f GB/s\n",
        nrec, SB, (double) nrec * SB / 1e6, warm * 1000, (double) nrec * SB / warm / 1e9);
    /* memcpy slot copy */
    void * dst = malloc(SB);
    s = now_s();
    for (int i = 0; i < nrec; i++) memcpy(dst, slot, SB);
    double mc = now_s() - s;
    printf("  memcpy   : %.3f ms -> %.2f GB/s\n", mc * 1000, (double) nrec * SB / mc / 1e9);
    /* cold pread with fadvise drop between reads */
    s = now_s();
    for (int i = 0; i < (nrec > 32 ? 32 : nrec); i++) {
        posix_fadvise(fd, (size_t) i * SB, SB, POSIX_FADV_DONTNEED);
        size_t off = (size_t) i * SB, got = 0;
        while (got < SB) {
            ssize_t r = pread(fd, (char *) slot + got, SB - got, off + got);
            if (r <= 0) { fprintf(stderr, "pread cold\n"); return 1; }
            got += r;
        }
    }
    double cold = now_s() - s;
    int nc = nrec > 32 ? 32 : nrec;
    printf("  pread_coldwise: %d recs %.3f ms -> %.2f GB/s (post-DONTNEED)\n",
        nc, cold * 1000, (double) nc * SB / cold / 1e9);
    /* sched_yield spin cost (phase6 readiness spin) */
    s = now_s();
    for (int i = 0; i < 100000; i++) sched_yield();
    printf("  sched_yield x100k: %.3f ms (%.1f ns/call)\n", (now_s() - s) * 1000, (now_s() - s) * 1e9 / 100000);
    close(fd);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: kreplay expert|dense|mach ...\n"); return 2; }
    if (getenv("KREPLAY_THREADS")) NTH = atoi(getenv("KREPLAY_THREADS"));
    if (getenv("KREPLAY_EVICT")) EVICT = atoi(getenv("KREPLAY_EVICT"));
    double t_init = now_s();
    /* required: FP16 table + CPU dispatch init (llama.cpp: ggml_backend_cpu_init) */
    ggml_cpu_init();
    /* required: IQ grid tables are runtime-initialized (llama.cpp does this at startup) */
    ggml_quantize_init(GGML_TYPE_IQ2_XXS);
    ggml_quantize_init(GGML_TYPE_IQ2_S);
    fprintf(stderr, "init_ms=%.1f\n", (now_s() - t_init) * 1000);
    evict_buf = malloc(evict_n * 4);
    for (size_t i = 0; i < evict_n; i++) evict_buf[i] = (float) i;
    if (!strcmp(argv[1], "expert") && argc == 5) return run_expert(argv[2], argv[3], argv[4]);
    if (!strcmp(argv[1], "dense") && argc == 4) return run_dense(argv[2], argv[3]);
    if (!strcmp(argv[1], "mach") && argc == 3) return run_mach(argv[2]);
    fprintf(stderr, "bad args\n");
    return 2;
}
