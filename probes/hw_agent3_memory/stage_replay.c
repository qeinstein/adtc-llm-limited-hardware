// stage_replay: faithful single-thread replay of the v10 STAGED expert path
// (BOUNDED_C in kaggle/native-sparse-staged-q2k-v1/staged_q2k_v1.py, sync arm:
// phase6_load/phase6_reserve_slot/phase6_tensor_ptr), driven over REAL L00
// IQ2 expert bytes with the REAL ggml AVX2 kernels, under different slot
// policies. Measures wall (manage vs GEMV split), min/maj faults, RSS,
// hits/misses/evictions, and output checksums for parity.
//
// Policies:
//   base   v10-verbatim: packed stride (876544 B = 214 x 4K), MADV_DONTNEED
//          on every eviction, memcpy refill (pread-from-pagecache analog).
//   noDN   same layout, NO madvise on evict (overwrite pages in place).
//   huge   2 MiB stride, 2 MiB-aligned base, MADV_HUGEPAGE, DONTNEED on evict.
//   hugeND 2 MiB stride/align, NO madvise on evict.
//   noslot infinite slots (no eviction): parity ref + zero-churn anchor.
//
// Usage: stage_replay <policy> <nslots> <ntokens> <zipf_s|uniform> [seed]
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void quantize_row_q8_K(const float * x, void * y, int64_t k);

#define DIN 2048
#define DFF 512
#define TOPK 8
#define GATEB 270336
#define DOWNB 335872
#define SLOTB (GATEB + GATEB + DOWNB)   /* 876544 = 214 * 4096 */
#define ROWB_GU 528
#define ROWB_DN 164
#define MAXEXP 256

static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static int cmp_dbl(const void * a, const void * b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static int cmp_long(const void * a, const void * b) {
    long x = *(const long *)a, y = *(const long *)b;
    return (x > y) - (x < y);
}
static uint64_t fnv1a(const void * p, size_t n, uint64_t h) {
    const uint8_t * b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
static long get_flt(long * minflt, long * majflt) {
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    *minflt = ru.ru_minflt; *majflt = ru.ru_majflt;
    return ru.ru_maxrss; /* KB */
}

/* ---- disk pool: real expert bytes, malloc'd (pagecache-resident GGUF analog) ---- */
static uint8_t * disk[MAXEXP][3];
static int have_exp[MAXEXP], nexp = 0;
static int exp_ids[MAXEXP];

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

static int load_l00(const char * dir) {
    char path[512];
    for (int e = 0; e < MAXEXP; e++) {
        snprintf(path, sizeof path, "%s/L00_E%03d_gate.iq2xxs", dir, e);
        void * g = load_file(path, GATEB);
        if (!g) continue;
        snprintf(path, sizeof path, "%s/L00_E%03d_up.iq2xxs", dir, e);
        void * u = load_file(path, GATEB);
        snprintf(path, sizeof path, "%s/L00_E%03d_down.iq2s", dir, e);
        void * d = load_file(path, DOWNB);
        if (!u || !d) { free(g); free(u); free(d); continue; }
        disk[e][0] = g; disk[e][1] = u; disk[e][2] = d;
        have_exp[e] = 1; exp_ids[nexp++] = e;
    }
    return nexp;
}

/* ---- staged slots (mirror phase6_slot / phase6_storage) ---- */
struct slot { int expert; uint64_t age; int valid; };
static struct slot * slots;
static int * bundle_slot; /* [MAXEXP] expert -> slot or -1 */
static unsigned char * storage;
static size_t stride, nslots;
static uint64_t age = 0;
static uint64_t n_req, n_hit, n_miss, n_evict;
static int use_dontneed;

static void store_init(size_t ns, size_t st, int align2m, int dontneed, int hugepage) {
    nslots = ns; stride = st; use_dontneed = dontneed;
    slots = calloc(nslots, sizeof *slots);
    bundle_slot = malloc(MAXEXP * sizeof *bundle_slot);
    for (int i = 0; i < MAXEXP; i++) bundle_slot[i] = -1;
    size_t total = stride * nslots;
    if (align2m) {
        /* over-allocate and align base up to 2 MiB */
        unsigned char * raw = mmap(NULL, total + (2 << 20), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (raw == MAP_FAILED) { perror("mmap"); exit(1); }
        uintptr_t a = ((uintptr_t)raw + (2 << 20) - 1) & ~((uintptr_t)(2 << 20) - 1);
        storage = (unsigned char *)a;
    } else {
        storage = mmap(NULL, total, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (storage == MAP_FAILED) { perror("mmap"); exit(1); }
    }
    if (hugepage) madvise(storage, total, MADV_HUGEPAGE);
    n_req = n_hit = n_miss = n_evict = 0; age = 0;
}

static int reserve_slot(int expert) {
    int slot = -1;
    for (size_t i = 0; i < nslots; i++) if (!slots[i].valid) { slot = (int)i; break; }
    if (slot < 0) {
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < nslots; i++)
            if (slots[i].age < oldest) { oldest = slots[i].age; slot = (int)i; }
        int old = slots[slot].expert;
        bundle_slot[old] = -1;
        n_evict++;
        if (use_dontneed)
            madvise(storage + (size_t)slot * stride, stride, MADV_DONTNEED);
    }
    slots[slot].expert = expert; slots[slot].age = ++age; slots[slot].valid = 1;
    bundle_slot[expert] = slot;
    return slot;
}

/* phase6_load analog; refill = memcpy (pread from hot pagecache equivalent).
 * Returns slot index; sets *was_miss. */
static int stage_load(int expert, int * was_miss) {
    n_req++;
    int slot = bundle_slot[expert];
    if (slot >= 0) { n_hit++; slots[slot].age = ++age; *was_miss = 0; return slot; }
    n_miss++; *was_miss = 1;
    slot = reserve_slot(expert);
    unsigned char * dst = storage + (size_t)slot * stride;
    /* NOTE: v10 packs planes back-to-back at in_slot offsets 0, GATEB, 2*GATEB.
     * huge-stride keeps the same intra-slot packing; only the stride changes. */
    memcpy(dst,           disk[expert][0], GATEB);
    memcpy(dst + GATEB,   disk[expert][1], GATEB);
    memcpy(dst + 2*GATEB, disk[expert][2], DOWNB);
    return slot;
}

/* ---- compute: one expert's gate/up/silu/down from SLOT bytes ---- */
static float x_[DIN], gout[DFF], uout[DFF], h[DFF], dout[DIN], y[DIN];
static uint8_t xq[DIN * 2], hq[DFF * 2];

static void expert_from_slot(int slot) {
    const unsigned char * base = storage + (size_t)slot * stride;
    const char * g = (const char *)base, * u = (const char *)base + GATEB;
    const char * d = (const char *)base + 2 * GATEB;
    for (int r = 0; r < DFF; r++)
        ggml_vec_dot_iq2_xxs_q8_K(DIN, &gout[r], 0, g + r * ROWB_GU, 0, xq, 0, 1);
    for (int r = 0; r < DFF; r++)
        ggml_vec_dot_iq2_xxs_q8_K(DIN, &uout[r], 0, u + r * ROWB_GU, 0, xq, 0, 1);
    for (int r = 0; r < DFF; r++) {
        float gg = gout[r];
        h[r] = (gg / (1.0f + expf(-gg))) * uout[r];
    }
    quantize_row_q8_K(h, hq, DFF);
    for (int r = 0; r < DIN; r++)
        ggml_vec_dot_iq2_s_q8_K(DFF, &dout[r], 0, d + r * ROWB_DN, 0, hq, 0, 1);
    const float w = 1.0f / TOPK;
    for (int r = 0; r < DIN; r++) y[r] += w * dout[r];
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <base|noDN|huge|hugeND|noslot> <nslots> <ntokens> <zipf_s|uniform> [seed]\n", argv[0]);
        return 1;
    }
    const char * policy = argv[1];
    size_t ns = (size_t)atol(argv[2]);
    int ntok = atoi(argv[3]);
    double zipf_s = -1;
    int uniform = !strcmp(argv[4], "uniform");
    if (!uniform) zipf_s = atof(argv[4]);
    unsigned seed = argc > 5 ? (unsigned)atoi(argv[5]) : 12345;

    int n = load_l00("/tmp/agent1_raw");
    if (n == 0) { fprintf(stderr, "no L00 experts found\n"); return 1; }
    fprintf(stderr, "loaded %d L00 experts\n", n);

    int is_noslot = !strcmp(policy, "noslot");
    int is_huge = !strncmp(policy, "huge", 4);
    int dontneed = !strcmp(policy, "base") || !strcmp(policy, "huge");
    if (is_noslot) { ns = MAXEXP; dontneed = 0; }
    store_init(ns, is_huge ? (2u << 20) : SLOTB, is_huge, dontneed, is_huge);
    fprintf(stderr, "policy=%s nslots=%zu stride=%zu store_bytes=%zu dontneed=%d\n",
        policy, nslots, stride, stride * nslots, dontneed);

    for (int i = 0; i < DIN; i++) x_[i] = sinf(i * 0.37f) * 0.6f + cosf(i * 0.11f) * 0.4f;
    quantize_row_q8_K(x_, xq, DIN);

    /* zipf sampler over resident experts */
    double * cdf = malloc(n * sizeof *cdf);
    double acc = 0;
    for (int i = 0; i < n; i++) { acc += uniform ? 1.0 : 1.0 / pow(i + 1, zipf_s); cdf[i] = acc; }
    srand(seed);
    int * trace = malloc((size_t)ntok * TOPK * sizeof *trace);
    for (int t = 0; t < ntok; t++) {
        int picked = 0, guard = 0;
        while (picked < TOPK && guard++ < 10000) {
            double r = (double)rand() / RAND_MAX * acc;
            int lo = 0, hi = n - 1;
            while (lo < hi) { int mid = (lo + hi) / 2; if (cdf[mid] < r) lo = mid + 1; else hi = mid; }
            int e = exp_ids[lo], dup = 0;
            for (int k = 0; k < picked; k++) if (trace[(size_t)t * TOPK + k] == e) { dup = 1; break; }
            if (!dup) trace[(size_t)t * TOPK + picked++] = e;
        }
    }

    double * t_tok = malloc(ntok * sizeof *t_tok);
    double * t_man = malloc(ntok * sizeof *t_man);
    double * t_gem = malloc(ntok * sizeof *t_gem);
    long * f_tok = malloc(ntok * sizeof *f_tok);
    uint64_t sum = 1469598103934665603ULL;
    long mf0, Mf0, mf1, Mf1;
    get_flt(&mf0, &Mf0);
    double man_tot = 0, gem_tot = 0;
    for (int t = 0; t < ntok; t++) {
        memset(y, 0, sizeof y);
        long a0, A0, a1, A1;
        get_flt(&a0, &A0);
        double t0 = now_ns(), man = 0;
        int slots8[TOPK];
        for (int k = 0; k < TOPK; k++) {
            double m0 = now_ns();
            int wm = 0;
            slots8[k] = stage_load(trace[(size_t)t * TOPK + k], &wm);
            man += now_ns() - m0;
        }
        double g0 = now_ns();
        for (int k = 0; k < TOPK; k++) expert_from_slot(slots8[k]);
        double t1 = now_ns();
        get_flt(&a1, &A1);
        t_tok[t] = t1 - t0; t_man[t] = man; t_gem[t] = t1 - g0;
        f_tok[t] = a1 - a0;
        man_tot += man; gem_tot += t1 - g0;
        sum = fnv1a(y, sizeof y, sum);
    }
    long maxrss = get_flt(&mf1, &Mf1);
    qsort(t_tok, ntok, sizeof(double), cmp_dbl);
    qsort(t_man, ntok, sizeof(double), cmp_dbl);
    qsort(t_gem, ntok, sizeof(double), cmp_dbl);
    qsort(f_tok, ntok, sizeof(long), cmp_long);
    double med = t_tok[ntok/2], p90 = t_tok[(ntok*9)/10];
    printf("policy=%s nslots=%zu ntok=%d trace=%s\n", policy, nslots, ntok, argv[4]);
    printf("req=%llu hit=%llu miss=%llu evict=%llu hitrate=%.4f\n",
        (unsigned long long)n_req, (unsigned long long)n_hit,
        (unsigned long long)n_miss, (unsigned long long)n_evict,
        (double)n_hit / (n_req ? n_req : 1));
    printf("tok_ms_med=%.3f tok_ms_p90=%.3f manage_ms_med=%.4f gemv_ms_med=%.3f\n",
        med / 1e6, p90 / 1e6, t_man[ntok/2] / 1e6, t_gem[ntok/2] / 1e6);
    printf("manage_ms_tot=%.2f gemv_ms_tot=%.2f manage_frac=%.4f\n",
        man_tot / 1e6, gem_tot / 1e6, man_tot / (man_tot + gem_tot));
    printf("minflt_tot=%ld majflt_tot=%ld minflt_per_tok=%.1f minflt_med=%ld maxrss_kb=%ld\n",
        mf1 - mf0, Mf1 - Mf0, (double)(mf1 - mf0) / ntok, f_tok[ntok/2], maxrss);
    printf("minflt_per_miss=%.1f faults_avoidable_note=see_report\n",
        (double)(mf1 - mf0) / (n_miss ? n_miss : 1));
    printf("checksum=%016llx\n", (unsigned long long)sum);
    printf("store_bytes=%zu bytes_per_expert=%d\n", stride * nslots, SLOTB);
    return 0;
}
