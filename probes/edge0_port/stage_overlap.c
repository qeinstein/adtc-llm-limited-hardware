// stage_overlap: real-trace staging replay + oracle-prefetch overlap bound.
//
// Fills two open gaps with REAL route traces (route_corpus.jsonl), REAL
// expert bytes, REAL ggml AVX2 kernels:
//  A. sync staging replay (noDN slots): management wall + hit-rate validation
//     of the Python sims in real code (prior replays used synthetic traces).
//  B. async oracle prefetch: producer fetches future tokens' miss-sets
//     (perfect corpus lookahead) while the consumer runs GEMV; measures the
//     DEPLOYABLE-UPPER-BOUND hidden fraction for Phase-3 prerouter overlap.
//
// Fetch modes: mem (memcpy from DRAM pool = pagecache-hit analog) and cold
// (pread from blob file with fadvise-DONTNEED eviction = cold-SSD analog).
// Layouts: sidecar (1 contiguous pread/miss) vs gguf (3 preads/miss).
//
// Memory-lean box design: bytes are fungible for timing, so (layer,expert)
// map onto a ~100MB pool of DISTINCT real expert bytes (checksums are
// sync-vs-async self-consistency). Slots capped at 512 (449MB).
//
// Usage: stage_overlap <sync|async1|async2|async4> <mem|cold|ssd>
//                        <sidecar|gguf> <nslots<=512> <ntokens<=256> [off]
// ssd = memcpy + calibrated 1GB/s delay per miss (transportable analog).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/stat.h>

#include "ggml.h"

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void quantize_row_q8_K(const float * x, void * y, int64_t k);

#define DIN 2048
#define DFF 512
#define TOPK 8
#define NLAYER 40
#define GATEB 270336
#define DOWNB 335872
#define SLOTB (GATEB + GATEB + DOWNB)
#define ROWB_GU 528
#define ROWB_DN 164
#define Q8B_2048 ((2048 / 256) * 292)
#define Q8B_512 ((512 / 256) * 292)
#define MAXTOK 256
#define MAXSLOTS 512
#define RAWDIR "/tmp/agent1_raw"
#define CORPUS "/home/fluxx/Workspace/adtc-llm-native-sparse/research/native_sparse_experiments/results/phase5e_route_corpus_v1/route_corpus.jsonl"
#define BLOB "/tmp/edge0_phase1/expert_blob.bin"

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
static uint64_t fnv1a(const void *p, size_t n, uint64_t h) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

// ---- trace ----
static int trace[MAXTOK][NLAYER][TOPK];
static int ntok = 0;
static void load_trace(int off, int n) {
    FILE *f = fopen(CORPUS, "r");
    if (!f) { fprintf(stderr, "no corpus\n"); exit(1); }
    char *line = NULL;
    size_t cap = 0;
    int idx = 0, got = 0;
    while (getline(&line, &cap, f) > 0 && got < n) {
        if (idx++ < off) continue;
        char *p = strstr(line, "\"layers\"");
        if (!p) { fprintf(stderr, "bad line\n"); exit(1); }
        int nums[NLAYER * TOPK], nn = 0;
        while (nn < NLAYER * TOPK) {
            while (*p && (*p < '0' || *p > '9') && *p != '-') p++;
            if (!*p) break;
            nums[nn++] = (int)strtol(p, &p, 10);
        }
        if (nn != NLAYER * TOPK) { fprintf(stderr, "short line %d\n", nn); exit(1); }
        for (int l = 0; l < NLAYER; l++)
            for (int k = 0; k < TOPK; k++)
                trace[got][l][k] = nums[l * TOPK + k];
        got++;
    }
    free(line);
    fclose(f);
    ntok = got;
    printf("trace: %d tokens from offset %d\n", ntok, off);
}

// ---- distinct real-expert pool (timing-fungible byte source) ----
static uint8_t **g_pool = NULL;
static int g_pooln = 0;
static void load_pool(void) {
    // gather present (layer,expert) with all 3 files, prefer full layers
    int layers[3] = {20, 39, 0};
    int cap = 110;
    g_pool = malloc(cap * sizeof(uint8_t *));
    char path[256];
    for (int li = 0; li < 3 && g_pooln < cap; li++) {
        for (int e = 0; e < 256 && g_pooln < cap; e++) {
            uint8_t *b = malloc(SLOTB);
            int ok = 1;
            for (int pr = 0; pr < 3; pr++) {
                const char *pn = pr == 0 ? "gate" : pr == 1 ? "up" : "down";
                const char *ex = pr == 2 ? "iq2s" : "iq2xxs";
                size_t want = pr == 2 ? DOWNB : GATEB;
                uint8_t *dst = b + (pr == 0 ? 0 : pr == 1 ? GATEB : 2 * GATEB);
                snprintf(path, sizeof(path), "%s/L%02d_E%03d_%s.%s",
                         RAWDIR, layers[li], e, pn, ex);
                int fd = open(path, O_RDONLY);
                if (fd < 0) { ok = 0; break; }
                size_t off = 0;
                while (off < want) {
                    ssize_t r = read(fd, dst + off, want - off);
                    if (r <= 0) { ok = 0; break; }
                    off += r;
                }
                close(fd);
                if (!ok) break;
            }
            if (!ok) { free(b); continue; }
            g_pool[g_pooln++] = b;
        }
    }
    printf("pool: %d distinct real bundles (%.0f MB)\n", g_pooln,
           (double)g_pooln * SLOTB / 1e6);
    if (g_pooln < 64) { fprintf(stderr, "pool too small\n"); exit(1); }
}
static inline uint8_t *pool_for(int l, int e) {
    return g_pool[(unsigned)(l * 131 + e * 17) % (unsigned)g_pooln];
}

// ---- blob file for cold reads ----
static int blob_fd = -1;
static size_t blob_n = 0;
static void build_blob(void) {
    struct stat st;
    if (stat(BLOB, &st) == 0 && st.st_size > 0) {
        blob_fd = open(BLOB, O_RDONLY);
        blob_n = st.st_size / SLOTB;
        printf("blob: reuse %s (%zu bundles)\n", BLOB, blob_n);
        return;
    }
    blob_fd = open(BLOB, O_RDWR | O_CREAT | O_TRUNC, 0644);
    size_t target = (128u << 20) / SLOTB;
    for (size_t n = 0; n < target; n++) {
        uint8_t *b = g_pool[n % g_pooln];
        if (write(blob_fd, b, SLOTB) != SLOTB) {
            fprintf(stderr, "blob write fail (disk?)\n"); exit(1);
        }
    }
    blob_n = target;
    printf("blob: built %s (%zu bundles)\n", BLOB, blob_n);
}
static size_t blob_off(int l, int e) {
    return ((size_t)((unsigned)(l * 131u + e * 17u) % blob_n)) * SLOTB;
}

// ---- slots (noDN: overwrite in place; clock eviction) ----
// state: 0 EMPTY, 1 READY, 2 FILLING. Async path guards metadata with mu;
// FILLING slots are never victimized (in-flight pin by construction).
struct slotset {
    uint8_t *mem;
    int *key;
    int *lru;
    int *state;
    int *pin;   // >0 while consumer runs GEMV on the slot (async only)
    int nslots, hand;
    pthread_mutex_t mu;
    pthread_cond_t cv;
};
static void slots_init(struct slotset *s, int n) {
    s->nslots = n;
    s->mem = aligned_alloc(4096, (size_t)n * SLOTB);
    memset(s->mem, 0, (size_t)n * SLOTB);
    s->key = malloc(n * sizeof(int));
    s->lru = malloc(n * sizeof(int));
    s->state = malloc(n * sizeof(int));
    s->pin = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) { s->key[i] = -1; s->lru[i] = 0; s->state[i] = 0; s->pin[i] = 0; }
    s->hand = 0;
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv, NULL);
}
static int slots_find(struct slotset *s, int key) {
    for (int i = 0; i < s->nslots; i++)
        if (s->key[i] == key) { s->lru[i] = 1; return i; }
    return -1;
}
static int slots_evict(struct slotset *s) {
    for (;;) {
        int i = s->hand;
        s->hand = (s->hand + 1) % s->nslots;
        if (s->key[i] < 0) return i;
        if (!s->lru[i]) return i;
        s->lru[i] = 0;
    }
}
// locked find: returns READY index, or -2 if FILLING, else -1 (caller holds mu)
static int slots_find_locked(struct slotset *s, int key) {
    int filling = -1;
    for (int i = 0; i < s->nslots; i++)
        if (s->key[i] == key) {
            if (s->state[i] == 1) { s->lru[i] = 1; return i; }
            filling = -2;
        }
    return filling;
}
// locked reserve: victim EMPTY else clock-over-READY; marks FILLING+key.
// Returns -1 if all FILLING (caller waits on cv). Caller holds mu.
static int slots_reserve_locked(struct slotset *s, int key) {
    for (int i = 0; i < s->nslots; i++)
        if (s->state[i] == 0) {
            s->state[i] = 2; s->key[i] = key; s->lru[i] = 1; return i;
        }
    for (int n = 0; n < 2 * s->nslots; n++) {
        int i = s->hand;
        s->hand = (s->hand + 1) % s->nslots;
        if (s->state[i] == 2 || s->pin[i]) continue;
        if (!s->lru[i]) {
            s->state[i] = 2; s->key[i] = key; s->lru[i] = 1; return i;
        }
        s->lru[i] = 0;
    }
    return -1;
}

// fetch one bundle into dst. Returns ns spent.
static int g_mode_cold = 0, g_mode_ssd = 0, g_layout_gguf = 0;
static uint64_t fetch_bundle(uint8_t *dst, int l, int e) {
    uint64_t t0 = now_ns();
    if (g_mode_ssd) {
        // transportable cold-SSD analog: real bytes + calibrated 1GB/s
        // delay (local pread is page-cache-resident; this box's SSD tells
        // nothing about the target's). sidecar 0.86ms, gguf 3x0.30ms.
        memcpy(dst, pool_for(l, e), SLOTB);
        struct timespec ts = {0, g_layout_gguf ? 900000 : 860000};
        nanosleep(&ts, NULL);
    } else if (!g_mode_cold) {
        memcpy(dst, pool_for(l, e), SLOTB);
    } else if (!g_layout_gguf) {
        size_t off = blob_off(l, e);
        posix_fadvise(blob_fd, off, SLOTB, POSIX_FADV_DONTNEED);
        size_t got = 0;
        while (got < SLOTB) {
            ssize_t r = pread(blob_fd, dst + got, SLOTB - got, off + got);
            if (r <= 0) { fprintf(stderr, "pread fail\n"); exit(1); }
            got += r;
        }
    } else {
        size_t span = blob_n * SLOTB;
        size_t base = blob_off(l, e);
        size_t o[3] = {base % (span - GATEB),
                       (base + span / 3) % (span - GATEB),
                       (base + 2 * span / 3) % (span - DOWNB)};
        size_t nb[3] = {GATEB, GATEB, DOWNB};
        size_t doff[3] = {0, GATEB, 2 * GATEB};
        for (int i = 0; i < 3; i++) {
            posix_fadvise(blob_fd, o[i], nb[i], POSIX_FADV_DONTNEED);
            size_t g = 0;
            while (g < nb[i]) {
                ssize_t r = pread(blob_fd, dst + doff[i] + g, nb[i] - g, o[i] + g);
                if (r <= 0) exit(1);
                g += r;
            }
        }
    }
    return now_ns() - t0;
}

// ---- GEMV of one resident bundle ----
static uint8_t *g_act2048, *g_act512;
static uint64_t gemv_bundle(uint8_t *slot, uint64_t *h) {
    float out[2048];
    uint64_t t0 = now_ns();
    for (int pr = 0; pr < 3; pr++) {
        uint8_t *w = slot + (pr == 0 ? 0 : pr == 1 ? GATEB : 2 * GATEB);
        int rows = pr == 2 ? 2048 : 512, K = pr == 2 ? 512 : 2048;
        size_t wr = pr == 2 ? ROWB_DN : ROWB_GU;
        uint8_t *act = pr == 2 ? g_act512 : g_act2048;
        for (int r = 0; r < rows; r++) {
            if (pr == 2)
                ggml_vec_dot_iq2_s_q8_K(K, &out[r], 0, w + r * wr, 0, act, 0, 1);
            else
                ggml_vec_dot_iq2_xxs_q8_K(K, &out[r], 0, w + r * wr, 0, act, 0, 1);
        }
        *h = fnv1a(out, rows * sizeof(float), *h);
    }
    return now_ns() - t0;
}

struct stats {
    uint64_t fetch_ns, gemv_ns, stall_ns, copy_ns;
    long hits, miss, promo, wasted, sync_fetch;
    uint64_t t_stall[MAXTOK];
    long t_miss[MAXTOK], t_promo[MAXTOK];
};

// SYNC replay
static void run_sync(struct slotset *s, struct stats *st) {
    uint64_t h = 1469598103934665603ULL;
    for (int t = 0; t < ntok; t++) {
        for (int l = 0; l < NLAYER; l++) {
            for (int k = 0; k < TOPK; k++) {
                int e = trace[t][l][k], key = (l << 8) | e;
                int si = slots_find(s, key);
                if (si >= 0) { st->hits++; }
                else {
                    st->miss++;
                    si = slots_evict(s);
                    st->fetch_ns += fetch_bundle(s->mem + (size_t)si * SLOTB, l, e);
                    s->key[si] = key;
                    s->lru[si] = 1;
                }
                st->gemv_ns += gemv_bundle(s->mem + (size_t)si * SLOTB, &h);
            }
        }
    }
    printf("sync checksum=%016llx\n", (unsigned long long)h);
}

// ASYNC oracle replay, direct-to-slot prefetch (production design):
// producer reserves victim slots and workers fetch straight into them;
// no staging buffer, no promote-copy, no two-tier eviction pathology.
static struct slotset *g_s;
static int g_look = 1;
static atomic_int g_stop = 0;
static struct stats *g_st;
static int g_cons_tok = -1;
static pthread_mutex_t g_prog_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_prog_cv = PTHREAD_COND_INITIALIZER;

// fetch work queue: producer-main enqueues, NFETCH workers execute.
// Parallel fetch is REQUIRED, not optional: a single serial prefetcher
// (0.86ms/miss) is producer-bound below consumer pace and hides nothing.
#define NFETCH 4
#define QMAX 256
struct fitem { int key, l, e, slot; };
static int g_qmax = QMAX;
static struct {
    struct fitem items[QMAX];
    int head, tail, count;
    pthread_mutex_t mu;
    pthread_cond_t ne, nf;
} g_fq;

static void *producer(void *arg) {
    (void)arg;
    for (int t = 0; t < ntok && !atomic_load(&g_stop); t++) {
        pthread_mutex_lock(&g_prog_mu);
        while (!atomic_load(&g_stop) && t > g_cons_tok + g_look)
            pthread_cond_wait(&g_prog_cv, &g_prog_mu);
        pthread_mutex_unlock(&g_prog_mu);
        if (atomic_load(&g_stop)) break;
        for (int l = 0; l < NLAYER; l++)
            for (int k = 0; k < TOPK; k++) {
                int e = trace[t][l][k], key = (l << 8) | e;
                pthread_mutex_lock(&g_s->mu);
                int f = slots_find_locked(g_s, key);
                int slot = -1;
                if (f < 0 && f != -2) {
                    // absent: reserve a victim (wait if all FILLING)
                    while (!atomic_load(&g_stop)) {
                        slot = slots_reserve_locked(g_s, key);
                        if (slot >= 0) break;
                        pthread_cond_wait(&g_s->cv, &g_s->mu);
                    }
                }
                pthread_mutex_unlock(&g_s->mu);
                if (slot < 0) continue; // present, or stop
                pthread_mutex_lock(&g_fq.mu);
                while (g_fq.count >= g_qmax && !atomic_load(&g_stop))
                    pthread_cond_wait(&g_fq.nf, &g_fq.mu);
                if (!atomic_load(&g_stop)) {
                    g_fq.items[g_fq.tail] = (struct fitem){key, l, e, slot};
                    g_fq.tail = (g_fq.tail + 1) % QMAX;
                    g_fq.count++;
                    pthread_cond_signal(&g_fq.ne);
                } else {
                    // stop with reservation held: release it
                    pthread_mutex_lock(&g_s->mu);
                    if (g_s->key[slot] == key && g_s->state[slot] == 2) {
                        g_s->state[slot] = 1; // no bytes, but run is over
                    }
                    pthread_mutex_unlock(&g_s->mu);
                }
                pthread_mutex_unlock(&g_fq.mu);
            }
    }
    return NULL;
}

static void *fworker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_fq.mu);
        while (g_fq.count == 0 && !atomic_load(&g_stop))
            pthread_cond_wait(&g_fq.ne, &g_fq.mu);
        if (g_fq.count == 0 && atomic_load(&g_stop)) {
            pthread_mutex_unlock(&g_fq.mu);
            break;
        }
        struct fitem it = g_fq.items[g_fq.head];
        g_fq.head = (g_fq.head + 1) % QMAX;
        g_fq.count--;
        pthread_cond_signal(&g_fq.nf);
        pthread_mutex_unlock(&g_fq.mu);
        uint64_t dt = fetch_bundle(g_s->mem + (size_t)it.slot * SLOTB, it.l, it.e);
        __atomic_fetch_add(&g_st->fetch_ns, dt, __ATOMIC_RELAXED);
        pthread_mutex_lock(&g_s->mu);
        if (g_s->key[it.slot] == it.key && g_s->state[it.slot] == 2)
            g_s->state[it.slot] = 1;
        pthread_cond_broadcast(&g_s->cv);
        pthread_mutex_unlock(&g_s->mu);
    }
    return NULL;
}

static void run_async(struct slotset *s, struct stats *st, int look) {
    g_s = s; g_st = st; g_look = look;
    atomic_store(&g_stop, 0);
    g_cons_tok = -1;
    g_qmax = QMAX;
    if (g_qmax > s->nslots / 2) g_qmax = s->nslots / 2;
    if (g_qmax < 8) g_qmax = 8;
    pthread_mutex_init(&g_fq.mu, NULL);
    pthread_cond_init(&g_fq.ne, NULL);
    pthread_cond_init(&g_fq.nf, NULL);
    g_fq.head = g_fq.tail = g_fq.count = 0;
    pthread_t th, fw[NFETCH];
    pthread_create(&th, NULL, producer, NULL);
    for (int i = 0; i < NFETCH; i++)
        pthread_create(&fw[i], NULL, fworker, NULL);
    uint64_t h = 1469598103934665603ULL;
    for (int t = 0; t < ntok; t++) {
        uint64_t st0 = st->stall_ns;
        long m0 = st->miss;
        for (int l = 0; l < NLAYER; l++)
            for (int k = 0; k < TOPK; k++) {
                int e = trace[t][l][k], key = (l << 8) | e;
                int si = -1;
                pthread_mutex_lock(&s->mu);
                int f = slots_find_locked(s, key);
                if (f >= 0) { si = f; st->hits++; s->pin[si]++; }
                pthread_mutex_unlock(&s->mu);
                if (si < 0) {
                    if (f == -2) {
                        // FILLING: wait out a fetch, then take over if raw
                        // (5ms covers ssd p99; longer spin only pays when
                        // the producer paces ahead, i.e. with headroom)
                        for (int spin = 0; spin < 250; spin++) {
                            struct timespec ts = {0, 20000};
                            nanosleep(&ts, NULL);
                            pthread_mutex_lock(&s->mu);
                            f = slots_find_locked(s, key);
                            if (f >= 0) { si = f; st->hits++; s->pin[si]++; }
                            pthread_mutex_unlock(&s->mu);
                            if (si >= 0) break;
                        }
                    }
                    if (si < 0) {
                        // sync install (stall path)
                        uint64_t f0 = now_ns();
                        pthread_mutex_lock(&s->mu);
                        while (!atomic_load(&g_stop)) {
                            si = slots_reserve_locked(s, key);
                            if (si >= 0) break;
                            pthread_cond_wait(&s->cv, &s->mu);
                        }
                        pthread_mutex_unlock(&s->mu);
                        uint64_t dt = fetch_bundle(s->mem + (size_t)si * SLOTB, l, e);
                        st->fetch_ns += dt;
                        pthread_mutex_lock(&s->mu);
                        if (s->key[si] == key) { s->state[si] = 1; s->pin[si]++; }
                        pthread_cond_broadcast(&s->cv);
                        pthread_mutex_unlock(&s->mu);
                        st->stall_ns += now_ns() - f0;
                        st->miss++;
                        st->sync_fetch++;
                    }
                }
                st->gemv_ns += gemv_bundle(s->mem + (size_t)si * SLOTB, &h);
                pthread_mutex_lock(&s->mu);
                s->pin[si]--;
                pthread_cond_broadcast(&s->cv);
                pthread_mutex_unlock(&s->mu);
            }
        st->t_stall[t] = st->stall_ns - st0;
        st->t_miss[t] = st->miss - m0;
        st->t_promo[t] = 0;
        pthread_mutex_lock(&g_prog_mu);
        g_cons_tok = t;
        pthread_cond_broadcast(&g_prog_cv);
        pthread_mutex_unlock(&g_prog_mu);
    }
    atomic_store(&g_stop, 1);
    pthread_mutex_lock(&g_prog_mu);
    pthread_cond_broadcast(&g_prog_cv);
    pthread_mutex_unlock(&g_prog_mu);
    pthread_mutex_lock(&g_fq.mu);
    pthread_cond_broadcast(&g_fq.ne);
    pthread_cond_broadcast(&g_fq.nf);
    pthread_mutex_unlock(&g_fq.mu);
    pthread_mutex_lock(&s->mu);
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mu);
    pthread_join(th, NULL);
    for (int i = 0; i < NFETCH; i++) pthread_join(fw[i], NULL);
    printf("async checksum=%016llx\n", (unsigned long long)h);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <sync|async1|async2|async4> <mem|cold|ssd> "
                "<sidecar|gguf> <nslots<=512> <ntokens<=256> [off]\n", argv[0]);
        return 1;
    }
    int is_async = strncmp(argv[1], "async", 5) == 0;
    int look = is_async ? atoi(argv[1] + 5) : 0;
    if (look < 1) look = 1;
    g_mode_cold = !strcmp(argv[2], "cold");
    g_mode_ssd = !strcmp(argv[2], "ssd");
    g_layout_gguf = !strcmp(argv[3], "gguf");
    int nslots = atoi(argv[4]), n = atoi(argv[5]), off = argc > 6 ? atoi(argv[6]) : 0;
    if (n > MAXTOK) n = MAXTOK;
    if (nslots > MAXSLOTS) { fprintf(stderr, "nslots cap %d (box RAM)\n", MAXSLOTS); return 1; }
    ggml_quantize_init(GGML_TYPE_IQ2_XXS);
    ggml_quantize_init(GGML_TYPE_IQ2_S);
    load_trace(off, n);
    load_pool();
    if (g_mode_cold) build_blob();
    float *f2048 = aligned_alloc(64, 2048 * 4), *f512 = aligned_alloc(64, 512 * 4);
    uint64_t seed = 0x12345;
    for (int i = 0; i < 2048; i++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f2048[i] = ((seed >> 33) / 2147483648.0f - 0.5f);
    }
    for (int i = 0; i < 512; i++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        f512[i] = ((seed >> 33) / 2147483648.0f - 0.5f);
    }
    g_act2048 = aligned_alloc(64, Q8B_2048);
    g_act512 = aligned_alloc(64, Q8B_512);
    quantize_row_q8_K(f2048, g_act2048, 2048);
    quantize_row_q8_K(f512, g_act512, 512);
    struct slotset s;
    slots_init(&s, nslots);
    struct stats st;
    memset(&st, 0, sizeof(st));
    struct rusage ru0, ru1;
    getrusage(RUSAGE_SELF, &ru0);
    uint64_t t0 = now_ns();
    if (is_async) run_async(&s, &st, look);
    else run_sync(&s, &st);
    uint64_t wall = now_ns() - t0;
    getrusage(RUSAGE_SELF, &ru1);
    long tot = st.hits + st.miss + st.promo;
    printf("mode=%s fetch=%s layout=%s slots=%d ntok=%d\n", argv[1], argv[2],
           argv[3], nslots, ntok);
    printf("hits=%ld miss=%ld promo=%ld wasted=%ld sync_fetch=%ld\n",
           st.hits, st.miss, st.promo, st.wasted, st.sync_fetch);
    printf("hit_rate=%.4f miss_per_tok=%.1f\n",
           (double)(st.hits + st.promo) / tot, (double)(st.miss) / ntok);
    printf("wall_ms=%.1f fetch_ms=%.1f gemv_ms=%.1f stall_ms=%.1f\n",
           wall / 1e6, st.fetch_ns / 1e6, st.gemv_ns / 1e6, st.stall_ns / 1e6);
    printf("per_tok_ms: wall=%.2f fetch=%.2f gemv=%.2f stall=%.2f copy=%.2f\n",
           wall / 1e6 / ntok, st.fetch_ns / 1e6 / ntok, st.gemv_ns / 1e6 / ntok,
           st.stall_ns / 1e6 / ntok, st.copy_ns / 1e6 / ntok);
    printf("minflt=%ld majflt=%ld maxrss_kb=%ld\n",
           ru1.ru_minflt - ru0.ru_minflt, ru1.ru_majflt - ru0.ru_majflt,
           ru1.ru_maxrss);
    if (is_async && ntok > 4) {
        uint64_t s0 = 0, s1 = 0;
        long m0 = 0, m1 = 0;
        for (int t = 0; t < ntok; t++) {
            if (t < 2) { s0 += st.t_stall[t]; m0 += st.t_miss[t]; }
            else { s1 += st.t_stall[t]; m1 += st.t_miss[t]; }
        }
        printf("startup(t0-1): stall=%.1fms miss=%.0f/tok | "
               "steady(t2+): stall=%.2fms miss=%.1f/tok\n",
               s0 / 1e6 / 2, (double)m0 / 2, s1 / 1e6 / (ntok - 2),
               (double)m1 / (ntok - 2));
    }
    return 0;
}
