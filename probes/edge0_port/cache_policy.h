// edge0 cache policy: static-hot pins (bitset) + LRU (hash + linked list).
// Header-only; integrator wires edge0_cache_request() into the fetch path:
//   hit (1)  -> bundle resident, no fetch
//   miss (0) -> fetch bundle, then it is resident (admitted on fill)
// Pins loaded from cache_config.json pins_global for the chosen RSS target.
// K4-independent: key space (layer<<8)|expert covers any top-k; JOIN step
// re-derives pin SETS on K4 traces, this code is unchanged.
#ifndef EDGE0_CACHE_POLICY_H
#define EDGE0_CACHE_POLICY_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EDGE0_NBUNDLES 10240
#define EDGE0_PIN_WORDS (EDGE0_NBUNDLES / 64)

typedef struct {
    uint64_t pin_bits[EDGE0_PIN_WORDS];
    int32_t * ht_key;   // open addressing, -1 = empty/tombstone(-2)
    int32_t * ht_val;   // node index
    int32_t * prevv;
    int32_t * nextv;
    int32_t * node_key;
    int32_t head, tail, cap, size, ht_mask;
    uint64_t hits, reqs;
} edge0_cache_t;

static inline uint32_t edge0_hash(int32_t k) {
    uint32_t x = (uint32_t) k;
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    return x;
}

static inline void edge0_cache_init(edge0_cache_t * c, int dyn_cap) {
    memset(c, 0, sizeof(*c));
    c->cap = dyn_cap < 0 ? 0 : dyn_cap;
    int ht = 16;
    while (ht < (c->cap + 1) * 4) ht <<= 1;
    c->ht_mask = ht - 1;
    c->ht_key = (int32_t *) malloc(ht * sizeof(int32_t));
    c->ht_val = (int32_t *) malloc(ht * sizeof(int32_t));
    for (int i = 0; i < ht; i++) c->ht_key[i] = -1;
    c->prevv = (int32_t *) malloc((c->cap + 1) * sizeof(int32_t));
    c->nextv = (int32_t *) malloc((c->cap + 1) * sizeof(int32_t));
    c->node_key = (int32_t *) malloc((c->cap + 1) * sizeof(int32_t));
    c->head = c->tail = -1;
}

static inline void edge0_cache_pin(edge0_cache_t * c, int key) {
    // NOTE: modulo, NOT &(WORDS-1): 160 is not a power of two, so 159 is
    // not a valid mask (it clears bits 5-6 and aliases layers 8-15).
    // Exact for keys < 10240; folds gracefully above.
    c->pin_bits[((uint32_t) key >> 6) % EDGE0_PIN_WORDS] |= 1ULL << (key & 63);
}

static inline int edge0_cache_pinned(edge0_cache_t * c, int key) {
    return (c->pin_bits[((uint32_t) key >> 6) % EDGE0_PIN_WORDS] >> (key & 63)) & 1;
}

static int edge0_ht_find(edge0_cache_t * c, int32_t key) {
    uint32_t h = edge0_hash(key) & (uint32_t) c->ht_mask;
    while (c->ht_key[h] != -1) {
        if (c->ht_key[h] == key) return c->ht_val[h];
        h = (h + 1) & (uint32_t) c->ht_mask;
    }
    return -1;
}

static void edge0_ht_insert(edge0_cache_t * c, int32_t key, int32_t val) {
    uint32_t h = edge0_hash(key) & (uint32_t) c->ht_mask;
    int32_t tomb = -1;
    while (c->ht_key[h] != -1) {
        if (c->ht_key[h] == key) { c->ht_val[h] = val; return; }
        if (c->ht_key[h] == -2 && tomb < 0) tomb = (int32_t) h;
        h = (h + 1) & (uint32_t) c->ht_mask;
    }
    if (tomb >= 0) h = (uint32_t) tomb;
    c->ht_key[h] = key;
    c->ht_val[h] = val;
}

static void edge0_ht_remove(edge0_cache_t * c, int32_t key) {
    uint32_t h = edge0_hash(key) & (uint32_t) c->ht_mask;
    while (c->ht_key[h] != -1) {
        if (c->ht_key[h] == key) { c->ht_key[h] = -2; return; }
        h = (h + 1) & (uint32_t) c->ht_mask;
    }
}

static void edge0_lru_touch(edge0_cache_t * c, int32_t idx) {
    if (idx == c->head) return;
    int32_t p = c->prevv[idx], n = c->nextv[idx];
    if (p >= 0) c->nextv[p] = n; else return;
    if (n >= 0) c->prevv[n] = p;
    if (idx == c->tail) c->tail = p;
    c->prevv[idx] = -1;
    c->nextv[idx] = c->head;
    if (c->head >= 0) c->prevv[c->head] = idx;
    c->head = idx;
}

// Atomic K-event request (JOIN-2 runtime semantics): all n hits/misses are
// determined from the SAME pre-event cache state; current-event keys are
// protected from eviction while the event's misses are admitted.
// Order: (1) score all n (no mutation); (2) touch hits in request order;
// (3) admit misses in request order to head, victim = tail walking upward
// past any current-event key (degenerate cap<=n: plain tail). Pins never
// enter the dynamic list. Returns #hits in this event.
static int edge0_cache_event(edge0_cache_t * c, const int * keys, int n) {
    int hits = 0;
    for (int i = 0; i < n; i++) {
        c->reqs++;
        int k = keys[i];
        if (edge0_cache_pinned(c, k)) { c->hits++; hits++; continue; }
        if (c->cap > 0 && edge0_ht_find(c, k) >= 0) {
            c->hits++;
            hits++;
        }
    }
    if (c->cap <= 0) return hits;
    for (int i = 0; i < n; i++) {  // touch pre-event hits
        int k = keys[i];
        if (edge0_cache_pinned(c, k)) continue;
        int idx = edge0_ht_find(c, k);
        if (idx >= 0) edge0_lru_touch(c, idx);
    }
    for (int i = 0; i < n; i++) {  // admit misses
        int k = keys[i];
        if (edge0_cache_pinned(c, k)) continue;
        if (edge0_ht_find(c, k) >= 0) continue;  // was a hit (or admitted)
        int idx;
        if (c->size < c->cap) {
            idx = c->size++;
        } else {
            idx = c->tail;  // walk past protected current-event keys
            while (idx >= 0) {
                int kk = c->node_key[idx], prot = 0;
                for (int j = 0; j < n; j++)
                    if (keys[j] == kk) { prot = 1; break; }
                if (!prot) break;
                idx = c->prevv[idx];
            }
            if (idx < 0) idx = c->tail;  // degenerate: cap <= n
            edge0_ht_remove(c, c->node_key[idx]);
            int32_t p = c->prevv[idx], q = c->nextv[idx];
            if (p >= 0) c->nextv[p] = q; else c->head = q;
            if (q >= 0) c->prevv[q] = p; else c->tail = p;
        }
        c->node_key[idx] = k;
        edge0_ht_insert(c, k, idx);
        c->prevv[idx] = -1;
        c->nextv[idx] = c->head;
        if (c->head >= 0) c->prevv[c->head] = idx;
        c->head = idx;
        if (c->tail < 0) c->tail = idx;
    }
    return hits;
}

// Returns 1 on hit, 0 on miss (miss also admits the bundle).
static int edge0_cache_request(edge0_cache_t * c, int key) {
    c->reqs++;
    if (edge0_cache_pinned(c, key)) { c->hits++; return 1; }
    if (c->cap <= 0) return 0;
    int idx = edge0_ht_find(c, key);
    if (idx >= 0) {
        c->hits++;
        edge0_lru_touch(c, idx);
        return 1;
    }
    if (c->size < c->cap) {
        idx = c->size++;
    } else {
        idx = c->tail;
        edge0_ht_remove(c, c->node_key[idx]);
        int32_t p = c->prevv[idx];
        if (p >= 0) c->nextv[p] = -1;
        c->tail = p;
        if (c->head == idx) c->head = -1;
    }
    c->node_key[idx] = key;
    edge0_ht_insert(c, key, idx);
    c->prevv[idx] = -1;
    c->nextv[idx] = c->head;
    if (c->head >= 0) c->prevv[c->head] = idx;
    c->head = idx;
    if (c->tail < 0) c->tail = idx;
    return 0;
}

static inline void edge0_cache_free(edge0_cache_t * c) {
    free(c->ht_key);
    free(c->ht_val);
    free(c->prevv);
    free(c->nextv);
    free(c->node_key);
}

#endif
