// cache_atomic_test: run the C pin+LRU policy over an event stream.
// Usage: cache_atomic_test <dyn_cap> <stream.bin> <atomic|seq> [pins.bin]
// stream.bin: u32 n_events, u32 k, then keys u16[n_events*k].
// pins.bin (optional): u32 npins, then pins u16[npins].
// Prints: hits <h> reqs <r>. Exit 0 always (comparison done by caller).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "cache_policy.h"

static uint8_t * slurp(const char * path, long * n) {
    FILE * fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "open %s\n", path); exit(1); }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t * buf = (uint8_t *) malloc(sz ? sz : 1);
    if (sz && fread(buf, 1, sz, fp) != (size_t) sz) exit(1);
    fclose(fp);
    if (n) *n = sz;
    return buf;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <dyn_cap> <stream.bin> "
                "<atomic|seq> [pins.bin]\n", argv[0]);
        return 2;
    }
    int dyn = atoi(argv[1]), atomic = !strcmp(argv[3], "atomic");
    long n;
    uint8_t * s = slurp(argv[2], &n);
    uint32_t ne = ((uint32_t *) s)[0], k = ((uint32_t *) s)[1];
    uint16_t * keys = (uint16_t *) (s + 8);
    edge0_cache_t c;
    edge0_cache_init(&c, dyn);
    if (argc > 4) {
        long np;
        uint8_t * pb = slurp(argv[4], &np);
        uint32_t nn = ((uint32_t *) pb)[0];
        uint16_t * pp = (uint16_t *) (pb + 4);
        for (uint32_t i = 0; i < nn; i++) edge0_cache_pin(&c, pp[i]);
        free(pb);
    }
    int * ev = (int *) malloc(k * sizeof(int));
    for (uint32_t e = 0; e < ne; e++) {
        for (uint32_t i = 0; i < k; i++) ev[i] = keys[e * k + i];
        if (atomic) {
            edge0_cache_event(&c, ev, (int) k);
        } else {
            for (uint32_t i = 0; i < k; i++) edge0_cache_request(&c, ev[i]);
        }
    }
    printf("hits %llu reqs %llu\n", (unsigned long long) c.hits,
           (unsigned long long) c.reqs);
    free(ev);
    free(s);
    edge0_cache_free(&c);
    return 0;
}
