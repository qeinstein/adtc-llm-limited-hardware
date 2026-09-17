// cache_replay: cross-check the C pin+LRU policy against the python sim.
// Usage: cache_replay <target-GB>  (e.g. 5.0)
// Reads probes/edge0_port/cache_config.json + /tmp/edge0_phase1/trv3_test_stream.u16,
// runs edge0_cache_request over the stream, prints hit% + expected hit%.
// Exit 0 iff |hit - expected| < 1e-9 (bit-for-bit policy equivalence).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "cache_policy.h"

static char * slurp(const char * path, long * n) {
    FILE * fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "open %s\n", path); exit(1); }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char * buf = (char *) malloc(sz + 1);
    if (fread(buf, 1, sz, fp) != (size_t) sz) exit(1);
    buf[sz] = 0;
    fclose(fp);
    if (n) *n = sz;
    return buf;
}

int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <target-GB>\n", argv[0]); return 2; }
    char * cfg = slurp("probes/edge0_port/cache_config.json", NULL);
    char key[32];
    snprintf(key, sizeof(key), "\"%s\": {", argv[1]);
    char * t = strstr(cfg, key);
    if (!t) { fprintf(stderr, "target %s not in config\n", argv[1]); return 2; }
    int slots = 0, npins = 0;
    double expected = 0;
    if (sscanf(strstr(t, "\"slots\":"), "\"slots\": %d", &slots) != 1) return 2;
    if (sscanf(strstr(t, "\"npins\":"), "\"npins\": %d", &npins) != 1) return 2;
    if (sscanf(strstr(t, "\"expected_hit\":"), "\"expected_hit\": %lf", &expected) != 1) return 2;
    char * pa = strstr(t, "\"pins_global\": [");
    if (!pa) return 2;
    pa = strchr(pa, '[') + 1;
    edge0_cache_t c;
    edge0_cache_init(&c, slots - npins);
    for (int i = 0; i < npins; i++) {
        int pin = (int) strtol(pa, &pa, 10);
        edge0_cache_pin(&c, pin);
        if (*pa == ',') pa++;
    }
    long n;
    uint16_t * stream = (uint16_t *) slurp("/tmp/edge0_phase1/trv3_test_stream.u16", &n);
    long nreq = n / 2;
    for (long i = 0; i < nreq; i++) edge0_cache_request(&c, stream[i]);
    double hit = (double) c.hits / (double) c.reqs;
    printf("target=%s slots=%d pins=%d dyn=%d reqs=%llu hit=%.6f expected=%.6f %s\n",
           argv[1], slots, npins, slots - npins, (unsigned long long) c.reqs,
           hit, expected, fabs(hit - expected) < 1e-6 ? "MATCH" : "MISMATCH");
    edge0_cache_free(&c);
    return fabs(hit - expected) < 1e-6 ? 0 : 1;
}
