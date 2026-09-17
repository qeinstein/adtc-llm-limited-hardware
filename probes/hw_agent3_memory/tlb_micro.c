// tlb_micro: (1) CPUID leaf 0x18 DTLB/STLB geometry; (2) micro-cost of the
// v10 evict path: MADV_DONTNEED + re-fault (memcpy) of one 876544 B bundle
// vs plain overwrite; (3) pagemap unique-present-page count of a staged
// region (TLB footprint proxy; PMU absent on this KVM box).
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cpuid.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define SLOTB 876544
#define NREP 200

static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
static int cmp_dbl(const void * a, const void * b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void dump_tlb(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid_max(0x18, NULL)) { printf("cpuid_0x18: unavailable\n"); return; }
    __cpuid_count(0x18, 0, a, b, c, d);
    printf("cpuid_0x18_maxsub=%u\n", a);
    for (unsigned s = 1; s <= a && s < 32; s++) {
        __cpuid_count(0x18, s, a, b, c, d);
        unsigned type = a & 31;
        if (type == 0) continue;
        /* b[15:0]=ways? Intel: EBX[15:0] = #sets? Actually: EAX: type;
         * EBX[15:0]=ways, EBX[31:16]=partitions, ECX=#sets, EDX: page sizes */
        const char * t = (type == 1) ? "DTLB" : (type == 2) ? "ITLB" : (type == 3) ? "STLB" : "?TLB";
        unsigned ways = b & 0xffff, parts = (b >> 16) & 0xffff;
        printf("tlb sub=%u type=%s ways=%u partitions=%u sets=%u pagesizes=0x%x entries~%u\n",
            s, t, ways, parts, c, d, ways * parts * c);
    }
}

int main(void) {
    dump_tlb();
    printf("---\n");

    /* source bytes (pagecache-hot analog) */
    uint8_t * src = mmap(NULL, SLOTB, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (size_t i = 0; i < SLOTB; i++) src[i] = (uint8_t)(i * 31 + 7);

    uint8_t * slot = mmap(NULL, SLOTB, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    memset(slot, 0, SLOTB); /* first touch */

    double t_madv[NREP], t_refill[NREP], t_over[NREP];
    struct rusage r0, r1;
    /* arm 1: DONTNEED + memcpy refill (v10 evict path) */
    getrusage(RUSAGE_SELF, &r0);
    for (int i = 0; i < NREP; i++) {
        double t0 = now_ns();
        madvise(slot, SLOTB, MADV_DONTNEED);
        double t1 = now_ns();
        memcpy(slot, src, SLOTB);
        double t2 = now_ns();
        t_madv[i] = t1 - t0; t_refill[i] = t2 - t1;
    }
    getrusage(RUSAGE_SELF, &r1);
    long flt_dn = r1.ru_minflt - r0.ru_minflt;
    qsort(t_madv, NREP, sizeof(double), cmp_dbl);
    qsort(t_refill, NREP, sizeof(double), cmp_dbl);
    printf("dontneed_path: madvise_us_med=%.2f refill_memcpy_us_med=%.2f sum_us_med=%.2f minflt_per_iter=%.1f\n",
        t_madv[NREP/2]/1e3, t_refill[NREP/2]/1e3,
        (t_madv[NREP/2]+t_refill[NREP/2])/1e3, (double)flt_dn/NREP);

    /* arm 2: plain overwrite (noDN) */
    getrusage(RUSAGE_SELF, &r0);
    for (int i = 0; i < NREP; i++) {
        double t0 = now_ns();
        memcpy(slot, src, SLOTB);
        double t1 = now_ns();
        t_over[i] = t1 - t0;
    }
    getrusage(RUSAGE_SELF, &r1);
    long flt_ov = r1.ru_minflt - r0.ru_minflt;
    qsort(t_over, NREP, sizeof(double), cmp_dbl);
    printf("overwrite_path: memcpy_us_med=%.2f minflt_per_iter=%.2f\n",
        t_over[NREP/2]/1e3, (double)flt_ov/NREP);
    printf("saving_per_miss_us=%.1f (%.1fx)\n",
        (t_madv[NREP/2]+t_refill[NREP/2]-t_over[NREP/2])/1e3,
        (t_madv[NREP/2]+t_refill[NREP/2])/t_over[NREP/2]);

    /* arm 3: overwrite when slot+src are cache-cold but PTE-present
     * (stream 256 MB eviction buffer between iters; TLB also largely cold).
     * Upper-bounds refill cost in the production async regime. */
    size_t evn = 256u << 20;
    uint8_t * evb = mmap(NULL, evn, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(evb, 1, evn);
    double t_cold[40];
    volatile uint8_t sink = 0;
    for (int i = 0; i < 40; i++) {
        for (size_t o = 0; o < evn; o += 64) sink += evb[o];
        double t0 = now_ns();
        memcpy(slot, src, SLOTB);
        double t1 = now_ns();
        t_cold[i] = t1 - t0;
    }
    qsort(t_cold, 40, sizeof(double), cmp_dbl);
    printf("cold_overwrite: memcpy_us_med=%.1f (PTE-present, cache/TLB-cold; sink=%u)\n",
        t_cold[20]/1e3, sink);

    /* pagemap: unique present pages of an 8-slot staged window (1 layer) */
    size_t winslots = 8;
    uint8_t * win = mmap(NULL, SLOTB * winslots, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    for (size_t s = 0; s < winslots; s++)
        memcpy(win + s * SLOTB, src, SLOTB);
    int pf = open("/proc/self/pagemap", O_RDONLY);
    long present = 0;
    if (pf >= 0) {
        for (size_t off = 0; off < SLOTB * winslots; off += 4096) {
            uint64_t e = 0;
            uintptr_t va = (uintptr_t)(win + off);
            if (pread(pf, &e, 8, (off_t)((va / 4096) * 8)) == 8 && (e & (1ULL << 63))) present++;
        }
        close(pf);
    }
    printf("pagemap: staged_8slot_present_pages=%ld of %zu (%.2f MB)\n",
        present, SLOTB * winslots / 4096, present * 4.0 / 1024);
    return 0;
}
