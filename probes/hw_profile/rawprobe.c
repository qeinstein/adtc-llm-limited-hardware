/* rawprobe: try opening raw PMU events, run an AVX2 loop, report counts.
 * Usage: rawprobe code:umask:name [code:umask:name ...]  (hex, e.g. c7:04:fp128)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <immintrin.h>

static int p_open(struct perf_event_attr *a) {
    return syscall(__NR_perf_event_open, a, 0, -1, -1, 0);
}

__attribute__((noinline)) static double avx2_loop(float *a, int n, int iters) {
    __m256 acc = _mm256_setzero_ps();
    for (int it = 0; it < iters; it++)
        for (int i = 0; i < n; i += 8) {
            __m256 x = _mm256_loadu_ps(a + i);
            acc = _mm256_fmadd_ps(x, x, acc);
        }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: rawprobe code:umask:name ...\n"); return 2; }
    int n = argc - 1;
    int *fds = calloc(n, sizeof(int));
    struct perf_event_attr attr;
    for (int i = 0; i < n; i++) {
        unsigned code, umask; char name[64];
        if (sscanf(argv[i+1], "%x:%x:%63s", &code, &umask, name) != 3) {
            fprintf(stderr, "bad spec %s\n", argv[i+1]); return 2;
        }
        memset(&attr, 0, sizeof(attr));
        attr.size = sizeof(attr);
        attr.type = PERF_TYPE_RAW;
        attr.config = code | ((uint64_t)umask << 8);
        attr.disabled = 0;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        fds[i] = p_open(&attr);
        printf("%-28s open=%s", argv[i+1], fds[i] < 0 ? strerror(errno) : "OK");
        if (fds[i] >= 0) ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
        printf("\n");
    }
    static float buf[4096] __attribute__((aligned(64)));
    for (int i = 0; i < 4096; i++) buf[i] = (float)i * 0.25f;
    for (int i = 0; i < n; i++) if (fds[i] >= 0) ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
    volatile double r = avx2_loop(buf, 4096, 200);
    for (int i = 0; i < n; i++) if (fds[i] >= 0) ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
    printf("checksum=%f\n", r);
    for (int i = 0; i < n; i++) {
        if (fds[i] < 0) continue;
        uint64_t v = 0;
        read(fds[i], &v, sizeof(v));
        printf("%-28s count=%lu\n", argv[i+1], (unsigned long)v);
        close(fds[i]);
    }
    free(fds);
    return 0;
}
