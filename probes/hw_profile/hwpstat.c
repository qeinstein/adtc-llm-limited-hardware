/* hwpstat: minimal perf-stat-like wrapper using perf_event_open(2).
 * No perf binary needed. Opens counters on a forked child (inherit across
 * threads), runs the command, prints raw counts + derived rates.
 * Usage: hwpstat -- <cmd> [args...]
 * Env: HWP_EVENTS=comma list to override default set; HWP_VERBOSE=1.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <linux/perf_event.h>
#include <time.h>

static int p_open(struct perf_event_attr *a, pid_t pid) {
    return syscall(__NR_perf_event_open, a, pid, -1, -1, 0);
}

struct ev { const char *name; uint32_t type; uint64_t config; int fds[64]; int nfds; uint64_t val; int ok; };

#define HW(n,c)  { n, PERF_TYPE_HARDWARE, PERF_COUNT_HW_##c, -1, 0, 0 }
#define SW(n,c)  { n, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_##c, -1, 0, 0 }
#define HWC(n,c) { n, PERF_TYPE_HW_CACHE, c, -1, 0, 0 }

static uint64_t hwc(uint64_t cache, uint64_t op, uint64_t res) {
    return cache | (op << 8) | (res << 16);
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int cut = -1;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--")) { cut = i; break; }
    if (cut < 0 || cut + 1 >= argc) {
        fprintf(stderr, "usage: %s -- <cmd> [args...]\n", argv[0]);
        return 2;
    }
    char **cmd = &argv[cut + 1];
    const char *only = getenv("HWP_EVENTS");

    struct ev evs[22 + 16] = {
        HW("cycles", CPU_CYCLES),
        HW("instructions", INSTRUCTIONS),
        HW("cache_references", CACHE_REFERENCES),
        HW("cache_misses", CACHE_MISSES),
        HW("branches", BRANCH_INSTRUCTIONS),
        HW("branch_misses", BRANCH_MISSES),
        HW("stalled_frontend", STALLED_CYCLES_FRONTEND),
        HW("stalled_backend", STALLED_CYCLES_BACKEND),
        HWC("L1_dcache_loads", 0),
        HWC("L1_dcache_load_misses", 0),
        HWC("LLC_loads", 0),
        HWC("LLC_load_misses", 0),
        HWC("dTLB_loads", 0),
        HWC("dTLB_load_misses", 0),
        HWC("LLC_stores", 0),
        HWC("LLC_store_misses", 0),
        SW("task_clock", TASK_CLOCK),
        SW("page_faults", PAGE_FAULTS),
        SW("minor_faults", PAGE_FAULTS_MIN),
        SW("major_faults", PAGE_FAULTS_MAJ),
        SW("context_switches", CONTEXT_SWITCHES),
        SW("cpu_migrations", CPU_MIGRATIONS),
    };
    evs[8].config  = hwc(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_ACCESS);
    evs[9].config  = hwc(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_MISS);
    evs[10].config = hwc(PERF_COUNT_HW_CACHE_LL,  PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_ACCESS);
    evs[11].config = hwc(PERF_COUNT_HW_CACHE_LL,  PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_MISS);
    evs[12].config = hwc(PERF_COUNT_HW_CACHE_DTLB,PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_ACCESS);
    evs[13].config = hwc(PERF_COUNT_HW_CACHE_DTLB,PERF_COUNT_HW_CACHE_OP_READ,  PERF_COUNT_HW_CACHE_RESULT_MISS);
    evs[14].config = hwc(PERF_COUNT_HW_CACHE_LL,  PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_ACCESS);
    evs[15].config = hwc(PERF_COUNT_HW_CACHE_LL,  PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_MISS);
    int nev = 22; /* base table entries above */
    /* raw events: HWP_RAW="name:code:umask,..." (hex), appended to table */
    struct ev extra[16]; int nx = 0;
    const char * raw = getenv("HWP_RAW");
    if (raw) {
        char * dup = strdup(raw), * save = NULL;
        for (char * tok = strtok_r(dup, ",", &save); tok && nx < 16; tok = strtok_r(NULL, ",", &save)) {
            char nm[48]; unsigned code, umask;
            if (sscanf(tok, "%47[^:]:%x:%x", nm, &code, &umask) != 3) {
                fprintf(stderr, "bad HWP_RAW tok %s\n", tok); return 2;
            }
            extra[nx].name = strdup(nm);
            extra[nx].type = PERF_TYPE_RAW;
            extra[nx].config = code | ((uint64_t) umask << 8);
            extra[nx].nfds = 0; extra[nx].val = 0; extra[nx].ok = 0;
            nx++;
        }
        free(dup);
    }
    for (int i = 0; i < nx; i++) evs[nev++] = extra[i];

    char *req = NULL;
    if (only) req = strdup(only);

    int syncmode = getenv("HWP_SYNC") != NULL;
    int out_p[2], go_p[2];
    if (syncmode) {
        if (pipe(out_p) || pipe(go_p)) { perror("pipe"); return 1; }
    }
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) {
        if (syncmode) {
            dup2(out_p[1], 1); dup2(out_p[1], 2);
            close(out_p[0]); close(out_p[1]); close(go_p[1]);
            char fdbuf[16]; snprintf(fdbuf, sizeof(fdbuf), "%d", go_p[0]);
            setenv("HWP_SYNC_FD", fdbuf, 1);
            unsetenv("HWP_SYNC");
            /* fds survive exec (no CLOEXEC) */
        } else {
            raise(SIGSTOP);
        }
        execvp(cmd[0], cmd);
        perror("execvp"); _exit(127);
    }
    int st;
    FILE * cap = NULL;
    if (syncmode) {
        close(out_p[1]); close(go_p[0]);
        cap = fdopen(out_p[0], "r");
        char line[4096];
        /* forward child output; wait for READY marker */
        while (fgets(line, sizeof(line), cap)) {
            fputs(line, stdout);
            if (!strncmp(line, "READY", 5)) break;
        }
    } else {
        // child stopped itself; attach counters then continue
        waitpid(pid, &st, WUNTRACED);
    }

    /* enumerate all live threads: inherit(1) misses threads born before open */
    pid_t tids[64]; int ntids = 0;
    {
        char taskdir[64]; snprintf(taskdir, sizeof(taskdir), "/proc/%d/task", pid);
        DIR * dp = opendir(taskdir);
        if (dp) {
            struct dirent * de;
            while ((de = readdir(dp)) && ntids < 64) {
                if (de->d_name[0] == '.') continue;
                tids[ntids++] = atoi(de->d_name);
            }
            closedir(dp);
        }
        if (ntids == 0) tids[ntids++] = pid;
        fprintf(stderr, "# threads attached: %d\n", ntids);
    }
    struct perf_event_attr attr;
    for (int i = 0; i < nev; i++) {
        if (req && !strstr(req, evs[i].name)) continue;
        evs[i].nfds = 0; evs[i].ok = 0;
        for (int t = 0; t < ntids; t++) {
            memset(&attr, 0, sizeof(attr));
            attr.size = sizeof(attr);
            attr.type = evs[i].type;
            attr.config = evs[i].config;
            attr.disabled = 1;
            attr.inherit = 1;
            attr.exclude_kernel = 1;
            attr.exclude_hv = 1;
            int fd = p_open(&attr, tids[t]);
            if (fd < 0) {
                if (getenv("HWP_VERBOSE")) fprintf(stderr, "# open %-18s tid=%d failed: %s\n", evs[i].name, tids[t], strerror(errno));
            } else {
                evs[i].fds[evs[i].nfds++] = fd;
                evs[i].ok = 1;
            }
        }
    }
    for (int i = 0; i < nev; i++) for (int k = 0; k < evs[i].nfds; k++) ioctl(evs[i].fds[k], PERF_EVENT_IOC_RESET, 0);
    for (int i = 0; i < nev; i++) for (int k = 0; k < evs[i].nfds; k++) ioctl(evs[i].fds[k], PERF_EVENT_IOC_ENABLE, 0);
    double t0 = now_s();
    if (syncmode) {
        char b = 0;
        if (write(go_p[1], &b, 1) != 1) { perror("go"); return 1; }
        close(go_p[1]);
    } else {
        kill(pid, SIGCONT);
    }
    int status = 0;
    if (syncmode) {
        /* drain remaining child output while waiting */
        char line[4096];
        /* use nonblocking wait loop */
        pid_t w;
        int flags = fcntl(fileno(cap), F_GETFL, 0);
        fcntl(fileno(cap), F_SETFL, flags | O_NONBLOCK);
        double deadline = now_s() + 3600;
        for (;;) {
            w = waitpid(pid, &status, WNOHANG);
            while (fgets(line, sizeof(line), cap)) fputs(line, stdout);
            if (w == pid) break;
            if (now_s() > deadline) { fprintf(stderr, "timeout\n"); return 1; }
            usleep(20000);
        }
        while (fgets(line, sizeof(line), cap)) fputs(line, stdout);
    } else {
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    }
    double wall = now_s() - t0;
    for (int i = 0; i < nev; i++) for (int k = 0; k < evs[i].nfds; k++) ioctl(evs[i].fds[k], PERF_EVENT_IOC_DISABLE, 0);
    for (int i = 0; i < nev; i++) {
        if (!evs[i].ok) continue;
        uint64_t sum = 0;
        for (int k = 0; k < evs[i].nfds; k++) {
            uint64_t v = 0;
            if (read(evs[i].fds[k], &v, sizeof(v)) == sizeof(v)) sum += v;
            close(evs[i].fds[k]);
        }
        evs[i].val = sum;
    }
    printf("# cmd: ");
    for (char **c = cmd; *c; c++) printf("%s ", *c);
    printf("\n# wall_sec=%.6f child_status=%d\n", wall,
           WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    printf("# nn: counter not opened in this environment\n");
    for (int i = 0; i < nev; i++) {
        if (req && !strstr(req, evs[i].name)) continue;
        if (evs[i].ok) printf("%-18s %" PRIu64 "\n", evs[i].name, evs[i].val);
        else printf("%-18s nn\n", evs[i].name);
    }
    // derived
    uint64_t cyc = 0, ins = 0; int hc = 0, hi = 0;
    for (int i = 0; i < nev; i++) {
        if (!strcmp(evs[i].name, "cycles") && evs[i].ok) { cyc = evs[i].val; hc = 1; }
        if (!strcmp(evs[i].name, "instructions") && evs[i].ok) { ins = evs[i].val; hi = 1; }
    }
    if (hc && hi && cyc) printf("# IPC=%.4f\n", (double) ins / (double) cyc);
    free(req);
    return 0;
}
