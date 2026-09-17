#define _GNU_SOURCE
#include <stdio.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
int main(int argc, char **argv) {
    pid_t p = fork();
    if (p == 0) { execvp(argv[1], argv+1); _exit(127); }
    int st; struct rusage ru;
    wait4(p, &st, 0, &ru);
    double u = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec/1e6;
    double s = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec/1e6;
    printf("rusage: utime=%.3f stime=%.3f minflt=%ld majflt=%ld nvcsw=%ld nivcsw=%ld\n", u, s, ru.ru_minflt, ru.ru_majflt, ru.ru_nvcsw, ru.ru_nivcsw);
    return 0;
}
