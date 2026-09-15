#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static volatile unsigned long long sink;
static void *spin(void *a){ (void)a;
    double e = now_s() + 1.0;
    unsigned long long x = 0;
    while (now_s() < e) for (int i=0;i<1000000;i++) x += i * 2654435761ULL;
    sink = x; return NULL;
}
int main(int argc, char **argv){
    int n = argc>1 ? atoi(argv[1]) : 1;
    if (n > 8) n = 8;
    printf("READY\n"); fflush(stdout);
    const char *fd = getenv("HWP_SYNC_FD");
    if (fd) { char b; if(read(atoi(fd), &b, 1)!=1) return 1; }
    pthread_t t[8];
    double s = now_s();
    for (int i=0;i<n;i++) pthread_create(&t[i],NULL,spin,NULL);
    for (int i=0;i<n;i++) pthread_join(t[i],NULL);
    printf("spin wall=%.3f\n", now_s()-s);
    return 0;
}
