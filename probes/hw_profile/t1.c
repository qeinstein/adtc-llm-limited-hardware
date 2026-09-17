#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "quants.h"
extern void ggml_quantize_init(void);
int main(void) {
    ggml_quantize_init();
    FILE *f = fopen("/tmp/agent1_raw/L00_E112_gate.iq2xxs","rb");
    if (!f) { printf("no file\n"); return 1; }
    static unsigned char w[270336];
    if (fread(w,1,270336,f)!=270336) { printf("short\n"); return 1; }
    fclose(f);
    float x[2048]; for (int i=0;i<2048;i++) x[i]=(i%7-3)*0.3f;
    static unsigned char xq[4096];
    quantize_row_q8_K(x, xq, 2048);
    float y[512]; memset(y,0,sizeof y);
    for (int r=0;r<512;r++) ggml_vec_dot_iq2_xxs_q8_K(2048,&y[r],0,w+r*528,0,xq,0,1);
    double s=0; for(int i=0;i<512;i++) s+=y[i];
    printf("y[0..3]=%f %f %f %f sum=%f\n", y[0],y[1],y[2],y[3], s);
    return 0;
}
