#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "ggml.h"
#include "quants.h"
static float fp16_to_fp32(uint16_t h) {
    int s=(h>>15)&1, e=(h>>10)&31, m=h&1023;
    if(!e) return (s?-1:1)*(m/1024.0f)*0.00006103515625f;
    if(e==31) return 0;
    float v=(1+m/1024.0f); int ee=e-15;
    while(ee-->0) v*=2; return s?-v:v;
}
int main(void) {
    static unsigned char w[270336];
    FILE *f = fopen("/tmp/agent1_raw/L00_E000_gate.iq2xxs","rb");
    fread(w,1,270336,f); fclose(f);
    // row 0: 8 super-blocks of 66B: d=u16 at block start
    for (int i=0;i<8;i++) {
        uint16_t d; memcpy(&d, w+i*66, 2);
        printf("blk%d d=0x%04x f=%g\n", i, d, fp16_to_fp32(d));
    }
    float x[2048]; for (int i=0;i<2048;i++) x[i]=(i%7-3)*0.3f;
    static unsigned char xq[4096];
    quantize_row_q8_K(x, xq, 2048);
    for (int i=0;i<8;i++) printf("q8 blk%d d=%f\n", i, *(float*)(xq+i*292));
    return 0;
}
