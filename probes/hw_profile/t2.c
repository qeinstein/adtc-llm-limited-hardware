#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "quants.h"
int main(void) {
    ggml_quantize_init(GGML_TYPE_IQ2_XXS);
    static unsigned char w[270336];
    FILE *f = fopen("/tmp/agent1_raw/L00_E000_gate.iq2xxs","rb");
    fread(w,1,270336,f); fclose(f);
    static float W[512*2048];
    f = fopen("/tmp/agent1_f32/L00_E000_gate.f32","rb");
    fread(W,4,512*2048,f); fclose(f);
    float x[2048]; for (int i=0;i<2048;i++) x[i]=(i%7-3)*0.3f;
    // f32 reference row 0
    double ref=0; for (int i=0;i<2048;i++) ref += (double)W[i]*x[i];
    printf("ref row0 = %f\n", ref);
    static unsigned char xq[4096];
    quantize_row_q8_K(x, xq, 2048);
    int nz=0; for(int i=0;i<2336;i++) if(xq[i]) nz++;
    // check q8 scales sane: print first block
    printf("xq nonzero bytes few=%d/2336 d0=%f\n", nz, *(float*)xq);
    float y=0;
    ggml_vec_dot_iq2_xxs_q8_K(2048,&y,0,w,0,xq,0,1);
    printf("k_dot row0 = %f\n", y);
    float yg=0;
    ggml_vec_dot_iq2_xxs_q8_K_generic(2048,&yg,0,w,0,xq,0,1);
    printf("generic row0 = %f\n", yg);
    return 0;
}
