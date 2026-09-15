#include <stdio.h>
#include <string.h>
#include "ggml.h"
#include "ggml-quants.h"
int main(void) {
    static unsigned char w[270336];
    FILE *f = fopen("/tmp/agent1_raw/L00_E000_gate.iq2xxs","rb");
    fread(w,1,270336,f); fclose(f);
    float x[2048]; for (int i=0;i<2048;i++) x[i]=(i%7-3)*0.3f;
    static float row[2048];
    dequantize_row_iq2_xxs((const void*)w, row, 2048);
    double s=0; for(int i=0;i<2048;i++) s+= (double)row[i]*x[i];
    printf("dequant+manual row0 = %f (row[0..2]=%f %f %f)\n", s, row[0], row[1], row[2]);
    return 0;
}
