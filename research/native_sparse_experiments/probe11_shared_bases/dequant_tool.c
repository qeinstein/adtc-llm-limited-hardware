// Dequant one GGUF tensor slice (single quant type) to raw F32.
// Usage: dequant in.bin out.f32 <iq2_xxs|iq2_s> ne0 nrows
// in.bin holds rows back-to-back (no header); out is ne0*nrows float32.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "ggml-quants.h"

// Minimal stubs for symbols quants.c references but the dequant path never calls.
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    (void)file; (void)line; (void)fmt;
    fprintf(stderr, "ggml_abort called\n"); abort();
}
size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    (void)type; (void)ne;
    fprintf(stderr, "ggml_row_size stub called\n"); abort(); return 0;
}
size_t ggml_type_size(enum ggml_type type) {
    (void)type;
    fprintf(stderr, "ggml_type_size stub called\n"); abort(); return 0;
}
const char * ggml_type_name(enum ggml_type type) {
    (void)type;
    fprintf(stderr, "ggml_type_name stub called\n"); abort(); return 0;
}

int main(int argc, char ** argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s in out type ne0 nrows\n", argv[0]); return 1; }
    const char * pin = argv[1], * pout = argv[2], * type = argv[3];
    long ne0 = atol(argv[4]), nrows = atol(argv[5]);
    int is_xxs = !strcmp(type, "iq2_xxs");
    int is_s = !strcmp(type, "iq2_s");
    if ((!is_xxs && !is_s) || ne0 % 256 != 0 || nrows <= 0) { fprintf(stderr, "bad args\n"); return 1; }
    size_t blk = is_xxs ? 66 : 82;  // bytes per 256 values (QK_K=256)
    size_t rowbytes = (size_t)(ne0 / 256) * blk;
    FILE * fi = fopen(pin, "rb");
    if (!fi) { perror("open in"); return 1; }
    float * out = malloc((size_t)ne0 * nrows * sizeof(float));
    uint8_t * row = malloc(rowbytes);
    if (!out || !row) { fprintf(stderr, "oom\n"); return 1; }
    for (long r = 0; r < nrows; r++) {
        if (fread(row, 1, rowbytes, fi) != rowbytes) { fprintf(stderr, "short read row %ld\n", r); return 1; }
        if (is_xxs) dequantize_row_iq2_xxs(row, out + r * ne0, ne0);
        else dequantize_row_iq2_s(row, out + r * ne0, ne0);
    }
    fclose(fi);
    FILE * fo = fopen(pout, "wb");
    if (!fo) { perror("open out"); return 1; }
    fwrite(out, sizeof(float), (size_t)ne0 * nrows, fo);
    fclose(fo);
    printf("dequant ok: %ld rows x %ld -> %s\n", nrows, ne0, pout);
    return 0;
}
