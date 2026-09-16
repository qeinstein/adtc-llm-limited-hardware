#!/bin/bash
# Build Phase-1 shootout. Proven recipe mirrored from probes/hw_profile/REPORT.md.
# Outputs go to /tmp/edge0_phase1 (scratch; not committed).
set -e
D=$(dirname "$0")
OUT=/tmp/edge0_phase1
mkdir -p "$OUT"
P=/tmp/llamacpp-pin/ggml
I="-I$D -I$P/include -I$P/src -I$P/src/ggml-cpu"
X86=$P/src/ggml-cpu/arch/x86/quants.c
test -f "$X86" || { echo "missing $X86"; exit 1; }

echo "== mini =="
gcc -O3 -march=native -D_GNU_SOURCE $I -c "$D/phase1_mini.c" -o "$OUT/mini.o"
echo "== ubench =="
gcc -O3 -march=native -D_GNU_SOURCE $I -c "$D/phase1_ubench.c" -o "$OUT/ubench.o"
echo "== ggml-quants =="
gcc -O3 -march=native -D_GNU_SOURCE $I -c "$P/src/ggml-quants.c" -o "$OUT/ggml-quants.o"
echo "== x86 quants =="
gcc -O3 -march=native -D_GNU_SOURCE $I -c "$X86" -o "$OUT/x86_quants.o"
echo "== ggml-cpu quants (dispatch) =="
gcc -O3 -march=native -D_GNU_SOURCE $I -c "$P/src/ggml-cpu/quants.c" -o "$OUT/cpu-quants.o"
echo "== tables shim =="
gcc -O3 -march=native -Dquantize_row_q8_K_ref=shim_unused_q8k_ref \
  -c "$D/../hw_agent1_kernel/ggml_tables_shim.c" -o "$OUT/shim.o"
echo "== link ubench =="
gcc -O3 -march=native -o "$OUT/ubench" "$OUT/ubench.o" "$OUT/mini.o" \
  "$OUT/ggml-quants.o" "$OUT/x86_quants.o" "$OUT/cpu-quants.o" "$OUT/shim.o" -lm -lpthread
echo "== qerr =="
gcc -O2 -march=native -D_GNU_SOURCE $I -c "$D/phase1_qerr.c" -o "$OUT/qerr.o"
gcc -O2 -march=native -o "$OUT/qerr" "$OUT/qerr.o" "$OUT/mini.o" \
  "$OUT/ggml-quants.o" "$OUT/cpu-quants.o" "$OUT/shim.o" -lm
echo "== overlap =="
gcc -O3 -march=native -D_GNU_SOURCE $I -c "$D/stage_overlap.c" -o "$OUT/overlap.o"
gcc -O3 -march=native -o "$OUT/overlap" "$OUT/overlap.o" "$OUT/mini.o" \
  "$OUT/ggml-quants.o" "$OUT/x86_quants.o" "$OUT/cpu-quants.o" "$OUT/shim.o" -lm -lpthread
echo "== hwpstat =="
gcc -O2 -o "$OUT/hwpstat" "$D/../hw_profile/hwpstat.c"
ls -la "$OUT/ubench" "$OUT/qerr" "$OUT/overlap" "$OUT/hwpstat"
