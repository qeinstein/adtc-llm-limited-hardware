#!/bin/bash
# stage matrix: policies x slot counts x traces. Raw logs to results/.
set -u
cd "$(dirname "$0")"
mkdir -p results
gcc -O3 -march=native -o stage_replay stage_replay.c /tmp/hw1/x86_quants.o /tmp/hw1/shim.o -lm || exit 1
i=0
for rep in 1 2; do
  for trace in 1.1 uniform; do
    for ns in 16 28 64; do
      for pol in base noDN huge hugeND; do
        i=$((i+1))
        out=results/stage_${pol}_s${ns}_${trace}_r${rep}.txt
        echo "[$i] $out"
        ./stage_replay $pol $ns 100 $trace $((12345+rep)) > $out 2>&1
      done
    done
  done
done
# parity references (infinite slots) for each trace/seed
for rep in 1 2; do
  for trace in 1.1 uniform; do
    ./stage_replay noslot 0 100 $trace $((12345+rep)) > results/stage_noslot_s0_${trace}_r${rep}.txt 2>&1
  done
done
echo DONE
grep -h "^policy\|^checksum\|^tok_ms\|^minflt_tot\|^req=" results/*.txt | paste - - - - - | sed 's/policy=//'
