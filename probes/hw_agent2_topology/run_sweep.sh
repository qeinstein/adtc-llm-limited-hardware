#!/bin/bash
# Full topology sweep driver: rotated order x N windows, CSV results.
# Usage: ./run_sweep.sh [windows=3] [trials=30]
cd "$(dirname "$0")"
W=${1:-3}
T=${2:-30}
OUT=results/sweep_$(date +%Y%m%d_%H%M%S).csv
LOG=results/sweep_$(date +%Y%m%d_%H%M%S).log
echo "mask,nth,part,aff,pool,barrier,trials,med_ns,p10_ns,p90_ns,min_ns,mean_ns,cv,bal,mig,vctxt,ictxt,minflt,majflt,maxdiff,nbad" > "$OUT"
echo "== sweep: $W windows x $T trials -> $OUT ==" | tee "$LOG"
if [ ! -f ref/layer_ref.bin ]; then ./topo_sweep --dump-ref ref/layer_ref 2>>"$LOG" || exit 1; fi
# config list: mask|nth|part|aff|pool|barrier
CFGS=(
 "0xf|1|row|free|stream|futex"
 "0xf|2|row|free|stream|futex"
 "0xf|3|row|free|stream|futex"
 "0xf|4|row|free|stream|futex"
 "0xf|6|row|free|stream|futex"
 "0xf|8|row|free|stream|futex"
 "0xf|2|row|pin|stream|futex"
 "0xf|3|row|pin|stream|futex"
 "0xf|4|row|pin|stream|futex"
 "0xf|6|row|pin|stream|futex"
 "0xf|8|row|pin|stream|futex"
 "0xf|2|expert|free|stream|futex"
 "0xf|3|expert|free|stream|futex"
 "0xf|4|expert|free|stream|futex"
 "0xf|6|expert|free|stream|futex"
 "0xf|8|expert|free|stream|futex"
 "0xf|2|expert|pin|stream|futex"
 "0xf|3|expert|pin|stream|futex"
 "0xf|4|expert|pin|stream|futex"
 "0xf|6|expert|pin|stream|futex"
 "0xf|8|expert|pin|stream|futex"
 "0xf|4|row|free|stream|spin"
 "0xf|4|row|pin|stream|spin"
 "0xf|4|expert|free|stream|spin"
 "0xf|4|expert|pin|stream|spin"
 "0xf|8|expert|pin|stream|spin"
 "0x3|2|row|free|stream|spin"
 "0x3|2|row|pin|stream|spin"
 "0x3|2|expert|pin|stream|spin"
 "0x5|2|row|pin|stream|spin"
 "0x5|2|expert|pin|stream|spin"
 "0x1|1|row|pin|stream|spin"
 "0xf|4|row|free|hot|spin"
 "0xf|4|expert|free|hot|spin"
 "0xf|1|row|free|hot|spin"
)
for ((w=1; w<=W; w++)); do
  echo "--- window $w/$W ---" | tee -a "$LOG"
  # rotate order per window to defeat drift
  N=${#CFGS[@]}
  for ((i=0; i<N; i++)); do
    idx=$(( (i + w*7) % N ))
    IFS='|' read -r mask nth part aff pool bar <<< "${CFGS[$idx]}"
    line=$(taskset "$mask" ./topo_sweep "$nth" "$part" "$aff" "$T" ref/layer_ref "$pool" "$bar" 2>>"$LOG")
    echo "$mask,$line" >> "$OUT"
    echo "w$w $mask t$nth $part $aff $pool $bar: $line" >> "$LOG"
  done
done
echo "done -> $OUT"
