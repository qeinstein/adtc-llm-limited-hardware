#!/usr/bin/env bash
# Local perf sweep: llama-bench (CPU, -ngl 0, profiler settings -p512 -n128) over
# every complete GGUF in the scratch dir. Produces a tg128 (decode tps) + pp512
# table for the model/quant A/B. NOTE: on Apple Silicon this is ARM-NEON CPU, a
# PROXY — absolute tps is far higher than the x86 scalar audit build; use it for
# RELATIVE ordering (size, quant). For audit-accurate numbers, run the same sweep
# with the scalar build from scripts/build_llamacpp_scalar.sh on an x86 box.
set -uo pipefail

BIN="${LLAMA_BIN:-$HOME/adtc_models/llama.cpp/build/bin}"
GDIR="${GGUF_DIR:-$HOME/adtc_models/gguf}"
BENCH="$BIN/llama-bench"

[ -x "$BENCH" ] || { echo "llama-bench not found at $BENCH"; exit 1; }

printf "%-32s %10s %12s %12s\n" "model" "size" "pp512 t/s" "tg128 t/s"
printf '%.0s-' {1..70}; echo
for f in "$GDIR"/*.gguf; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    size=$(du -h "$f" | cut -f1)
    # -ngl 0 = CPU only; -r 3 keeps it quick; parse the json rows
    out=$("$BENCH" -m "$f" -p 512 -n 128 -ngl 0 -r 3 --output json 2>/dev/null)
    pp=$(printf '%s' "$out" | grep -o '"avg_ts"[^,]*' | sed -n '1p' | grep -o '[0-9.]*')
    tg=$(printf '%s' "$out" | grep -o '"avg_ts"[^,]*' | sed -n '2p' | grep -o '[0-9.]*')
    printf "%-32s %10s %12s %12s\n" "$name" "$size" "${pp:-?}" "${tg:-?}"
done
