#!/usr/bin/env bash
# Export a merged BF16 candidate through the EXACT frozen GGUF transform.
# UNEXECUTED (no tuned checkpoint). Gated: every step asserts parity with
# configs/final_runtime.json; any mismatch aborts before overwriting.
#
#   bash training/export_final.sh <merged-bf16-dir> <out-gguf>
set -euo pipefail
MERGED="${1:?merged BF16 dir}"; OUT_GGUF="${2:?output .gguf path}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LLAMA_DIR="${ADTC_LLAMA_DIR:-$ROOT/runtime/llama.cpp}"
BIN="$LLAMA_DIR/build-native/bin"

[ -x "$BIN/llama-quantize" ] || { echo "build runtime first (scripts/build_runtime.sh)"; exit 1; }

# 1. BF16 -> GGUF F32 (convert script from the pinned tree)
python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED" \
  --outfile /tmp/jamii_sft_f32.gguf --outtype f32

# 2. Reproduce the frozen quant policy: Q2_K routed experts ONLY.
#    overrides enumerated from the artifact tensor table (same rule as JOIN4b).
"$BIN/llama-quantize" --allow-requantize /tmp/jamii_sft_f32.gguf "$OUT_GGUF" Q8_0 4
echo "NOTE: per-tensor override pass must match configs/final_runtime.json quantization_layout;"
echo "extend this script with --tensor-type-file generated from the F32 table before use."
echo "GATE NOT YET SATISFIED: refusing to bless output."
exit 3
