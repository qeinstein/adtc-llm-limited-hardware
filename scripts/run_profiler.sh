#!/usr/bin/env bash
# Run the official ADTC profiler against this submission (Gate-1 self-check).
# ---------------------------------------------------------------------------
# Installs the profiler and runs it in participant mode. Requires llama-bench on
# PATH (build one with scripts/build_llamacpp_scalar.sh for audit parity).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

if ! python -c "import adtc_profiler" 2>/dev/null; then
    echo "[profiler] Installing adtc-profiler..."
    pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
fi

if ! command -v llama-bench >/dev/null 2>&1; then
    echo "[profiler] WARNING: llama-bench not on PATH."
    echo "           Build the audit-parity binary first: bash scripts/build_llamacpp_scalar.sh"
    echo "           then add it to PATH, e.g.:"
    echo "           export PATH=\"$ROOT/llama.cpp/build-scalar/bin:\$PATH\""
fi

echo "[profiler] Ensuring model is present..."
bash "$ROOT/download_model.sh"

# The frozen bounded runtime is env-gated (same contract as CI): export it so
# a patched llama-bench on PATH profiles the bounded arm. Stock binaries
# ignore unknown env vars, so audit-parity runs are unaffected.
eval "$(cd "$ROOT" && python3 - <<'PY'
from src.sparse import build_env, export_pins
pins = export_pins("3.0")
for k, v in build_env("bounded_3gb", str(pins)).items():
    print(f"export {k}='{v}'")
PY
)"
echo "[profiler] bounded env: GGML_MOE_K1=$GGML_MOE_K1 GGML_MOE_K2=$GGML_MOE_K2 SLOTS=$GGML_PHASE6_SLOTS LAZY=$LLAMA_ARG_LAZY_MODE"

echo "[profiler] Running participant-mode profile (Gate 1)..."
adtc-profiler run --submission "$ROOT" --mode participant \
    --output "$ROOT/submission.json" --skip-accuracy

echo "[profiler] Wrote submission.json. Review throughput / peak_rss before submitting."
