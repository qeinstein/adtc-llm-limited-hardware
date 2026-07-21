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

echo "[profiler] Running participant-mode profile (Gate 1)..."
adtc-profiler run --submission "$ROOT" --mode participant \
    --output "$ROOT/submission.json" --skip-accuracy

echo "[profiler] Wrote submission.json. Review throughput / peak_rss before submitting."
