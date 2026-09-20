#!/usr/bin/env bash
# Build the FROZEN Jamii Afya sparse runtime: pinned llama.cpp + edge0 patch
# set (K4/16 graph patch, lazy experts, JOIN4 bounded executor, CLI precision).
# Idempotent: reuses the checkout when the pin + patch stamp match.
#
#   bash scripts/build_runtime.sh [llama-dir]
#
# Binaries: llama-server (web UI backend), llama-cli (repro/bench),
# llama-bench (adtc-profiler throughput/memory).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LLAMA_DIR="${1:-${ADTC_LLAMA_DIR:-$ROOT/runtime/llama.cpp}}"
PIN="3057bb66c86c46d5781e50e85462a760ba7d1feb"
BUILD="$LLAMA_DIR/build-native"
STAMP="$LLAMA_DIR/.edge0-stamp"

header_sha() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$ROOT/probes/edge0_port/join4_phase6.h" | awk '{print $1}'
    else shasum -a 256 "$ROOT/probes/edge0_port/join4_phase6.h" | awk '{print $1}'; fi
}

if [ ! -d "$LLAMA_DIR/.git" ]; then
    echo "[build_runtime] cloning llama.cpp -> $LLAMA_DIR"
    mkdir -p "$(dirname "$LLAMA_DIR")"
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
git fetch --depth 1 origin "$PIN"
git checkout --detach "$PIN"
HEAD="$(git rev-parse HEAD)"
[ "$HEAD" = "$PIN" ] || { echo "[build_runtime] ERROR: NOT on pin: $HEAD" >&2; exit 1; }
echo "[build_runtime] on pin $HEAD"

WANT="$PIN $(header_sha)"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$WANT" ]; then
    echo "[build_runtime] patches already applied (stamp match), skipping"
else
    if [ -f "$STAMP" ]; then
        echo "[build_runtime] stamp mismatch: resetting tracked changes and re-patching"
        git checkout -- .
    fi
    python3 "$ROOT/probes/edge0_port/join4_apply.py" "$LLAMA_DIR"
    echo "$WANT" > "$STAMP"
fi

cmake -S "$LLAMA_DIR" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON -DLLAMA_CURL=ON
cmake --build "$BUILD" --config Release -j"$(nproc 2>/dev/null || echo 4)" \
    --target llama-server llama-cli llama-bench

echo "[build_runtime] binaries:"
for b in llama-server llama-cli llama-bench; do
    p="$BUILD/bin/$b"
    [ -x "$p" ] || { echo "[build_runtime] ERROR: missing $p" >&2; exit 1; }
    echo "  $p"
done
