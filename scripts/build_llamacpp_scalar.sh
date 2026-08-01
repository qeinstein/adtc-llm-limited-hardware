#!/usr/bin/env bash
# Build a SCALAR (no-SIMD) llama.cpp that matches the adtc-profiler audit image.
# ---------------------------------------------------------------------------
# The audit VM builds llama.cpp with ALL SIMD disabled (see the profiler Dockerfile:
# GGML_NATIVE/AVX/AVX2/AVX512/FMA/F16C/BLAS = OFF). Decode throughput there is a
# fraction of a normal AVX2 laptop build. Benchmarking against THIS build is how we
# make our self-reported Gate-1 numbers match the Gate-2 audit (±25% throughput /
# ±15% memory tolerance) instead of failing the comparator.
#
# Produces:  llama.cpp/build-scalar/bin/llama-bench   (use with src/benchmark.py --profiler-parity)
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LLAMA_DIR="$ROOT/llama.cpp"
REF="${LLAMACPP_REF:-master}"   # profiler pins ARG LLAMACPP_REF=master

if [ ! -d "$LLAMA_DIR" ]; then
    echo "[scalar-build] Cloning llama.cpp ($REF)..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" fetch --depth 1 origin "$REF" && git -C "$LLAMA_DIR" checkout FETCH_HEAD || true

echo "[scalar-build] Configuring no-SIMD build (audit parity)..."
cmake -B "$LLAMA_DIR/build-scalar" -S "$LLAMA_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
    -DGGML_FMA=OFF -DGGML_F16C=OFF \
    -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF

# Detect core count portably (Linux: nproc, macOS: sysctl), then CAP it.
# llama.cpp is now a large multi-file codebase; an unbounded `-j` (no number) tells
# the compiler to launch unlimited parallel jobs, which reliably OOM-kills the build
# on small/shared CI runners (this is exactly what happened the first time this
# workflow ran: `gmake: *** Terminated`, exit 143 = killed for memory pressure).
if command -v nproc >/dev/null 2>&1; then
    NPROC=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=4
fi
JOBS=$(( NPROC > 4 ? 4 : NPROC ))
[ "$JOBS" -lt 1 ] && JOBS=1
echo "[scalar-build] Building with -j$JOBS (capped, to avoid OOM on small runners)..."

cmake --build "$LLAMA_DIR/build-scalar" --config Release -j"$JOBS" --target llama-bench llama-cli

echo ""
echo "[scalar-build] Done."
echo "  Benchmark for audit parity:"
echo "    PYTHONPATH=. python -m src.benchmark --profiler-parity \\"
echo "      --llama-bench $LLAMA_DIR/build-scalar/bin/llama-bench"
echo ""
echo "  (For the fast INTERACTIVE app on your own machine, build a native version"
echo "   instead: cmake -B build -S \"$LLAMA_DIR\" -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release)"
