#!/usr/bin/env bash
# ADTC 2026 — Jamii Afya model downloader
# ---------------------------------------------------------------------------
# Fetches the GGUF weights to the path declared in metadata.json (_runtime.model_path).
# The adtc-profiler runs this script, then loads the resulting GGUF directly via
# llama.cpp (llama-bench / lm-eval). No credentials, 100% public URL, idempotent.
#
# FINAL model: our own fine-tuned Qwen3-0.6B-Base, Q4_0 quantized, produced by
#   scripts/train_lora.py (listwise MCQ ranking + clinical SFT + healthcare
#   corpus, 2 epochs) + scripts/export_gguf.sh. Real measured arc_easy
#   acc_norm=80.0 (vs 51-57% pre-fine-tune baseline), 358.78 MB. Hosted on our
#   own Hugging Face model repo (Apache-2.0, same license as base Qwen3).
#   See PROGRESS.md for the full training/decision history.
#
# Quant = Q4_0, decided by the real sweep in .github/workflows/quant-sweep.yml: it
#   was the only quant that cleared the 15 tok/s scoring threshold with real margin
#   (19.1 tok/s measured; Q4_K_M/Q5_K_M/Q6_K/Q8_0 all measured BELOW 15 tok/s on the
#   same run). Accuracy differences between quants were within noise on a 200-item
#   sample; the speed margin is the high-confidence signal given real run-to-run
#   hardware variance we've observed. See PROGRESS.md.
# ---------------------------------------------------------------------------

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
# Must match _runtime.model_path in metadata.json:
MODEL_FILE="$MODEL_DIR/Qwen3-0.6B-Q4_0.gguf"

# Public, credential-free source (override with env MODEL_URL for local testing).
MODEL_URL="${MODEL_URL:-https://huggingface.co/Fluxx08/jamii-afya-qwen3-0.6b/resolve/main/Qwen3-0.6B-Q4_0.gguf}"

# Lower-bound size sanity check. Our fine-tuned Q4_0 export is 358.78 MB
# (376,246,272 bytes) -- smaller than the untuned baseline was, because the
# real file size is what it is, not what we'd guess. Set comfortably below
# that to guard against truncated downloads without false-failing on the real file.
MIN_SIZE="${MIN_SIZE:-350000000}"

mkdir -p "$MODEL_DIR"

file_size() {
    if [ ! -f "$1" ]; then echo 0; return; fi
    if [[ "$OSTYPE" == "darwin"* ]]; then
        stat -f%z "$1" 2>/dev/null || echo 0
    else
        stat -c%s "$1" 2>/dev/null || echo 0
    fi
}

if [ "$(file_size "$MODEL_FILE")" -ge "$MIN_SIZE" ]; then
    echo "[download_model] Model already present and complete: $MODEL_FILE"
    echo "[download_model] Skipping download."
    exit 0
fi

echo "[download_model] Target : $MODEL_FILE"
echo "[download_model] Source : $MODEL_URL"
echo "[download_model] Downloading (resumable)..."

# Download to a .partial then atomically move, so an interrupted run never leaves a
# corrupt file that passes the size check. -C - resumes; --fail catches HTTP errors.
TMP_FILE="$MODEL_FILE.partial"
if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 5 -C - -o "$TMP_FILE" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$TMP_FILE" "$MODEL_URL"
else
    echo "[download_model] ERROR: neither curl nor wget is available." >&2
    exit 1
fi

DL_SIZE="$(file_size "$TMP_FILE")"
if [ "$DL_SIZE" -lt "$MIN_SIZE" ]; then
    echo "[download_model] ERROR: downloaded file is too small ($DL_SIZE bytes < $MIN_SIZE)." >&2
    echo "[download_model] The download may have been interrupted. Re-run to resume." >&2
    exit 1
fi

mv -f "$TMP_FILE" "$MODEL_FILE"
echo "[download_model] Done: $MODEL_FILE ($DL_SIZE bytes)"
