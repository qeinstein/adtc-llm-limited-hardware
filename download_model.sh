#!/usr/bin/env bash
# Download your model weight file.
#
# Rules:
#   - Must be idempotent (safe to run multiple times).
#   - Must download without any credentials (public URL only).
#   - The output path must match `_runtime.model_path` in metadata.json.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/Qwen2.5-14B-Instruct-Q4_0.gguf"
MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_0.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    echo "Model file already exists at $MODEL_FILE. Skipping download."
    exit 0
fi

echo "Downloading Qwen 2.5 14B Instruct Q4_0 GGUF..."
# Use curl with follow-location and show-error
curl -L -o "$MODEL_FILE" "$MODEL_URL"

echo "Download completed successfully at $MODEL_FILE."
