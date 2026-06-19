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

# Expected minimum size for the real Qwen 2.5 14B Q4_0 GGUF file (~8.54 GB)
MIN_SIZE=8500000000

# Get current file size if it exists
FILE_SIZE=0
if [ -f "$MODEL_FILE" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        FILE_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || echo 0)
    else
        FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || echo 0)
    fi
fi

if [ "$FILE_SIZE" -ge "$MIN_SIZE" ]; then
    echo "Model file already exists and appears complete at $MODEL_FILE. Skipping download."
    exit 0
fi

echo "Downloading Qwen 2.5 14B Instruct Q4_0 GGUF (Support Resuming)..."
# Use curl with follow-location, show-error, and resume flag (-C -)
curl -L -C - -o "$MODEL_FILE" "$MODEL_URL"

echo "Download completed successfully at $MODEL_FILE."
