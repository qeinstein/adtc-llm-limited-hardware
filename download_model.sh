#!/usr/bin/env bash
# ADTC 2026 — Jamii Afya model downloader (Qwen3.6-35B-A3B Q2K-experts GGUF).
#
# The model URL is a STATIC literal per the Gate-2 requirements: a reviewer
# opens this script and sees exactly what will be downloaded. Downloads are
# never accepted on size alone — the hosted .sha256 sidecar (or MODEL_SHA256)
# must verify.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"
MODEL_URL="https://huggingface.co/Fluxx08/jamii-afya-qwen36-35b-q2k/resolve/e938cd2af04dd5f30731922bc4780ef2f264f032/Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"
MODEL_SHA256="${MODEL_SHA256:-}"
MIN_SIZE=12000000000

file_size() {
    if [ ! -f "$1" ]; then echo 0; return; fi
    if [[ "${OSTYPE:-}" == "darwin"* ]]; then stat -f%z "$1" 2>/dev/null || echo 0
    else stat -c%s "$1" 2>/dev/null || echo 0
    fi
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    else shasum -a 256 "$1" | awk '{print $1}'
    fi
}

fetch_expected_hash() {
    if [ -n "$MODEL_SHA256" ]; then return 0; fi
    local sidecar="${MODEL_URL}.sha256" sidecar_file="$MODEL_FILE.sha256"
    if command -v curl >/dev/null 2>&1 && curl -L --fail --silent --show-error --retry 2 -o "$sidecar_file" "$sidecar"; then
        MODEL_SHA256="$(grep -Eo '[[:xdigit:]]{64}' "$sidecar_file" | head -n 1 | tr '[:upper:]' '[:lower:]')"
    elif command -v wget >/dev/null 2>&1 && wget -q -O "$sidecar_file" "$sidecar"; then
        MODEL_SHA256="$(grep -Eo '[[:xdigit:]]{64}' "$sidecar_file" | head -n 1 | tr '[:upper:]' '[:lower:]')"
    fi
    rm -f "$sidecar_file"
    [ "${#MODEL_SHA256}" -eq 64 ]
}

verify_model() {
    local actual
    [ "$(file_size "$MODEL_FILE")" -ge "$MIN_SIZE" ] || { echo "[download_model] ERROR: artifact is too small" >&2; return 1; }
    [ "${#MODEL_SHA256}" -eq 64 ] || { echo "[download_model] ERROR: MODEL_SHA256 or hosted .sha256 sidecar is required" >&2; return 1; }
    actual="$(sha256_file "$MODEL_FILE")"
    # Portable lowercase (no ${VAR,,}: macOS ships bash 3.2, which lacks it).
    expected_hash="$(printf '%s' "$MODEL_SHA256" | tr '[:upper:]' '[:lower:]')"
    [ "$actual" = "$expected_hash" ] || { echo "[download_model] ERROR: SHA256 mismatch expected=$MODEL_SHA256 actual=$actual" >&2; return 1; }
    echo "[download_model] verified $MODEL_FILE ($(file_size "$MODEL_FILE") bytes) sha256=$actual"
}

mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_FILE" ]; then
    if fetch_expected_hash; then verify_model; exit 0; fi
    echo "[download_model] Existing artifact cannot be verified; re-download or set MODEL_SHA256." >&2
fi

fetch_expected_hash || {
    echo "[download_model] ERROR: host did not provide a SHA256 sidecar; set MODEL_SHA256 explicitly." >&2
    exit 1
}
tmp_file="$MODEL_FILE.partial"
echo "[download_model] Downloading $MODEL_URL"
if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 5 -C - -o "$tmp_file" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$tmp_file" "$MODEL_URL"
else
    echo "[download_model] ERROR: neither curl nor wget is available" >&2
    exit 1
fi
mv -f "$tmp_file" "$MODEL_FILE"
verify_model
