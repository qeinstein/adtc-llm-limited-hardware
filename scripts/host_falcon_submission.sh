#!/usr/bin/env bash
# Publish the already validated final GGUF and its SHA256 sidecar.
# Requires: huggingface-cli login (or HF_TOKEN) and a public model repository.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-$ROOT/model/Falcon-H1-1.5B-Deep-JamiiAfya-Q4_K_M.gguf}"
REPO_ID="${HF_REPO_ID:-Fluxx08/jamii-afya-falcon-h1-1.5b}"
MODEL_NAME="Falcon-H1-1.5B-Deep-JamiiAfya-Q4_K_M.gguf"

[ -f "$MODEL" ] || { echo "missing GGUF: $MODEL" >&2; exit 1; }
SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"
BYTES="$(stat -c%s "$MODEL")"
SIDE_CAR="$(mktemp)"
PUBLISH_CARD="$(mktemp)"
trap 'rm -f "$SIDE_CAR" "$PUBLISH_CARD"' EXIT
printf '%s  %s\n' "$SHA256" "$MODEL_NAME" > "$SIDE_CAR"

python3 - "$ROOT/MODEL_CARD.md" "$PUBLISH_CARD" "$REPO_ID" "$SHA256" "$BYTES" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
destination = Path(sys.argv[2])
text = path.read_text(encoding='utf-8')
text += f"\n\nPublished artifact record\n\n- Repository: {sys.argv[3]}\n- SHA256: {sys.argv[4]}\n- Bytes: {sys.argv[5]}\n"
destination.write_text(text, encoding='utf-8')
PY

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "huggingface-cli is required for hosting" >&2
    exit 1
fi
huggingface-cli upload "$REPO_ID" "$MODEL" "$MODEL_NAME" --repo-type model
huggingface-cli upload "$REPO_ID" "$SIDE_CAR" "$MODEL_NAME.sha256" --repo-type model
huggingface-cli upload "$REPO_ID" "$PUBLISH_CARD" README.md --repo-type model
echo "hosted $REPO_ID/$MODEL_NAME bytes=$BYTES sha256=$SHA256"
