#!/usr/bin/env bash
# Reproducible Falcon adapter -> merged HF -> F16 GGUF -> deployment GGUF.
# The old scripts/export_gguf.sh is Qwen-specific and is not used here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
ADAPTER="${1:?usage: $0 ADAPTER_DIR|--stock [OUT_DIR]}"
OUT_DIR="${2:-$ROOT/experiments/falcon-production-v1/export}"
BASE_MODEL="tiiuae/Falcon-H1-1.5B-Deep-Instruct"
MODEL_REVISION="b6648636ddc906688974282de6e7a243395f5423"
LLAMA_DIR="${LLAMA_DIR:-$ROOT/llama.cpp}"
# Keep the converter on the same tracked upstream line used by the official
# scalar profiler build.  The former short ref (451b89b) was pruned upstream
# and made export fail after a completed training run.
LLAMA_REVISION="${LLAMA_REVISION:-master}"
QUANT="Q4_K_M"

mkdir -p "$OUT_DIR"
MERGED="$OUT_DIR/merged-hf"
F16="$OUT_DIR/Falcon-H1-1.5B-Deep-JamiiAfya-f16.gguf"
DEPLOY="$OUT_DIR/Falcon-H1-1.5B-Deep-JamiiAfya-${QUANT}.gguf"
PROMOTION_MANIFEST="${PROMOTION_MANIFEST:-}"

if [ "$ADAPTER" = "--stock" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] exporting pinned stock Falcon emergency fallback"
  "$PY" "$ROOT/scripts/merge_falcon_submission.py" --stock --out "$MERGED" --config "$ROOT/configs/falcon-production-v1.json"
else
  ADAPTER="$(cd "$ADAPTER" && pwd)"
  if [ -z "$PROMOTION_MANIFEST" ]; then
    PROMOTION_MANIFEST="$ADAPTER/promotion_manifest.json"
  elif [[ "$PROMOTION_MANIFEST" != /* ]]; then
    PROMOTION_MANIFEST="$(cd "$(dirname "$PROMOTION_MANIFEST")" && pwd)/$(basename "$PROMOTION_MANIFEST")"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] promotion verification adapter=$ADAPTER manifest=$PROMOTION_MANIFEST"
  "$PY" "$ROOT/scripts/verify_falcon_promotion.py" \
    --adapter "$ADAPTER" \
    --manifest "$PROMOTION_MANIFEST" \
    --minimum-pass-rate 100
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] merge start adapter=$ADAPTER base=$BASE_MODEL@$MODEL_REVISION"
  "$PY" "$ROOT/scripts/merge_falcon_submission.py" \
    --adapter "$ADAPTER" \
    --out "$MERGED" \
    --config "$ROOT/configs/falcon-production-v1.json" \
    --adapter-commit "${ADAPTER_COMMIT:-}"
fi

if [ ! -d "$LLAMA_DIR/.git" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cloning llama.cpp@$LLAMA_REVISION"
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" fetch --quiet origin "$LLAMA_REVISION"
git -C "$LLAMA_DIR" checkout --quiet "$LLAMA_REVISION"
if [ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]; then
  cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
  cmake --build "$LLAMA_DIR/build" --config Release -j"${CMAKE_BUILD_PARALLEL_LEVEL:-4}" --target llama-quantize
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] convert start"
"$PY" "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED" --outfile "$F16" --outtype f16
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] quantize start quant=$QUANT"
"$LLAMA_DIR/build/bin/llama-quantize" "$F16" "$DEPLOY" "$QUANT"

"$PY" - "$OUT_DIR" "$DEPLOY" "$LLAMA_REVISION" <<'PY'
import hashlib, json, sys
from pathlib import Path

out = Path(sys.argv[1])
deploy = Path(sys.argv[2])
llama_revision = sys.argv[3]
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
manifest = {
  "deployment_model": str(deploy),
    "deployment_bytes": deploy.stat().st_size,
    "deployment_sha256": sha(deploy),
    "quantization": deploy.stem.rsplit("-", 1)[-1],
    "llama_cpp_revision": llama_revision,
    "merged_hf": str(out / "merged-hf"),
  "f16_gguf": str(out / "Falcon-H1-1.5B-Deep-JamiiAfya-f16.gguf"),
  "base_model": "tiiuae/Falcon-H1-1.5B-Deep-Instruct",
  "base_revision": "b6648636ddc906688974282de6e7a243395f5423",
  "training_config_sha256": json.loads((out / "merged-hf" / "merge_manifest.json").read_text()).get("training_config_sha256"),
  "adapter_commit": json.loads((out / "merged-hf" / "merge_manifest.json").read_text()).get("adapter_commit"),
}
(out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] export complete deployment=$DEPLOY"
