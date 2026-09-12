#!/usr/bin/env bash
# Reproducible Falcon adapter -> merged HF -> F16 GGUF -> deployment GGUF.
# The old scripts/export_gguf.sh is Qwen-specific and is not used here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
ADAPTER="${1:?usage: $0 ADAPTER_DIR [OUT_DIR]}"
OUT_DIR="${2:-$ROOT/experiments/falcon-production-v1/export}"
BASE_MODEL="${BASE_MODEL:-tiiuae/Falcon-H1-1.5B-Deep-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-b6648636ddc906688974282de6e7a243395f5423}"
LLAMA_DIR="${LLAMA_DIR:-$ROOT/llama.cpp}"
LLAMA_REVISION="${LLAMA_REVISION:-451b89b}"
QUANT="${QUANT:-Q4_K_M}"

ADAPTER="$(cd "$ADAPTER" && pwd)"
mkdir -p "$OUT_DIR"
MERGED="$OUT_DIR/merged-hf"
F16="$OUT_DIR/Falcon-H1-1.5B-Deep-Instruct-f16.gguf"
DEPLOY="$OUT_DIR/Falcon-H1-1.5B-Deep-Instruct-${QUANT}.gguf"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] merge start adapter=$ADAPTER base=$BASE_MODEL@$MODEL_REVISION"
"$PY" - "$BASE_MODEL" "$MODEL_REVISION" "$ADAPTER" "$MERGED" <<'PY'
import hashlib, json, sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base, revision, adapter, out = sys.argv[1:]
adapter_path = Path(adapter)
if not (adapter_path / "adapter_config.json").exists():
    raise SystemExit(f"adapter_config.json missing: {adapter_path}")
model = AutoModelForCausalLM.from_pretrained(
    base, revision=revision, trust_remote_code=True, torch_dtype=torch.float16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(base, revision=revision, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
model = model.merge_and_unload()
Path(out).mkdir(parents=True, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
tokenizer.save_pretrained(out)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

manifest = {
    "base_model": base,
    "base_revision": revision,
    "adapter": str(adapter_path),
    "adapter_files": {str(p.relative_to(adapter_path)): sha(p) for p in adapter_path.rglob("*") if p.is_file()},
    "merged_files": {str(p.relative_to(out)): sha(p) for p in Path(out).rglob("*") if p.is_file()},
}
Path(out, "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"merged": out, "files": len(manifest["merged_files"])}, sort_keys=True))
PY

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
    "f16_gguf": str(out / "Falcon-H1-1.5B-Deep-Instruct-f16.gguf"),
}
(out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] export complete deployment=$DEPLOY"
