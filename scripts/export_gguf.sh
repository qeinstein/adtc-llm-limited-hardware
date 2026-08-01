#!/usr/bin/env bash
# Merge LoRA -> convert to GGUF -> DOMAIN imatrix -> quantize.
# ---------------------------------------------------------------------------
# Produces model/Qwen3-0.6B-<QUANT>.gguf (the path in metadata.json _runtime).
# The imatrix is calibrated on our EN+SW medical corpus (output/calibration_corpus.txt
# from scripts/prepare_dataset.py), so precision is biased toward our use case.
#
# QUANT defaults to Q4_K_M but should be set from .github/workflows/quant-sweep.yml's
# result (it may pick Q5_K_M/Q8_0 instead — 0.6B is unusually quant-sensitive, see
# PROGRESS.md): QUANT=Q8_0 bash scripts/export_gguf.sh
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B-Base}"
QUANT="${QUANT:-Q4_K_M}"
LORA_DIR="${1:-$ROOT/output/jamii-lora}"
MERGED_DIR="$ROOT/output/jamii-merged"
GGUF_F16="$ROOT/output/jamii-0.6b-f16.gguf"
CALIB="$ROOT/output/calibration_corpus.txt"
IMATRIX="$ROOT/output/imatrix.dat"
OUT_GGUF="$ROOT/model/Qwen3-0.6B-${QUANT}.gguf"
LLAMA_DIR="$ROOT/llama.cpp"

echo "== [1/4] Merge LoRA adapter into base =="
python - "$BASE_MODEL" "$LORA_DIR" "$MERGED_DIR" <<'PY'
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base, lora, out = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora)
model = model.merge_and_unload()
model.save_pretrained(out); tok.save_pretrained(out)
print("merged ->", out)
PY

echo "== [2/4] Ensure llama.cpp present (for convert + quantize + imatrix) =="
if [ ! -d "$LLAMA_DIR" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
if [ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]; then
    cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
    # Capped parallelism: a bare `-j` (unlimited) OOM-killed our CI build once
    # already (llama.cpp's ~100+ source files can spawn more jobs than a small
    # machine's RAM can hold). RunPod GPU boxes usually have plenty of RAM, but
    # cap anyway for safety/portability.
    JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    [ "$JOBS" -gt 8 ] && JOBS=8
    cmake --build "$LLAMA_DIR/build" --config Release -j"$JOBS" --target llama-quantize llama-imatrix
fi
pip install -q -r "$LLAMA_DIR/requirements.txt" || true

echo "== [3/4] Convert merged HF model -> GGUF f16 =="
python "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" --outfile "$GGUF_F16" --outtype f16

echo "== [4/4] Domain imatrix + quantize to $QUANT =="
mkdir -p "$ROOT/model"
if [ -f "$CALIB" ]; then
    "$LLAMA_DIR/build/bin/llama-imatrix" -m "$GGUF_F16" -f "$CALIB" -o "$IMATRIX" --chunks 100
    "$LLAMA_DIR/build/bin/llama-quantize" --imatrix "$IMATRIX" "$GGUF_F16" "$OUT_GGUF" "$QUANT"
else
    echo "WARN: $CALIB missing (run scripts/prepare_dataset.py). Quantizing without imatrix."
    "$LLAMA_DIR/build/bin/llama-quantize" "$GGUF_F16" "$OUT_GGUF" "$QUANT"
fi

echo ""
echo "Done -> $OUT_GGUF"
echo "Verify: PYTHONPATH=. python -m src.benchmark --profiler-parity --llama-bench $LLAMA_DIR/build-scalar/bin/llama-bench"
