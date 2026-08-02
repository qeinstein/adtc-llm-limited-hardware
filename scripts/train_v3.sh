#!/usr/bin/env bash
# v3: fix the two things still broken — Kiswahili generation, and basic questions.
# ---------------------------------------------------------------------------
# v2 fixed English rambling but left two gaps, both traceable to the data mix:
#   - Kiswahili still looped: v2's 5,000 instruction rows were ~96% English.
#   - "hi" / "asante" / "what can you do" produced clinical answers or invented
#     doses, because every example the model had ever seen was a clinical question.
#
# v3 mix (all distilled through OpenRouter, grounded in our own vetted guidelines):
#   1,182  Kiswahili clinical (591 x2) — long multi-clause answers + escalation
#          dialogues where turn 2 genuinely changes the recommendation
#     720  general/identity/social/health-literacy, bilingual (360 x2)
#     400  our hand-written clinical set (80 x5)
#   2,500  English medical Q&A
#   1,200  MCQA items as a ranking anchor (~4,600 forward rows)
#
# Trains a FRESH adapter on the recovered fine-tuned weights (not on v2's adapter)
# so this is one coherent run rather than a stack of partial corrections. The
# recovered weights carry the arc_easy 79.5 ranking ability; the MCQA anchor holds
# it in place.
#
# SAFETY: the shipped GGUF is backed up and restored; v3 lands in its own file and
# ships only if the A/B says it should.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/output/v3.log"
SUMMARY="$ROOT/output/V3_RESULT.md"

RECOVERED="$ROOT/output/jamii-hf-recovered"
ADAPTER="$ROOT/output/jamii-lora-v3"
SHIPPED="$ROOT/model/Qwen3-0.6B-Q4_0.gguf"
BACKUP="$ROOT/model/.shipped-backup.gguf"
NEW="$ROOT/model/Qwen3-0.6B-Q4_0-v3.gguf"

exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "v3 started: $(date)"
echo "=================================================================="
step() { echo ""; echo "===== $* ($(date +%H:%M:%S)) ====="; }

[ -f "$SHIPPED" ] && cp -f "$SHIPPED" "$BACKUP" && echo "shipped GGUF backed up"

step "1/4 train"
PYTHONUNBUFFERED=1 "$PY" scripts/train_lora.py \
    --base_model "$RECOVERED" \
    --accuracy_file "$ROOT/output/mcqa_v3.jsonl" \
    --clinical_file "$ROOT/output/sft_v3.json" \
    --healthcare_corpus_file /dev/null \
    --clinical_repeat 1 \
    --batch_size 1 --grad_accum 32 --max_len 320 \
    --epochs 2 --lr 5e-5 --save_steps 60 \
    --output_dir "$ADAPTER"
RC=$?; echo "train exit: $RC"
if [ "$RC" -ne 0 ] || [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
    echo "TRAIN FAILED — shipped model untouched."
    printf '# v3 FAILED\n\nTraining exited %s. Shipped model UNCHANGED.\nSee `output/v3.log`.\n' "$RC" > "$SUMMARY"
    exit 1
fi

step "2/4 export GGUF"
BASE_MODEL="$RECOVERED" QUANT=Q4_0 PYTHON="$PY" bash scripts/export_gguf.sh "$ADAPTER"
ERC=$?
if [ -f "$SHIPPED" ] && [ "$ERC" -eq 0 ]; then mv -f "$SHIPPED" "$NEW"; echo "new -> $NEW"; fi
[ -f "$BACKUP" ] && cp -f "$BACKUP" "$SHIPPED" && echo "shipped GGUF restored"
if [ ! -f "$NEW" ]; then
    echo "EXPORT FAILED — shipped model restored."
    printf '# v3 FAILED at export\n\nAdapter is at `output/jamii-lora-v3`. Shipped model UNCHANGED.\n' > "$SUMMARY"
    exit 1
fi

step "3/4 accuracy A/B"
NA=$("$PY" scripts/mcq_eval.py --model "$NEW"     --task arc_easy --limit 300 2>/dev/null | tail -1)
OA=$("$PY" scripts/mcq_eval.py --model "$SHIPPED" --task arc_easy --limit 300 2>/dev/null | tail -1)
NM=$("$PY" scripts/mcq_eval.py --model "$NEW"     --task medmcqa  --limit 300 2>/dev/null | tail -1)
OM=$("$PY" scripts/mcq_eval.py --model "$SHIPPED" --task medmcqa  --limit 300 2>/dev/null | tail -1)
echo "arc_easy NEW: $NA"; echo "arc_easy OLD: $OA"
echo "medmcqa  NEW: $NM"; echo "medmcqa  OLD: $OM"

step "4/4 generation A/B (clinical + the basics that were broken)"
PYTHONPATH=. "$PY" scripts/stress_test_qualitative.py --model "$NEW" > "$ROOT/output/gen_v3.txt" 2>/dev/null
PYTHONPATH=. "$PY" scripts/basics_check.py --model "$NEW"     > "$ROOT/output/basics_v3.txt" 2>/dev/null
PYTHONPATH=. "$PY" scripts/basics_check.py --model "$SHIPPED" > "$ROOT/output/basics_old.txt" 2>/dev/null
echo "wrote output/gen_v3.txt, output/basics_v3.txt, output/basics_old.txt"

{
  echo "# v3 result — $(date)"
  echo
  echo '| task | OLD (shipped) | NEW (v3) |'
  echo '|---|---|---|'
  echo "| arc_easy (n=300) | \`$OA\` | \`$NA\` |"
  echo "| medmcqa (n=300)  | \`$OM\` | \`$NM\` |"
  echo
  echo "Note: at n=300 the 95% CI is about +/-4.7 points, so treat gaps smaller"
  echo "than that as noise, not improvement."
  echo
  echo "- NEW: \`model/Qwen3-0.6B-Q4_0-v3.gguf\` (NOT shipped)"
  echo "- Shipped: \`model/Qwen3-0.6B-Q4_0.gguf\` (unchanged)"
  echo "- Clinical generation: \`output/gen_v3.txt\`"
  echo "- Basics A/B: \`output/basics_v3.txt\` vs \`output/basics_old.txt\`"
  echo "- Log: \`output/v3.log\`"
} > "$SUMMARY"

echo ""; echo "DONE $(date)"; cat "$SUMMARY"
