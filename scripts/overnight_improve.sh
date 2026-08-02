#!/usr/bin/env bash
# Unattended overnight run: fix GENERATION quality without losing the ranking score.
# ---------------------------------------------------------------------------
# Context. The 8-hour RunPod run produced a model that RANKS superbly
# (arc_easy acc_norm 80.0) but WRITES badly — it restated the system prompt,
# invented citations, looped, and drifted into Chinese. Root cause, measured:
# that run trained on ~93,000 MCQA ranking rows against just 80 chat rows
# (upsampled to 240). We taught it to rank and never taught it to write, and
# because we fine-tuned from Qwen3-0.6B-Base (no instruction post-training of
# its own) those 240 rows were the only thing teaching it to answer or to stop.
#
# This run does a LoRA-SFT pass on top of the recovered fine-tuned weights
# (scripts/gguf_to_hf.py — the pod with the original adapter is gone), mixing:
#      5,000 instruction rows  <- the fix (the original run had 80)
#      1,200 MCQA rows         <- anchor, so the 80.0 ranking doesn't drift
# MCQA items cost ~3.9 forward rows each (one per answer choice) vs 1 for an SFT
# row, so they dominate wall-clock; the anchor is deliberately small because the
# ranking ability already lives in the recovered weights and only needs holding
# in place, not relearning.
# Retraining from Base instead would forfeit that 80.0: on an M4 we can afford
# ~18k item-passes overnight vs the original run's ~186k, and the automated
# accuracy score is 50% of the total. Preserve it, don't gamble it.
#
# SAFETY: the currently-shipped GGUF is copied aside and restored at the end.
# The new model is written to a SEPARATE file so nothing ships until the A/B in
# the morning says it should.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/output/overnight.log"
SUMMARY="$ROOT/output/OVERNIGHT_RESULT.md"

RECOVERED="$ROOT/output/jamii-hf-recovered"
ADAPTER="$ROOT/output/jamii-lora-gen"
SHIPPED="$ROOT/model/Qwen3-0.6B-Q4_0.gguf"
BACKUP="$ROOT/model/.shipped-backup.gguf"
NEW="$ROOT/model/Qwen3-0.6B-Q4_0-v2.gguf"

exec > >(tee -a "$LOG") 2>&1
echo "###################################################################"
echo "# overnight_improve.sh started: $(date)"
echo "###################################################################"

step() { echo ""; echo "===== $* ($(date +%H:%M:%S)) ====="; }

# Protect the known-good model before anything can overwrite it.
[ -f "$SHIPPED" ] && cp -f "$SHIPPED" "$BACKUP" && echo "backed up shipped GGUF"

step "1/5 LoRA-SFT on recovered weights"
# batch_size 1 (effective batch still 32 via grad_accum) and max_len 320: a
# batch_size-2 run drove swap to 12.6GB of 13.3GB on this 16GB machine, and the
# step time degraded from 44s to 90s as it thrashed. Smaller micro-batches keep
# the (rows x len x 152k vocab) logit tensor small, which is what dominates.
# PYTHONUNBUFFERED so loss lines actually reach the log while it runs (stdout is
# block-buffered through tee, so they were invisible until process exit).
PYTHONUNBUFFERED=1 "$PY" scripts/train_lora.py \
    --base_model "$RECOVERED" \
    --accuracy_file "$ROOT/output/mcqa_night.jsonl" \
    --clinical_file "$ROOT/output/sft_night.json" \
    --healthcare_corpus_file /dev/null \
    --clinical_repeat 1 \
    --batch_size 1 --grad_accum 32 --max_len 320 \
    --epochs 1 --lr 5e-5 --save_steps 40 \
    --output_dir "$ADAPTER"
TRAIN_RC=$?
echo "train exit: $TRAIN_RC"
if [ "$TRAIN_RC" -ne 0 ] || [ ! -d "$ADAPTER" ]; then
    echo "TRAINING FAILED — shipped model untouched. Stopping."
    { echo "# Overnight run FAILED"; echo; echo "Training did not complete (exit $TRAIN_RC).";
      echo "The shipped model at \`model/Qwen3-0.6B-Q4_0.gguf\` is UNCHANGED and still valid.";
      echo "See \`output/overnight.log\`."; } > "$SUMMARY"
    exit 1
fi

step "2/5 Merge + export to GGUF Q4_0"
BASE_MODEL="$RECOVERED" QUANT=Q4_0 PYTHON="$PY" bash scripts/export_gguf.sh "$ADAPTER"
EXPORT_RC=$?
echo "export exit: $EXPORT_RC"

# export_gguf.sh writes to the shipped filename; move the new artifact aside and
# put the original back so the live path is never left pointing at an unvetted file.
if [ -f "$SHIPPED" ] && [ "$EXPORT_RC" -eq 0 ]; then
    mv -f "$SHIPPED" "$NEW"
    echo "new model -> $NEW"
fi
[ -f "$BACKUP" ] && cp -f "$BACKUP" "$SHIPPED" && echo "restored shipped GGUF"

if [ ! -f "$NEW" ]; then
    echo "EXPORT FAILED — shipped model restored. Stopping."
    { echo "# Overnight run FAILED at export"; echo;
      echo "Training finished (adapter at \`output/jamii-lora-gen\`) but GGUF export failed.";
      echo "Shipped model is UNCHANGED. See \`output/overnight.log\`."; } > "$SUMMARY"
    exit 1
fi

step "3/5 Accuracy A/B (must hold ~80.0 on arc_easy)"
NEW_ACC=$("$PY" scripts/mcq_eval.py --model "$NEW" --task arc_easy --limit 200 2>/dev/null | tail -1)
echo "NEW : $NEW_ACC"
OLD_ACC=$("$PY" scripts/mcq_eval.py --model "$SHIPPED" --task arc_easy --limit 200 2>/dev/null | tail -1)
echo "OLD : $OLD_ACC"

step "4/5 Accuracy A/B on medmcqa (the weaker task)"
NEW_MED=$("$PY" scripts/mcq_eval.py --model "$NEW" --task medmcqa --limit 200 2>/dev/null | tail -1)
echo "NEW : $NEW_MED"
OLD_MED=$("$PY" scripts/mcq_eval.py --model "$SHIPPED" --task medmcqa --limit 200 2>/dev/null | tail -1)
echo "OLD : $OLD_MED"

step "5/5 Generation A/B"
GEN_NEW="$ROOT/output/gen_new.txt"
GEN_OLD="$ROOT/output/gen_old.txt"
PYTHONPATH=. "$PY" scripts/stress_test_qualitative.py --model "$NEW"     > "$GEN_NEW" 2>/dev/null
PYTHONPATH=. "$PY" scripts/stress_test_qualitative.py --model "$SHIPPED" > "$GEN_OLD" 2>/dev/null
echo "wrote $GEN_NEW and $GEN_OLD"

{
  echo "# Overnight run result — $(date)"
  echo
  echo "## What this run did"
  echo
  echo "LoRA-SFT pass on the recovered fine-tuned weights, mixing 5,000 instruction"
  echo "rows (the fix — the original run had only 80) with 1,200 MCQA rows as an anchor"
  echo "to stop the 80.0 ranking score from drifting."
  echo
  echo "## Accuracy A/B (200 items each, held-out test splits)"
  echo
  echo '| task | OLD (shipped) | NEW (this run) |'
  echo '|---|---|---|'
  echo "| arc_easy | \`$OLD_ACC\` | \`$NEW_ACC\` |"
  echo "| medmcqa  | \`$OLD_MED\` | \`$NEW_MED\` |"
  echo
  echo "## Files"
  echo
  echo "- NEW model: \`model/Qwen3-0.6B-Q4_0-v2.gguf\` (NOT shipped yet)"
  echo "- Shipped model: \`model/Qwen3-0.6B-Q4_0.gguf\` (unchanged, still live)"
  echo "- Adapter: \`output/jamii-lora-gen\`"
  echo "- Generation A/B: \`output/gen_new.txt\` vs \`output/gen_old.txt\`"
  echo "- Full log: \`output/overnight.log\`"
  echo
  echo "## Decision rule"
  echo
  echo "Ship the new model ONLY if arc_easy holds (within ~2 points of the old"
  echo "number — that is sample noise at n=200) AND generation is visibly better."
  echo "If arc_easy dropped meaningfully, keep the shipped model: the automated"
  echo "accuracy score is 50% of the total and outweighs prose quality."
} > "$SUMMARY"

echo ""
echo "###################################################################"
echo "# DONE $(date) — see output/OVERNIGHT_RESULT.md"
echo "###################################################################"
cat "$SUMMARY"
