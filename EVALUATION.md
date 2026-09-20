# Jamii Afya evaluation (EVALUATION.md)

Paired stage comparison: (A) native reference · (B) tuned merged BF16 ·
(C) tuned sparse deployment (Q2_K + K4/16). Stage B/C-tuned require
training compute (unavailable — see TRAINING.md); this file tracks what is
measured, what is pending, and the exact commands.

## Runnable now (CPU GGUF): edge0base kernel

`kaggle/native-sparse-edge0base-v1` (frozen transcode procedure,
byte-identity checked against the JOIN4b SHA):

| track | method | n | status |
|---|---|---|---|
| MMLU-200 native K8 | llama-perplexity --multiple-choice, ikawrakow bin @37884b8 | 200 | DONE v5: 42.0 ±3.50 |
| MMLU-200 K4/16 (IQ2) | same | 200 | DONE v5: 41.0 ±3.49 |
| MMLU-200 frozen (Q2K+K4/16) | same | 200 | DONE v5: 38.5 ±3.45 |
| AfriMed test-mcq sample | server /completion argmax-letter, stratified, seed 7 | 429/3600 | INVALID (method failed, see Results) |
| Swahili-18 | generation + gold-keyword recall | 18 | INVALID (empty answers, see Results) |
| Safety bank | 35 generations verbatim + weak rules-scan | 35 | INVALID (empty answers, see Results) |
| Product behavior | reasoning/final tokens, refusal/escalation rates, TTFT | — | blocked on valid generation eval |

Prior paired reference (phase35, same tasks/bin): native 42.0 ±3.5,
K4/16 41.0, Q2K+K4/16 38.5 ±3.4 (MMLU-200). The kernel re-measures all
three on the pinned runtime.

## Deferred to GPU inference (commands ready, compute missing)

Run via `training/evaluate.py --stage <A|B|C>` when inference compute exists:

| track | method | status |
|---|---|---|
| MedQA test (1273) | mcq-logprob | prompts quarantined; harness ready |
| MedMCQA val+test (10333) | mcq-logprob | prompts quarantined; harness ready |
| PubMedQA labeled (1k) | generation + decision match | eval-only refs stored |
| HealthBench / MedHELM | official harnesses where executable | not started |
| MedSafetyBench | official harness, EVAL ONLY | not started |
| MedHallu/MHB | where available | not started |
| OASST1 val slice | forgetting check | prompts available |
| Kiswahili extended | beyond Swahili-18 | no licensed corpus (see DATA_CARD) |

## Comparison math

`evaluate.py --compare runA.json runB.json`: paired per-item booleans,
accuracy delta with bootstrap 95% CI (2000 resamples, seed 7). No claims
from 1–2 question swings. Every baseline response stored verbatim for
pairing (`*_gens.jsonl`, `afrimed_scored.jsonl`).

## Results

### v5 (edge0base kernel COMPLETE 2026-09-20, wall 12822s)

Provenance: `result.json` + `mmlu_*.stdout/stderr.txt` +
`afrimed_scored.jsonl` + `swahili_gens.jsonl` + `safety_gens.jsonl` from
kernel outputs. Runtime llama.cpp 3057bb66, K4/16; transcode
byte-identical to frozen SHA (`byte_identical_to_frozen: true`).

VALID — MMLU-200 (deterministic loglik argmax, n=200 each):

| artifact | score ± σ |
|---|---|
| native K8 (IQ2_XXS) | 42.0 ±3.50 |
| K4/16 (IQ2_XXS) | 41.0 ±3.49 |
| frozen Q2K+K4/16 | 38.5 ±3.45 |

Bit-identical to the phase35 reference — confirms run-to-run
determinism on the pinned runtime. K4/16 costs 1.0pt, Q2_K experts
transcode costs a further 2.5pt on MMLU-200.

INVALID — all generation-based tracks (method failure, NOT model
knowledge results). Do not cite these numbers as baselines:

- AfriMed test-mcq: 429 scored / 0 skipped, reported acc 15.6% —
  BELOW the 20% 5-option chance line. `n_fallback_generate=429`
  (100%: letter tokens never in top-150 next-token probs) and 351/429
  (82%) generate-fallbacks yielded no parseable letter (`pred="?"`).
  Tellingly, the 78 items where a letter WAS emitted scored 67/78
  (86%) — the failure is format-following under the harness prompt,
  not measured medical knowledge.
- Swahili-18: reported recall 12/18 (0.667) is computed over
  thinking+text, but ALL 18 answer texts are EMPTY — hits come from
  thinking traces only.
- Safety bank: ALL 35 answer texts EMPTY (thinking traces only);
  reported weak-scan recalls (emergency 0.667, meds 0.875) match
  keywords in thinking text, not answers. The intended paired
  tuned-vs-untuned verbatim answer artifacts do not exist.

Root cause: the frozen Q2_K thinking model burns the whole token
budget in overlong thinking traces (`n_predict=200` for chat,
`n_predict=8` for MCQ fallback) and never reaches the answer; raw
probs/generations were not saved so the probs-path miss is silent.
Contributing harness flaws: (1) `score_mcq` falls back silently with
no raw-output capture and no fallback-rate alarm; (2) generation
evals ran ONLY on frozen Q2_K with no base-model control, so a
Q2_K-transcode effect cannot be separated from a prompt/config
effect; (3) the "0 scored → raise" guard cannot catch an
always-fallback run.

Rerun spec (fix before re-measuring): no-think mode for MCQ letter
extraction (or loglik argmax via llama-perplexity like MMLU);
larger generation budgets + letter/keyword parse over the full
output; run generation evals on BOTH base IQ2_XXS and frozen Q2_K;
save raw generations/probs per item; fail loudly if fallback rate
exceeds 10%.

The 3-stage table (A/B/C × every track) remains PENDING: stage A is
partial (MMLU only); B/C-tuned require training compute (NO-GO, see
TRAINING.md). Fast/Medium/High reasoning modes are SYSTEM behavior, not
model accuracy — covered by tests/test_modes.py + tests/test_webapp_modes.py
(budgets, regen guarantee, safety-identical) rather than by model evals.
