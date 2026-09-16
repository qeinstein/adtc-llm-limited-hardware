# Edge0 Phase 2 — v3 adjudication (3.5 vs 3.6) + q2k-quality confirmation

Status: v3 MMLU VALID but INCONCLUSIVE; v3 generation gates VOID (harness bug);
v4 (fixed harness) LAUNCHED. No base-model choice made.

## v3 MMLU-100 (logprob multiple-choice, thinking-independent) — VALID

- A_qwen35: 37.0 ± 4.85 (n=100, 1062s) | B_qwen36: 42.0 ± 4.96 (n=100, 1068s)
- Delta +5.0pp B over A; se_diff=6.94, z=0.72, p≈0.47 → weak lean B, NOT significant.
- Cross-kernel replication: q2k-quality-v1's independent control arm (same model
  sha 2a809de3, same dataset revision 37884b, same 100 tasks) also scored exactly
  37.0. The harness is consistent; n=100 is just underpowered (v4 → n=200).
- Caveat: B has 117/120 IQ2 expert tensors vs A's 120/120
  ("same_expert_recipe_as_A": false) — 3 experts differ in recipe; matched-quant
  assumption slightly violated, noted for Phase 4.

## v3 generation gates (18 Kiswahili + 24 heldout + 2 meta) — VOID

- 87/88 records returned empty `content` with completion_tokens=300 (hit max).
- Root cause: Qwen3 thinking consumed the whole 300-token budget server-side;
  evidence: the single nonempty record (A h14 'DENY') cost 279 tokens for
  4 chars of content. Server healthy (~4.8 t/s); client parsing correct.
- The 0/118 Swahili rate is a HARNESS artifact, not a model result. The
  pre-registered grader (phase2_grade.py) was NOT run on void data.

## v4 fix (same slug, pushed as new version)

- `/no_think` appended to all generation prompts (Qwen3-documented trigger;
  deployment-faithful — production cannot afford think-tokens at 18 tok/s).
- Captures `reasoning_content` + chars alongside content; per-prompt logging.
- Fail-fast: aborts if ≥2 of first 8 outputs are empty (no more blind 2.5h burns).
- MMLU 100 → 200 tasks (halves sigma; +~18 min/arm, affordable since no_think
  generations run ~2× faster).

## q2k-quality-v1 (Phase 1d) — CONFIRMED from fresh pull

- control 37.0±4.85 | Q2_K-experts 39.0±4.90 (Δ+2.0 → KEEP, rule: reject if >2pp
  drop) | Q2_K-all 26.0±4.41 (Δ−11.0 → REJECT_QUALITY_REGRESSION).
- Independently verified from stdout tails (100/37.0, 100/39.0, 100/26.0);
  overrides files present (733 lines each); transcodes 120/432 Q2K tensors.
- In-kernel decider applied the rule correctly; my read agrees. No action needed.

## v4 post-mortem — fail-fast tripped correctly, /no_think ineffective, MMLU-A-200 salvaged

- v4 ran 75 min then ERRORED via the fail-fast guard (working as designed —
  no blind 2.5h burn). Log: all 8 probes ctok=300, out_chars=0,
  rsn_chars ~900-1200. The `/no_think` prompt suffix does NOT disable thinking
  through llama-server chat completions for this model.
- Salvaged from partial outputs (ERROR runs keep /kaggle/working files):
  MMLU-A-200 = 40.5% (81/200, se 3.47); tasks 1-100 reproduce 37.0 EXACTLY —
  third independent replication of the control (q2k-quality, v3 A, v4 A).
- v5: sends "enable_thinking": false (harmless if ignored) + 1200-token budget
  so think+answer fits on the slow path; server already splits
  reasoning_content from content so grading stays clean either way. Fail-fast
  now trips only when BOTH fields are empty; persistent thinking warns and
  continues (~100s/prompt slow path, total ~4.5h, within limits).

## edge0phase2-v1 v2 — UNRECOVERABLE, cannot adjudicate

- The Kaggle CLI ignores the /<version> suffix on `output`, `logs`, and `files`
  (proven: v2 pull byte-identical to v3 — same wall_sec to 16 digits, same log
  md5; `files` size column is garbage for both). Only the latest run is
  retrievable, so v2's true outputs/logs are gone and its "COMPLETE" status
  cannot be trusted (likely the latest status echoed back).
- v2 is moot regardless: it shared v3's 300-token thinking-ate-everything
  generation path (no /no_think), so any generations it produced would be
  empty too. v4 remains necessary. Lesson: pull outputs BEFORE pushing the
  next version.
