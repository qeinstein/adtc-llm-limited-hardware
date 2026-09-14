# Phase 7C selective-Q4 quality v2

The corrected reasoning-disabled generation gate completed on the pinned
Qwen3.5-35B-A3B control and selective attention/GDN Q4_K challenger.

The candidate passed 8/8 clinical probes and 7/7 safety checks; the control
passed 7/8 and 6/7 respectively. This is evidence against an obvious clinical
safety regression, not a broad capability claim.

The reported MCQ counts (3/6 control, 2/6 candidate) are invalid as a model
comparison. Every 48-token response stopped before the requested `FINAL:`
answer. The harness then fell back to the last standalone A-D letter appearing
in truncated explanatory prose, so it sometimes graded a distractor mentioned
by the model instead of the answer. That signal is excluded.

Decision: `KEEP_PENDING_LIKELIHOOD_GATE`. Phase 7C quality v3 replaces the
invalid MCQ portion with llama.cpp's length-normalized multiple-choice
likelihood evaluator on the same 100-item MMLU sample for both models.

The full structured output, including every prompt and response, is preserved
in `raw/result.json`. The launch harness remains under
`kaggle/native-sparse-selective-q4-quality-v2/`.
