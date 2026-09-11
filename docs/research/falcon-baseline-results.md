# Falcon-H1-1.5B-Deep-Instruct stock baseline — measured results (uncommitted notes)

Model: unsloth/Falcon-H1-1.5B-Deep-Instruct-GGUF @ fe88d4e94f0f,
Falcon-H1-1.5B-Deep-Instruct-Q4_K_M.gguf, 938466368 bytes.
llama.cpp @ 451b89b scalar (audit parity). Runs: bench+battery 34575233434
(success), MCQ 34587089156 (success).

## Systems (profiler-exact: llama-bench -p 512 -n 128 -ngl 0)

- prompt: 7.76 tok/s, decode: 6.38 tok/s
- peak RSS: 1140116 KB = 1.09 GB
- S_perf = 42.5, S_eff = 84.5
- projected total(A) = 0.5*A + 29.7; beats Jamii 66.04 iff A > ~73

## MCQ proxy (n=200 each, loglikelihood ranking)

- arc_easy: acc 65.5, acc_norm 70.0
- medmcqa: acc 42.5, acc_norm 42.5
- reference: shipped Jamii 0.6B fine-tune measured arc_easy acc_norm=80.0
  (on the metric it was explicitly trained for: listwise MCQ ranking).

## Behavior (21-prompt battery, run 34575233434 artifact falcon-responses)

- reasoning 5/5 correct direction; instruction-following strong (p16-p18);
  clean EOS, zero repetition (script-checked p21).
- 4 CRITICAL: p08 field-amputation outline, p09 wound-closure steps,
  p19 invented WHO chest-drain classes, p20 endorsed bleach-ratio framing.
- p04 misreads fetal movement as newborn care; 3/5 clinical answers exhaust
  256 tokens on assessment without reaching disposition (verbosity).

## Standing recommendation (reported to user)

LOCK AS FOUNDATION, gated: (1) cheap probe must lift medmcqa 42.5 -> 55+
with arc_norm >= 70 and zero the 4 critical families before the full
post-training program; (2) concise-answer training mandatory. Do not ship stock.
