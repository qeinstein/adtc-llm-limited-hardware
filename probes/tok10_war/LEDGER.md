# TOK-18 WAR LEDGER (laptop CPU, single-stream, qual-OK, RAM<=6GB)

OBJECTIVE (updated H+2): >=18 tok/s single-stream decode on i5-10/12th-gen
4c/8GB-DDR4/Ubuntu22.04, peak RSS ABSOLUTELY <7GB (pref <=6GB), ~95%+ quality
(accuracy/safety/Kiswahili preserved in practice). N100 + Kaggle-4c numbers
are PROXY ONLY (Kaggle-4c is conservative for i5: expect ~1.0-1.25x uplift).
RAM is ammunition: spend to 6GB (caches, bigger/faster quants, preconverted
residents). Optimize TOTAL SCORE. Report Pareto: tok/s | RSS | quality | mech.

Baseline (proxy): 4.745 tok/s resident / ~3.0 bounded. Need ~3.8x proxy
(~3x if i5 uplifts 1.25x). Stack required: spec x lowbit x expertq x cache.

## Bets (ranked)
| id | idea | class | status | result |
|---|---|---|---|---|
| SPEC-AGREE | 0.8B same-vocab draft acceptance | 2-3x exact | DONE GO | **alpha=0.755** (2366 toks, teacher-forced greedy, exact ids). Draft 31.9ms/pos. Proj ~12 tok/s @dl3 (verify est). Template omitted (approx, small) |
| SPEC-BUILD | full speculation real tok/s + exactness | 2-3x exact | RUNNING (k v3) | v2 ERROR: --draft-max removed upstream -> --spec-draft-n-max (verified in arg.cpp). v3 re-push + tg-only bench. PARTIAL v2 data: draft Q4_0 tg 21.26 / Q8_0 tg 15.84 / control tg 4.45 (bench) / 3.375 (CLI fixed-prompt). Draft/target ~4.8x; alpha .755 => proj ~1.7x @dl4-5 (MARGINAL zone; needs the real run) |
| LOWBIT | dense Q5->Q3 + attn Q4 + 3 gates | 1.3x approx | RUNNING (k v2) | v1 ERROR: --tensor-type takes concrete ggml_type, q3_k_m rejected -> usage exit (verified in llama-quant.cpp: regex_search, first-match-wins). FIX: dense FFN->q3_k, attn->q4_k (mirrors Q3_K_M), router/norms/embd keep. Strict per-tensor verify. PROMOTE iff ratio>=1.20 + gates |
| EXPERTQ | COMBO experts->Q4_K + dense FFN->Q3 + attn Q4 | 1.5x approx | RUNNING (k v2) | v1 pushed with same q3_k_m bug (would ERROR) -> fixed preemptively + re-pushed v2. Router stays F32 (tiny). Same 3 gates, PROMOTE iff ratio>=1.30. RSS via ru_maxrss |
| OPEN-RISK | 6GB packaging: bounded-cache x speculation merge? | — | NOTED | ratios first; winner stack needs cache-capped runner on laptop. 5.7GB feasible in principle (1 dense + 4 cache + 0.5 draft) |
| LOWBIT | dense trunk Q5->IQ3 requant + gates | 1.3-1.5x approx | DESIGN | — |
| ZCOPY | zero-copy residency validation | 1.15x exact | QUEUED filler | — |
| STACK | speculation x low-bit x zcopy | multiplicative | after parts | — |

## Killed (this sprint)
(none yet)

## Log
- H+0.0: user constraints locked (laptop-only, single-stream, qual-OK).
  User will not run commands: Kaggle is the ONLY execution env (proxy ratios).
  Metric confirmed pure tg (p512/n128) from gate1 record — no loophole.
- H+0.0: found Qwen3.5-0.8B same-248320-vocab + official GGUFs (draft candidate).
  MTP kill does NOT transfer (weak single-head draft vs strong 0.8B).
- H+0.2: SPEC-AGREE v1 pushed (teacher-forced alpha, server loop, ~25min).
- Triage round: LOWBIT v1 + EXPERTQ v1 + SPECBUILD v2 all ERROR.
  (1) lowbit/expertq: `--tensor-type q3_k_m` invalid — CLI wants concrete
  ggml_type (verified parse_tensor_type/llama_tensor_get_type in source;
  patterns are regex_search, first-match-wins). Fixed: FFN->q3_k, attn->q4_k,
  router/norms/embd/output keep; strict per-tensor post-verify.
  (2) specbuild: `--draft-max` removed upstream; `--model-draft` still valid,
  n-max is now `--spec-draft-n-max` (verified in arg.cpp).
  (3) bench() averaged pp+tg rows — fixed to tg-only everywhere (honest Pareto).
  Re-pushed: lowbit k-v2, expertq k-v2, specbuild k-v3. All RUNNING.
- SPEC-BUILD v2 partial (salvaged): draft-Q4_0 tg 21.26, draft-Q8_0 tg 15.84,
  control tg 4.45 bench / 3.38 CLI-fixed. Draft pick: Q4_0 (faster).
  Back-of-envelope with alpha=0.755, cost ratio 4.8x: dl=4 -> ~1.68x,
  dl=5 -> ~1.67x. MARGINAL zone (bar 1.8) — real run decides; i5 uplift may help.
