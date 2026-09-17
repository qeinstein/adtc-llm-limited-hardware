# Dynamic Skipping + Future-Route Prefetch — Cheap Falsification Sprint

**Date:** 2026-09-16 · **Model:** Qwen3.5-35B-A3B (hidden 2048, 40 layers, 256 experts, top-8 + shared)
**Anchor:** L20 REAL contextual states (2933 rows = 917 prefill + 2016 decode, 32 prompts, route cross-check 2015/2016).
L00/L39 are PROXY only (embed directions, N=320, routing logcorr 0.17 vs real corpus) — suggestive, not decisive.

## 0. Measurement contract (read before the numbers)

- **Error definition.** True block-relative error = `||MoE|| / ||h1+MoE||`, `h1` = post-mixer residual.
  `h1` was NOT dumped, so we report `||MoE|| / ||X||`, `X` = dumped MoE input (post-attention-norm).
  True error = reported ratio / RMS(h1). RMS(h1) is unmeasured; if ≈1 (typical mid-layer), reported ≈ true;
  if RMS(h1)≈2, true is HALF reported (verdicts would soften one bar — see §5).
- **Shared expert.** Contextual routed output R8 is exact (cached IQ2-dequant f32 experts).
  Contextual shared output S uses bf16 weights (GGUF uses Q5_K/Q6_K — small quant delta, stated).
  `B = R8+S`; measured `cos(R8,S)≈0.00` (orthogonal), `|S|/|R8|≈0.62` contextual (0.87 proxy).
- **Cosine** = `cos(X, X+MoE)` proxy (assumes h1 direction ≈ X direction; exact values, assumption-labeled).
- **Logits error: NOT measured.** Needs a full-model run (no 10 GB weights locally; disk 0.3 GB free).
  Cost on Kaggle ≈ one 16–30 min kernel (same infra as the L20 hdump). This is the explicitly missing gate.
- **Mixer-output norm: NOT measured** (needs h0/h1/mixer dump). Whole-block verdict at 1% follows by
  reduction (§3); at 2%/5% it is bounded, not closed.
- **Thresholds** are in-sample maxima unless marked `held` (train 22 prompts → held 10 prompts).

## 1. Skip MoE branch only (oracle, per-token error)

L20 contextual, decode-only (n=2016 — the tokens that set tok/s):

| skip target | \|MoE\| med | ratio med | <1% oracle | <2% oracle | <5% oracle | cos med |
|---|---|---|---|---|---|---|
| routed only (keep shared) | 0.615 | 1.375% | 6.70% | 89.7% | 100% | 0.99989 |
| full MoE (R8+S) | 0.744 | 1.679% | **0.00%** | 76.4% | 100% | 0.99984 |

(`cos` = mean 0.99984/0.99989, min 0.99846 — uninformative at these ratios, as expected.)
Decode `ratio_b` minimum is 1.0145%: zero decode tokens sneak under 1%.
The 74 all-token `<1%` hits (2.5%) are ALL prefill template-prefix tokens (pos 0–10, 32 identical BOS rows,
top8mass 0.69 vs 0.15 baseline) — useless for decode tok/s.

Layer variation (⚠️ proxy for L00/L39):

| layer | source | R8/X med | R8 <1% | R8 <2% | B_est <1% | B_est <2% |
|---|---|---|---|---|---|---|
| L00 | proxy | 0.715% | 79% | 97% | 55–67% | 93–95% |
| L20 | **context** | 1.375% | 6.7% | 90% | **0%** | 76% |
| L39 | proxy | 3.399% | 0% | 0.6% | 0% | 0–0.3% |

L00-proxy says early layers may skip at strict 1%; L39-proxy says late layers cannot even at 2%.
Proxy directions are unreliable at depth (L39 `|R8|`=4.67 smells out-of-distribution) — needs contextual confirmation.

## 2. Skip MoE only when a cheap signal predicts low impact (decode-only, STRICT max-error)

Best cheap signal per bar (oracle threshold; all 6 signals × both directions searched):

| skip target | bar | oracle | best signal strict | rate | transfers? |
|---|---|---|---|---|---|
| routed | 1% | 6.7% | top1/low (8 toks) | 0.40% | — |
| routed | 2% | 89.7% | **top1 ≤ 0.020** | **21.1%** | **YES: held 22.7%, 0 viol, max 1.72%** |
| routed | 5% | 100% | any | 100% | — |
| full | 1% | 0% | — | **0%** | — |
| full | 2% | 76.4% | ent8/high (19 toks) | 0.94% | no (noise) |
| full | 5% | 100% | any | 100% | — |

Signal notes: router is extremely diffuse (entropy 5.19/5.55, top8mass med 0.13, top1 med 0.031).
Correlations with error are weak (`|r|` 0.06–0.25); AUROC at 2% is 0.57–0.67 (near chance) —
except the diffuse tail (top1<2%) which is reliably small for ROUTED-only.
`cos(X,B)` needs MoE (not cheap); mixer-output norm unmeasured; hidden norm useless (`r`=-0.06).
Mean-error operating points are 100% at 2% (mean 1.75%) but admit 5.8% max — wrong safety metric, ignored.

## 3. Skip whole block (y = x)

Unmeasured directly (no mixer dump). Closed at 1% by reduction: whole-block error `||m+B||` can beat
1% when `||B||` alone exceeds 1% only via measure-zero cancellation (`m≈−B`, independently computed
branches). `||B||` exceeds 1% on 100% of decode tokens ⇒ whole-block oracle at 1% is ≈0%.
At 2%: MoE alone consumes 1.68% of the 2% budget (median), leaving 0.3% for the mixer, which does
22.9% of wall work with 26–46% per-head ablation error (agent2) — implausible; formal bound needs one
mixer-norm dump (same kernel, §6). Verdict: KILL at 1%; likely-KILL at 2% pending that dump.

## 4. ms/token and tok/s (measured phase5a/6b/9b frame)

Resident base 210.7 ms (4.745 tok/s); routed 87.0 ms, shared 8.0 ms. Bounded base 312.7 ms (3.20 tok/s).

| policy | save (resident) | → tok/s | save (bounded) | → tok/s | staging saved |
|---|---|---|---|---|---|
| 21% routed @2% (the ONE keep) | 18 ms | 5.17 (+9%) | 37 ms | 3.63 (+13%) | 13–25 MB/tok |
| 20% full-MoE hypothetical | 19 ms | 5.22 (+10%) | 35 ms | 3.60 (+13%) | same |
| 76% full @2% oracle (unreachable) | 72 ms | 7.22 (+52%) | — | — | — |
| 3% signals @1–2% (everything else) | ≤3 ms | ≤4.81 (+1%) | ≈0 | — | ≈0 |

No policy reaches 15 tok/s alone; the 21% policy is a +9–13% contributor, not a solution.

## 5. KEY QUESTION — verdicts

- **Strict 1%, full MoE: KILL HARD.** Decode oracle 0% — impossible at any signal quality (RMS(h1)≈1).
  If RMS(h1)≈2 the true oracle rises to ≈2.5% (still <10% ⇒ still KILL).
- **Strict 1%, routed-only: KILL.** Oracle 6.7%, signals 0.4%.
- **2%, routed-only (keep shared): MARGINAL KEEP.** 21% strict-safe via `top1≤0.02`, transfers to held
  prompts (22.7%, zero violations). One layer, one signal, 2% unvalidated end-to-end — do NOT build a
  controller on this yet.
- **2%, full MoE: KILL.** Signals 0.9% vs 76% oracle (25× gap).
- **5%: rates KEEP (100%) but error bar too loose** — 5%/layer × 8 skipped layers ≈ 14% RMS drift (40% worst-case aligned);
  needs logits gate; not "strict".
- **Whole-block ≥20%: KILL** (0% at 1% by reduction; 2% implausible, dump to close).
- **Future-route prefetch (ID-copy): KILL HARD.** L→L+1 exact-top8 recall 3.19% ≈ random 3.125%,
  median 0, 0% exact-set-match, 77% zero-overlap; L→L+2 identical (3.33%). Static popularity 23% and
  adjacent-token same-layer retention 32% both beat cross-layer copying 7–10× — stage from temporal
  locality + cache, not from current-layer IDs. Learned X_L→route_{L+1} is untested (needs paired
  two-layer states) but exact-8/8 under entropy 5.2 looks hopeless; and one layer-time (5.3 ms) barely
  covers staging 1.6–3.3 MB (2–17 ms across 200–1000 MB/s) — prediction would arrive too late anyway.

## 6. Next ONE action

Run ONE Kaggle kernel that (a) dumps L00+L39 MoE inputs in a single 32-prompt run (same hdump hook,
3 layer-name matches — closes the depth-generality gap: is L00 really 79%-skippable at strict 1%?),
AND (b) replays decode with the frozen `top1≤0.02` routed-skip policy at L20 while capturing per-token
top-1 agreement + KL vs the exact control (closes the logits gate for the one 2% KEEP).
**Kill the branch if:** L00-contextual <10% at 1% AND logits show >1% top-1 flips or KL>0.01 on skipped tokens.
**Promote to multi-layer policy search if:** either gate passes.

## Repro

Scripts: `/tmp/dynskip/s1_l20_oracle.py s2_signals.py s3_decode_split.py s4_l00.py s5_l39.py
s6_prefetch.py s7_decode_signals.py s8_transfer.py` · tables: `/tmp/dynskip/l20_ctx.npz
l00_proxy.npz l39_proxy.npz is_decode.npy`. Inputs: `probes/functional_basis_oracle/fb_assets/`
(L20 hdumps + routes), `probes/atom_factorization_one_layer/assets/shexp_L20.npz`,
`/tmp/agent1_{raw,f32}/` (routers, norms, experts). No training, no model download, no controller.
