# Dynamic Skipping — Contextual Validation (Kaggle, REAL states)

**Kernel:** `toheebogunade/jamii-native-sparse-dynskip-ctx-v1` v4 (COMPLETE, 37 min wall)
**Model:** Qwen3.5-35B-A3B-UD-IQ2_XXS pinned rev, llama.cpp 3057bb66, greedy decode.
**Data:** 2272 decode tokens (71×32 prompts), L00/L20/L39, 12 streams
(x/h0/h1/B per layer). Prompt-disjoint held split reused (22 train / 10 held).
**Routing semantics:** preserved (control↔policy route-prefix equality holds
pre-divergence; skip = exact-zero of combine weights, router/ids untouched).

## 0. Verdict: KILL the branch (all of it)

| question | result |
|---|---|
| >=30% routed-skip at <=1% (STRONG KEEP bar) | 0.00% all layers (nearest token 5.8%) |
| >=20% MoE skip, strict error | 0.00% at 1/2/5% on all layers |
| >=20% whole-block skip | 0.00% at 1/2/5% (medians 26–82%) |
| frozen L20 policy (`top1<=0.02`) flips | 19/32 prompts diverged (59%) — KILL bar was >1% |
| strongest cheap rule | NONE (every signal 0% strict-safe at every bar) |
| projected tok/s gain | +0% (nothing safely skippable) |
| full 40-layer scan | **NO** — 3/3 depth-spanning layers at 0% with 6–59% medians; no interpolation overturns that |
| logits KL | unmeasured (top-k post-logits hook unsupported); moot — flips fired 59× over bar |

## 1. Why v1 was wrong (denominator artifact, corrected)

v1 reported MoE/X ratios ~1.7% assuming RMS(hidden)≈1. The dumps prove the
residual stream is tiny and grows with depth:

| layer | \|\|h1\|\| med (RMS) | \|\|B\|\| med | \|\|X\|\| med | v1 assumed RMS | true RMS |
|---|---|---|---|---|---|
| L00 | 0.84 (0.020) | 0.25 | 24.6 | 1.0 (50× off) | 0.020 |
| L20 | 3.82 (0.085) | 0.74 | 43.5 | 1.0 (12× off) | 0.085 |
| L39 | 14.1 (0.31) | 8.37 | 58.6 | — (proxy) | 0.31 |

Post-norm X is g-scaled (norm 25–59) and tells nothing about residual scale.
True block-relative errors use Y=h1+B. Proxy MoE *output magnitudes* were
fine (~2×); proxy is fundamentally blind to residual scale, hence to skips.

## 2. Contextual tables (TRUE denominators, decode-only, n=2272)

Per-token error = ||skipped|| / ||h1+B||. Cosine = cos(kept, h1+B), exact.

| layer | target | median | min | <1% | <2% | <5% | cos mean/min |
|---|---|---|---|---|---|---|---|
| L00 | full MoE | 28.6% | 13.9% | 0% | 0% | 0% | 0.954/0.722 |
| L00 | routed | 24.9% | 8.9% | 0% | 0% | 0% | — |
| L00 | whole block | 81.8% | 56.1% | 0% | 0% | 0% | 0.626/0.390 |
| L20 | full MoE | 19.0% | 9.7% | 0% | 0% | 0% | 0.977/0.768 |
| L20 | routed | 15.7% | 5.8% | 0% | 0% | 0% | — |
| L20 | whole block | 26.2% | 16.4% | 0% | 0% | 0% | 0.961/0.759 |
| L39 | full MoE | 58.9% | 35.8% | 0% | 0% | 0% | 0.815/0.553 |
| L39 | routed | 24.9% | 4.5% | 0% | 0% | 0.13% (3 toks) | — |
| L39 | whole block | 78.7% | 49.2% | 0% | 0% | 0% | 0.729/0.416 |

Mixer/Y medians: 78% (L0), 21% (L20), 41% (L39) — branches are load-bearing,
not perturbations. Global nearest-to-bar: L39-routed min 4.5% (still misses 5%
bar for all but 3 tokens, and no cheap rule isolates them: best-signal 0%).
Margins vs the 3%-of-B local-vs-runtime noise floor (≈0.6% of Y): 8–90×. Robust.

Cheap signals (top1, entropy, top8mass, ||R8||/||X||, ||S||/||X||, residual
cosine, mixer norm, hidden norm): best strict-safe rate **0.00%** at every
bar/layer/target — oracles are empty, so signaling is moot. False-safe rate
N/A (no rule fires). Frozen-policy false-safe: 100% (476/476 skipped violated 2%).

## 3. Frozen L20 policy (`skip iff runtime top1 <= 0.02`)

- Runtime↔local top1 max diff 5e-7 (policy implemented EXACTLY).
- Skip rate 21.0% overall / 20.7% held — rate transferred, safety did not.
- Skipped-token hidden error: median 11.2%, max 22.1% (true denominator).
- **19/32 prompts diverged (59%)**, byte agreement 75%; flips at tokens
  16–73 of ~77 (early and late — single-layer ~15% errors flip top-1 fast).
- Worst cases in `tables.json` → policy.worst (8 examples with contexts).
- KILL: flips exceed the 1% bar ~59× (prompt-level) / ~25× (token-level).

## 4. Validation chain (why these numbers are trustworthy)

- Routes recomputed from dumped X match hook ids 100% / 99.96% / 100%.
- X == h1/rms(h1)·g to 4e-3 / 2e-4 / 1e-5 (eps-limited at small L0 scale).
- Local B == runtime-dumped B to 2–3% mean (s_fit=0.9994: no scale bug;
  residual is IQ2/Q8 quant-noise floor, 8–90× below verdict margins).
- Determinism re-run byte-identical (dumps/routes/generation).
- Greedy (temp 0) both arms; control log-only proven non-mutating.

## 5. Multi-layer projection

Per-layer skip map: {0%, 0%, 0%} measured → conservative map all-zeros.
Expert ms saved: 0. Projected tok/s: baseline 4.745 resident / 3.20 bounded (+0%).
A 40-layer scan is NOT justified: the kill mechanism (branch fractions
19–59%, smooth residual growth 0.4→16) is structural across all probed depths.

## 6. Prefetch side note (temporal reuse, existing 2016×40 routes)

- Adjacent-token same-layer top8 overlap: mean 31.7%, med 25% (L0: 7%, L14: 40%).
- L20 weighted router-mass reuse: mean 43.7%, med 46% (reused experts are heavier).
- Recent-window coverage: W=1: 32%, W=2: 42%, W=4: 53% (med 4/8).
- Reuse distance: med 2 tokens; 54% ≤2, 67% ≤4, 79% ≤8.
- A per-layer recent-expert residency (≈32 slots from last 4 tokens) covers
  ~half the next token's experts and ~half its router mass — real staging
  help, but half the misses remain; combine with popularity, not cross-layer IDs.

## 7. Next ONE action

**Kill dynamic skipping permanently; archive `output/dynskip-ctx-v1` dumps
(X/h0/h1/B × 3 layers × 2272 tokens) as the program's contextual calibration
asset and redirect the next sprint to zero-copy residency validation** (the
14–19% EXACT win — the only remaining class consistent with load-bearing
19–59% branch fractions; no further approximate-branch work without a new
mechanism).

## Repro

Kernel: `kaggle/native-sparse-dynskip-ctx-v1/` (v4 COMPLETE).
Analysis: `probes/dynskip_contextual/analyze.py` → `tables.json` (committed).
Raw dumps: `output/dynskip-ctx-v1/` (git-ignored, 218 MB) + Kaggle output
`toheebogunade/jamii-native-sparse-dynskip-ctx-v1`. No controller trained.
