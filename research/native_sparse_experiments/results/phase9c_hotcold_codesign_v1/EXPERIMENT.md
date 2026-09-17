# Phase 9C — Hot/cold expert co-design model (offline corpus analysis)

## Hypothesis

Storing hot experts in a larger/faster (2x-byte class) representation while
cold experts stay IQ2_XXS could cut expert compute for most accesses without
exploding RAM.

## Method

Frequency analysis of `phase5e_route_corpus_v1` (2016 tokens, 645,120
accesses, 9,577 unique bundles). Assumes a 2x-byte hot format that halves
expert GEMV time (optimistic); kernel savings apply only to covered
accesses' share of the ~87 ms expert core.

## Results

- Global skew is weak (top-256 bundles cover only 20.1% of accesses).
- Per-layer skew is moderate: top-16/32/64 per layer cover 36.4% / 53.0% /
  72.6% mean (min 20.5% / 34.2% / 54.0%).
- Cache cost at 2x bytes: top-32/layer displaces 1280 of 2281 slots (56%
  of the 2 GB cache) and adds 1.12 GB; top-16 displaces 640 (28%).
- Optimistic compute saving: top-32 → 0.53 × 87 × 0.5 ≈ 23 ms (7.4%);
  top-16 → ≈15.7 ms (5.0%). Both BEFORE displacement cost, which
  re-inflates fresh traffic for the remaining accesses and eats most of
  the gain. Net plausibly <5%, possibly negative at top-32+.

## Decision

**NO-GO as a standalone Kaggle branch** — fails the >5%-plausible rule
once cache displacement is priced. Keep only as a possible add-on inside a
larger representation redesign, not as its own experiment.
