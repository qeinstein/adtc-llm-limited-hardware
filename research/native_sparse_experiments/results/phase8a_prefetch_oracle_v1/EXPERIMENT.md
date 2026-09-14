# Phase 8A — Prefetch oracle and route-only predictor

## Hypothesis

A perfect-next-token prefetch oracle bounds the value of hiding expert
fetch latency; a small deployable route-only transition predictor captures
part of that bound without touching native routing.

## Configuration

- Corpus: `phase5e_route_corpus_v1` (2016 tokens, 645,120 route requests)
- Simulator: single serialized NVMe service, 767 decimal MB/s, 0.05 ms
  plane latency, 876,544-byte bundles, compute 225.56 ms/token
- Predictor: transition frequencies conditioned on (layer, current top-8),
  trained on prompts 0–15, evaluated held-out on prompts 16–31; predicted
  IDs issue speculative reads only, demand routes stay exact
- Tests: `test_prefetch_oracle.py`, `test_prefetch_predictor.py` (5 passed)

## Results

Perfect-next-token oracle at the 2,281-slot / 2 GB cache point (simulated):

- baseline 2.976 tok/s (110.46 ms/token unhidden stall) -> oracle
  4.239 tok/s (10.35 ms/token unhidden stall), i.e. ~+42% projected

Deployable predictor at the 826-slot / 2.5 GiB cache point (held-out):

- copy-last fallback: 1.000054x vs baseline (~0.005%)
- transition-frequency: 1.010008x vs baseline (~1.0%)
- train-set transition diagnostic: 1.6475x (overfit reference, not
  deployable — memorized transitions do not transfer)

## Recalibration against measured wall clock

The oracle's +42% class of projections assumes fresh bytes block at modeled
SSD bandwidth. The phase 8C critical-path measurement on real Kaggle
hardware bounds fully removing physical storage I/O at ~7.5% serial and
~3.7% staged end-to-end. The simulator overstates the opportunity by roughly
6x because real execution already overlaps/hides most I/O. The deployable
held-out predictor itself only achieves ~1.0% in simulation — before the
recalibration discount is even applied.

## Decision

**KILL under the <5% rule.** The staged executor's measured storage ceiling
is 3.7%; the best deployable predictor shows ~1.0% in an optimistic
simulator. No further route-prediction-for-prefetch work unless it exposes a
mechanism beyond hiding physical reads.

Simulator, predictor, oracle outputs, and tests are preserved in this
directory for reuse as a falsification tool against future storage claims.
