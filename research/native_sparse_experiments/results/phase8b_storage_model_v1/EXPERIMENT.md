# Phase 8B — Storage wildcard model (offline, no runtime)

## Hypothesis

Four storage-layout wildcards could cut fresh expert traffic enough to
matter: split gate/up/down residency, co-access physical placement,
route-set syscall coalescing, and duplicated hot/cluster copies.

## Configuration

- Corpus: `phase5e_route_corpus_v1` (2016 tokens, 645,120 route requests,
  9,577 unique expert bundles, 876,544-byte bundles)
- Offline model only; no model/runtime was run
- Decision rule: kill any branch whose optimistic end-to-end gain is below 5%
- Full numbers in `ANALYSIS.json` (`decision_table`, `overall_conclusion`)

## Results

Split gate/up/down residency — **KILL.** Best <=4 GiB traffic reduction
0.0496%, optimistic end-to-end gain 0.022% (decision-table replay at
2,281 slots: 0.02197%).

Co-access physical placement — **KILL.** Adjacent range counts improved
~14–20% on train layouts, but transferred bytes did not decrease; allowing
one-record gaps amplified physical traffic (~1.0–2.0x in the holdout
replay, i.e. +0.98–4.3% class increases).

Route-set syscall coalescing — **KILL.** Fewer ranges, unchanged bytes.
Phase 8C wall-clock evidence independently shows syscall reduction is not a
useful throughput branch.

Duplicated hot/cluster copies — **KILL.** Eight-expert lane costs +0.280 GB
disk and amplifies physical traffic (~1.15–2.01%); disk duplication does
not change RAM/cache traffic.

## Decision

**KILL all four; do not implement on Kaggle.** Co-access/range metrics are
not tok/s measurements. Combined with the phase 8C wall-clock ceiling
(3.7% staged), the storage-layout branch is closed; effort moves to
compute, graph execution, representation, and multi-token amortization.
