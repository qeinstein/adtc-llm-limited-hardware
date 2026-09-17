# Phase 9A — Whole-MoE-layer execution analysis (offline, no runtime)

## Hypothesis

An exact Qwen-specific whole-layer executor (prepare activation once, one
scheduling unit for top-8, fused route weighting) could remove repeated
per-expert/per-matrix scheduling cost and clear the 5% bar.

## Method

Read-only audit of the v10 staged executor
(`phase7b_staged_pipeline_v1/raw/script.py`), the phase8c bounded patch
(which changes storage ownership only — decode arithmetic is the exact
native control), and the phase 5A/8C measured budgets. No Kaggle run.

## Measured budget

| Arm | ms/token | Role |
|---|---|---|
| Staged cold | 312.7 | budget T |
| Staged warm | 301.1 | 11.6 ms storage ceiling |
| Phase5a resident | 210.7 | of which expert GEMV ~87 ms |
| mmap control | ~225 | bounded tax ~76 ms over mmap |

Shape: 40 layers x 3 MUL_MAT_ID nodes (gate/up/down) x top-8 = 120 nodes /
960 expert-GEMVs per decode token.

## Optimistic fusion saving

Row-quant dedup, dispatch/barrier collapse (120 nodes to 40 units),
scratch-init sharing, graph-node reduction, v10 prepare/scan removal:
~5–8 ms total, i.e. 1.6–2.6% end-to-end against the 15.6 ms (5%) bar.
Phase 2 independently kills the easy kernel win inside this budget (panel
path 43% slower than generic vec-dot at 1 row/expert).

## Decision

**KILL scheduling-only fusion.** Fails the 5% rule by ~2x even under
optimistic assumptions. Highest-EV redirections: (1) the ~87 ms expert
GEMV core — a dedicated fused single-row IQ2 gate+up kernel sharing one
quantized activation; (2) the ~76 ms bounded-vs-mmap gap (copy traffic +
indirection tax, not storage stall).
