# Atom Factorization — One-Layer Falsification Experiment

**Verdict: KILL** (strict criterion 1 fires with margin; no dictionary variant can survive the ceiling)

**Branch:** `research/atom-factor-one-layer`
**Date:** 2026-09-15 · **Layer:** L20 · **Held-out tokens:** 384 (disjoint IDs)

---

## 1. Exact layer chosen and why

**L20** (of 40), because:

- Middle layer ⇒ most representative of steady-state MoE behavior
  (L0 is embedding-adjacent, L39 is unembedding-adjacent; both atypical).
- Full 256/256 experts already cached locally as dequantized f32
  (`/tmp/agent1_f32/L20_*`, pinned UD-IQ2_XXS GGUF), so the experiment
  needed zero expert-weight downloads and covered the complete layer.
- All 256 experts actually route at least once over our 704 tokens
  (union calib 249 / held 251), so nothing was extrapolated.

## 2. Dataset / calibration size

| Split | Tokens | Token IDs | Source |
|---|---|---|---|
| calib | 320 | existing `probes/agent3_funcmoe/weights/calib_ids.json` | real clinical EN/SW token IDs |
| held-out | 384 | newly mined, **disjoint** from calib (asserted) | real token IDs from `data/*.json` via real Qwen3.5 tokenizer |

Inputs use the repo's established **proxy-REAL** methodology (same as
`agent2_attn` / `agent3_funcmoe`): `x = RMSNorm(real bf16 embed row) ×
real post_attention_norm scale`. Router is the REAL F32 GGUF router with
exact llama.cpp semantics (softmax/256, top-8, renorm). Expert weights are
REAL IQ2-dequantized f32; all math f32 with f64 accumulation.

Validation of the proxy: top-1/2/4/8 routing mass =
0.026/0.042/0.067/0.108 (diffuse routing); selection-frequency logcorr vs
the 2016-token REAL route corpus (`phase5e_route_corpus_v1`) = **0.17**
(weak — see Limitations). Teacher |R8| = 1.07; expert mean-output cosine
+0.07 (low redundancy); expert norm max/min = 2.34×.

## 3. Method

Teacher: `R8(x) = Σ_{e∈top8} α_e Σ_j s_{e,j}(x)·d_{e,j}`, 4096 atoms/token.

**Ceiling-first design.** Before building any dictionary, we ran the
**oracle test**, which upper-bounds the entire hypothesis family: per
held-out token, rank the *true* 4096 routed-atom contributions by norm,
keep top-M with *true* coefficients (best per-token subset, no sharing
loss, no composition loss). Any routing-composed dictionary (fixed expert
codes, shared medoid atoms, β-selection without per-token solves) can only
do worse: it is strictly less adaptive (one global atom set for all
tokens) and its atoms are single SwiGLU neurons, so no "super-atom" can
cover several true atoms in one nonlinear eval.

Controls:
- **Sanity:** oracle at M=4096 reconstructs R8 to 0.0000 (atom sum = R8).
- **Refit headroom:** top-M support + per-token least-squares coefficient
  refit on 64 held tokens (NOT part of the hypothesis; tests whether the
  failure is subset-selection vs coefficients).
- **Coherence diagnostics:** Σ||C||²/||R8||² (cancellation check) and
  top-2048 energy fraction, computed for all 320 calib tokens from saved
  signatures (seconds, no expert reloads).

Scripts `04_global_dict.py`, `05_community.py`, `06_private.py`,
`07_tables.py` + `lib.py` are implemented, syntax-checked, and lib math is
unit-verified (`/tmp/selftest_lib.py`: exact-recovery check passes), but
were **deliberately not run**: the ceiling test killed the idea, and the
task rules say to stop spending time on a dead approach.

## 4. Global vs community vs private results

**Not run — moot.** The oracle (attainable by no dictionary) already fails
every bar (see §5). Concretely, at the hypothesis's operating points:

| Operating point | Oracle (ceiling) held-out error |
|---|---|
| BREAKTHROUGH M≤512, ≤2% | 37% median — off by **18×** |
| STRONG KEEP M≤1024, ≤2% | 23% median — off by **11×** |
| PROMISING M≤2048, <2% | 8.9% median — off by **4×** |
| KILL bar: >5% near M=2048 | **8.9% median → KILL fires** |

No clustering, community structure, or private set can recover a 4–18×
gap to an upper bound. Private atoms are further ruled out structurally:
the failure is in the *bulk* (see §6), not in outliers.

## 5. Error-vs-active-atoms table (held-out, n=384)

Oracle, best-subset / true-coeffs:

| M | mean | median | p95 | p99 | max | cos |
|---|---|---|---|---|---|---|
| 128 | 0.593 | 0.617 | 0.681 | 0.696 | 0.729 | 0.799 |
| 256 | 0.482 | 0.503 | 0.563 | 0.580 | 0.625 | 0.872 |
| 512 | 0.357 | 0.373 | 0.429 | 0.443 | 0.478 | 0.931 |
| 1024 | 0.221 | 0.230 | 0.272 | 0.283 | 0.316 | 0.974 |
| 2048 | 0.086 | 0.089 | 0.106 | 0.114 | 0.124 | 0.996 |
| 4096 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

Atoms the *oracle itself* needs (median over tokens): **2551 for <5%,
3137 for <2%, 3439 for <1%** (max 2819/3316/3560; every token eventually
converges). Teacher uses 4096. Oracle-level savings are 1.6×/1.3×/1.2× —
and no realizable dictionary reaches oracle.

Refit control (64 tokens, optimal per-token coefficients on top-M support):
M=128: 0.593→0.575; M=256: 0.482→0.446; M=512: 0.357→0.302. Refit barely
helps ⇒ the failure is **missing subspace** (signal lives in unselected
atoms), not coefficient quality. No code-fitting scheme fixes this.

## 6. Rare-event analysis

There is no rare-event story — the failure is systematic and tight:

- At M=2048: median 8.9%, p99 11.4%, max 12.4%. The spread is narrow;
  *typical* tokens fail, not outliers. Hiding behind mean MSE is
...[truncated 5019 chars]