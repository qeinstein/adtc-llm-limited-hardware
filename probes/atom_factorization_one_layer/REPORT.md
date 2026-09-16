# Atom Factorization — One-Layer Falsification Experiment

**Verdict (FINAL on proxy): KILL HARD on sparse-subset pruning; LEARNED BASIS
CEILING ALSO FAILS BADLY (strong evidence against the thesis, contextual
confirmation still formally required).**
The oracle bounds only subset selection — but the decisive dense SwiGLU
student sweep (h=512→3072 vs full block B, train 2463, test held-384) fails
at every width: best test median 1.08 vs the 0.975 mean-predictor floor and
the 0.02 bar. More capacity → worse test (pure memorization); 4× data →
4% gain (flat scaling). No final architecture kill until a contextual run
lands (proxy routing logcorr 0.17), but reversal is implausible — see §11.

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
**oracle test**, which upper-bounds the sparse-subset branch (it does NOT
bound learned/synthesized atoms — see Revision note): per
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

**Not run — moot for subset schemes.** The oracle (attainable by no
subset dictionary) already fails every bar (see §5). Concretely, at the
subset hypothesis's operating points:

| Operating point | Oracle (ceiling) held-out error |
|---|---|
| BREAKTHROUGH M≤512, ≤2% | 37% median — off by **18×** |
| STRONG KEEP M≤1024, ≤2% | 23% median — off by **11×** |
| PROMISING M≤2048, <2% | 8.9% median — off by **4×** |
| KILL bar: >5% near M=2048 | **8.9% median → KILL fires** |

No subset clustering, community structure, or private set can recover a
4–18× gap to an upper bound. Private atoms are further ruled out
structurally: the failure is in the *bulk* (see §6), not in outliers.
(Learned atoms are a different branch — see §8–§10.)

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

## 6. Rare-event analysis (subset hypothesis)

There is no rare-event story — the subset failure is systematic and tight:

- At M=2048: median 8.9%, p99 11.4%, max 12.4%. The spread is narrow;
  *typical* tokens fail, not outliers. No mean-MSE hiding: median ≈ mean
  at every M.
- Mechanism (all 320 calib tokens): Σ||C||²/||R8||² = 0.87 mean
  (no cancellation pathology — contributions align); top-2048 atoms carry
  99.2% of atom energy, yet dropping the bottom 2048 loses 8.6% coherent
  signal. The tail is ~2000 small ALIGNED atoms: real signal, spread thin.
  No subset trick, refit, or outlier handling recovers it.
- Consequence: a private/outlier set cannot help — the missing component
  is bulk, not outliers. This rules out branch D (shared+private) as
  firmly as branch B (global).

## 7. Projected compute/storage savings

For sparse-subset execution the projections are moot (killed): the oracle
itself needs 2551/3137 atoms for 5%/2% (1.6×/1.3× mult reduction vs the
teacher's 25.2M mults/token/layer), below every bar, and no realizable
scheme reaches oracle. Reference points only:

| config | mults/token/layer | R | layer ms (2.75 base) | tok/s (5.196 frame) |
|---|---|---|---|---|
| teacher (8×512) | 25,165,824 | 1.0× | 2.75 | 5.20 |
| oracle M=2551 @5% | ~15.7M | 1.6× | 1.72 | 6.73 |
| oracle M=3137 @2% | ~19.3M | 1.3× | 2.11 | 6.06 |
| hypothesis M=512 @2% | ~3.2M | 8× | 0.34 | 10.4 |

(Assumes ~110 ms/token total MoE time over 40 layers, 82.5 ms non-MoE;
see script `07_tables.py` for the model. The last row is what the
hypothesis needed and did not get — not even at oracle level.)

For the LEARNED-basis branch, projections await the student sweep: a dense
h=1024 student costs 3×1024×2048 = 6.3M mults (4× reduction); h=2048 costs
12.6M (2×). Reductions ≥2× are live iff the student fits.

## 8. KEEP / KILL verdict

**KILL HARD on sparse-subset pruning; learned-basis ceiling FAILS (strong
evidence against the thesis; contextual run still required for final kill).**

- KILL HARD: pruning / sparse subset of original atoms (branches B/C/D as
  subset schemes). Oracle: 8.9% median at M=2048 vs the 5% bar; 2551 atoms
  needed for 5%. Do not pursue atom pruning.
- LEARNED BASIS CEILING FAILS: dense SwiGLU students h=512/1024/2048/3072
  vs full block B=R8+S (train 2463, val 256, test held-384) reach test
  medians 1.08/1.12/1.17/1.19 — all at or above the 0.975 mean-predictor
  floor, ~50× above the 1–2% bars. More capacity → worse test
  (memorization: h=3072 train 0.09 / test 1.19); val never beats 1.0 at any
  checkpoint for any width. The 4× data-scaling check is flat (1.093→1.051
  pooled). Independent corroboration: peer s2_06 L0/proxy run gets pooled
  0.95/0.97/0.90/0.90 across the same widths — floor-level everywhere.
- Formal caveat: inputs are proxy (routing logcorr 0.17), so the FINAL
  architecture kill awaits one contextual run. But the failure mechanism —
  inability to learn input-dependent 8-of-256 routing + expert outputs,
  with flat data scaling — is distribution-agnostic; contextual inputs are
  higher-entropy, not easier. Reversal is implausible.

## 9. Strongest version discovered

- Subset branch: the oracle itself (unrealizable, and still dead).
- Learned branch: h=512 student (smallest, least-overfit) at test median
  1.08 — still above the mean floor. There is no surviving version; every
  width fails, and capacity hurts. The coherent-tail (§6) and
  memorization-dynamics (§11) findings jointly say the routed block's
  input-dependent structure is not compressible into few nonlinear units
  by any method tested.

## 10. Next ONE experiment

ONE contextual closer run: collect ~400 real contextual L20-MoE inputs
(llama.cpp hidden-state dump of the 10.66 GB GGUF — needs a bigger box;
15 of 20 prefix layers are Gated DeltaNet, ruling out hand-rolled prefix
forward; local disk/RAM cannot host it), then run the frozen student
protocol (`11`, h=2048+3072 only) against exact R8/S targets computed from
cached L20 experts. Bar: h=3072 test median <5% would reopen the thesis;
anything near floor (≈1.0) finalizes the kill. Do NOT run
global/community dictionary clustering — the ceiling it would need
(a fittable learned basis) has already failed.

## 11. Learned-basis ceiling: dense SwiGLU student (L20, B=R8+S)

Protocol (`11_student_big.py`, `11b`, `12_datascale.py`): train 2463 /
val 256 (fresh proxy pool + calib) / test held-384 fixed disjoint IDs;
input+target standardized; minibatch Adam 400 epochs lr=3e-4 with
val-checkpointing (keep best of 20); 1 seed. Targets exact (cached L20
experts + fetched shared expert, |B|=1.42). Baselines on held-384:
linear ridge pooled 0.994, mean predictor 0.975.

| h | best ep | train | pooled | median | p95 | p99 | max | cos |
|---|---|---|---|---|---|---|---|---|
| 512 | 20 | 0.848 | 1.051 | 1.080 | 1.193 | 1.255 | 1.666 | 0.156 |
| 1024 | 20 | 0.694 | 1.082 | 1.118 | 1.244 | 1.331 | 1.748 | 0.155 |
| 2048 | 20 | 0.462 | 1.116 | 1.167 | 1.323 | 1.369 | 1.696 | 0.165 |
| 3072 | 400 | 0.093 | 1.134 | 1.194 | 1.389 | 1.503 | 2.145 | 0.164 |

Reading: (1) every width is at/above the mean floor — no generalizable
signal learned (val never beats 1.0 at any checkpoint); (2) capacity
inverts the ranking — h=3072 memorizes train (0.09) and tests worst;
(3) h≤2048 pick the FIRST checkpoint (ep 20) — later training is pure
memorization. Small-N reference (`09`, train 256): h=512 train 0.011 /
test median 1.32 — same overfit regime, motivating the expansion.

Data scaling (`12`, h=512): n=615 → pooled 1.093; n=1231 → 1.078;
n=2463 → 1.051. Four times the data buys 4% — flat. No reasonable N
closes a 50× gap to the 2% bar.

Corroboration: independent peer run s2_06 (L0, N=320 proxy, full-batch
1500 steps): pooled 0.945/0.971/0.900/0.897 at h=512→3072 — floor-level
at a different layer with a different protocol. (Both that run and our
first full-grid run were OOM-killed on side computations — exit 137 —
after printing all widths; numbers preserved from logs.)

Cost note: h=3072 needed a batch-128 rerun (`11b`) after the OOM kill;
same protocol otherwise.

## Revision note (2026-09-15)

v1 marked the whole hypothesis KILL. Corrected (v2): the oracle bounds
only subset selection — learned basis stayed OPEN with the student sweep
as decider. v3 (this): student ceiling fails at all widths with flat data
scaling ⇒ strong evidence against the thesis; only the formal contextual
closer remains (§10).
...[truncated 5019 chars]