# Atom Factorization — One-Layer Falsification Experiment

**Verdict (REVISED): KILL HARD on sparse-subset pruning; OPEN on learned nonlinear basis.**
The oracle only upper-bounds sparse SUBSET SELECTION from the original 4096
atoms — it does not bound a dictionary of newly learned/synthesized
nonlinear atoms, which can span directions no small original subset spans.
The decisive learned-basis ceiling (dense SwiGLU student sweep) is tracked
separately; no final architecture kill until it lands on real contextual
states (current proxy routing logcorr 0.17 is insufficient for that).

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

**KILL HARD on sparse-subset pruning; OPEN on learned nonlinear basis.**

- KILL HARD: pruning / sparse subset of original atoms (branches B/C/D as
  subset schemes). Oracle: 8.9% median at M=2048 vs the 5% bar; 2551 atoms
  needed for 5%. Do not pursue atom pruning.
- OPEN: learned nonlinear basis / newly synthesized atoms. The oracle does
  not bound learned atoms. Decisive test = dense SwiGLU student sweep
  (`09`, in flight): h≤1024 at ~1–2% ⇒ major KEEP; h≤2048 strong ⇒ KEEP;
  even h=3072 failing badly on real contextual held-out ⇒ strong evidence
  against the broader thesis. No final architecture kill on proxy data
  alone (routing logcorr 0.17 insufficient).

## 9. Strongest version discovered

- Subset branch: the oracle itself (unrealizable, and still dead).
- Learned branch: TBD by the student sweep. Prior: s2_06's L0/proxy
  dense-student run reports test_rel ≈ 0.95–0.97 at h=512/1024 (pooled,
  N=320 proxy) — a bad omen, but L0 ≠ L20 and pooled-rel ≠ tails; our
  L20 run (N=704, held-384 tails) is the cleaner read.

## 10. Next ONE experiment

The dense SwiGLU student sweep (`09_student.py`, h=512/1024/2048/3072 vs
B=R8+S, train calib-256, test held-384, median/p95/p99/max/cosine) — in
flight at time of writing. Global/community dictionary clustering stays
parked until this ceiling lands. If the student succeeds on proxy, the
mandatory follow-up is re-testing on REAL contextual L20 states; if it
fails badly on proxy, the contextual run is still required before a final
thesis kill (scoped separately — needs full-model inference: 15 of 20
prefix layers are Gated DeltaNet, ruling out a hand-rolled prefix
forward; realistic path is a llama.cpp hidden-state dump of the 10 GB
GGUF, which currently does not fit local disk/RAM).

## Revision note (2026-09-15)

v1 of this report marked the whole hypothesis KILL. Corrected: the oracle
bounds only subset selection, not learned atoms. §3–§5 numbers are
unchanged (they concern the subset branch only); §8/§10 now reflect
KILL-HARD-subset / OPEN-learned with the student sweep as decider.
...[truncated 5019 chars]