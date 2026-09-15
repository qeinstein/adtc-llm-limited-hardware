# Agent 2 — Attention gate / query-head reduction (Qwen3.5-35B-A3B gated full-attention)

Workspace: `/home/fluxx/Workspace/adtc-llm-native-sparse/probes/agent2_attn/`
Target region: attention projections, 47.5 ms/token of a 215.2 ms/token baseline (4.647 tok/s).
Goal: <66.7 ms/token (>15 tok/s), <3 GiB RAM, accuracy invariant. No training.

## 1. Architecture (inspected, not assumed)

- HF `config.json` (`Qwen/Qwen3.5-35B-A3B`): hidden 2048, 40 layers, 10 full-attention
  layers (3,7,...,39), GQA 16 Q / 2 KV, head_dim 256, `attn_output_gate: true`.
- `llama.cpp` `qwen35moe.cpp::build_layer_attn` + HF `qwen3_next` modeling agree:
  - Fused QG projection `q_proj`: 2048 -> 8192, per-head contiguous `[q_h(256), g_h(256)]`.
  - `o_h = Attn(RMSNorm(q_h), K, V)` (GQA, 8 Q heads per KV head), partial RoPE.
  - `y_h = sigmoid(g_h) * o_h` (elementwise), `z = W_o [y_0..y_15]`, `W_o`: 4096 -> 2048.
- GDN (linear-attn) layers: fused `qkv` 2048 -> 8192 + gate `z` 2048 -> 4096 (30 layers).

## 2. Method (all values from REAL weights)

- Range-fetched bf16 weights from HF safetensors shards 13/14 (+ real `embed_tokens`
  rows for 117 calibration token IDs from 4 clinical EN/SW prompts, shard 9):
  layers {3, 19, 39} (early/mid/late): `q_proj, o_proj, k_proj, v_proj, q_norm,
  k_norm, input_layernorm`. See `weights/manifest.json`. Scripts `01..04`.
- Proxy states `x = RMSNorm(emb) * input_ln` (N=117). The small activation scale is
  STRUCTURAL, not a proxy artifact: learned `input_ln` RMS is 0.15/0.26/0.15, so
  true layer inputs have the same ~0.2 RMS (norm output scale is set by `w`,
  independent of hidden-state direction). Only input *direction* is proxy-dependent.
- Decode simulation: K/V cache = real projections of all 117 states; 32 decode
  queries; no RoPE (A2). Measured attention is ~uniform (entropy 4.70 vs 4.76 max),
  which UNDERSTATES head differentiation -> pruning errors below are conservative
  (true peaked-attention damage is likely HIGHER).
- Gaussian-x replication (matched RMS) agrees with embedding-proxy on every
  headline number (gate 0.085-0.127 vs 0.090-0.148; abl-med 0.27-0.45 vs 0.26-0.46;
  lowrank8 0.056-0.088 vs 0.061-0.114; static8 0.61-1.01 vs 0.48-0.90).

## 3. Measured results (REAL weights, mean over L3/L19/L39 unless noted)

### 3.1 Gate magnitude distribution (pre-sigmoid `g`, 117 toks x 16 heads x 256 dims)
| layer | mean|g| | p50 | p90 | p99 | std | P(sig<0.01) | P(sig<0.10) | P(sig>0.90) |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.092 | 0.076 | 0.191 | 0.318 | 0.116 | 0 | 0 | 0 |
| 19 | 0.148 | 0.119 | 0.310 | 0.555 | 0.185 | 0 | 0 | 0 |
| 39 | 0.090 | 0.070 | 0.189 | 0.378 | 0.122 | 0 | 0 | 0 |

Near-zero gate dim pct: **0.00%** (100% of sigmoid(g) in (0.1, 0.9)).
Within-head dim std (mean over heads/toks): 0.101 / 0.157 / 0.105.
Token std per dim (mean): 0.109 / 0.171 / 0.108.
Gate row norms ||w||: 0.77 / 0.59 / 0.72 (Q rows: 0.71 / 0.70 / 0.81) — same scale.
Head gate-vector spectral norm: 4.6 / 4.8 / 4.5 (worst-case alignment COULD saturate;
typical alignment does not; true-state alignment unmeasured).

### 3.2 Effective rank per head gate vector
| layer | eff-rank (activations G_h, 117x256) | eff-rank (weight W_g,h, 256x2048) |
|---|---|---|
| 3 | 36.3 | 135.1 |
| 19 | 26.5 | 73.1 |
| 39 | 31.7 | 105.8 |

Weight spectrum is white-ish (73-135 of 256) -> no shared low-dim gate subspace.

### 3.3 Low-rank gate G_h(x) ~= U_h a_h(x)
Activation-SVD (oracle coefficients), variance explained / rel recon err:
r1: 29%/84%, r2: 38%/78%, r4: 46%/73%, r8: 55%/67%, r16: 67%/58% (means).
Deployable W-SVD (a(x)=V_r^T x), BLOCK-OUTPUT-domain rel err ||dz||/||z||:
r1: 9.0%, r2: 8.8%, r4: 8.4%, r8: 7.9%, r16: 7.3% (means; per-layer r8: 6.1/11.4/6.3%).
The rank curve is FLAT (r1 ~= r16): gates carry little information, but the
residual 6-11%/block error has no clean operating point and is unvalidated end-to-end.

### 3.4 Per-head output contribution (share of SUM_h ||c_h||, 32 decode queries)
L3: heads 0-7 ~5.0-5.6%, heads 8-15 ~6.7-8.0% (KV-group-1 stronger).
L19: heads 0-7 ~4.9-5.3%, heads 8-15 ~7.1-7.7%. L39: ~uniform 5.5-7.3%.
Share std across tokens ~0.001-0.003 -> contributions CONSTANT per token.
Wo per-head Fro norms uniform within 6% (L3/L19), 15% (L39). No prunable heads.

### 3.5 Single-head ablation rel err ||c_h||/||z|| (exact: output is a linear head sum)
Median over heads: L3 0.26, L19 0.29, L39 0.46. Mean over layers per head: 0.31-0.42,
all 16 heads. Dropping even ONE head rewrites ~1/3 of the block output.

### 3.6 Static top-N (drop globally-lowest-||c|| heads) and dynamic oracle top-N
Block-output rel err, mean over 3 layers:
| keep | 15 | 14 | 12 | 8 | 4 | 2 | 1 |
| static | 0.30 | 0.45 | 0.69 | 0.67 | 0.94 | 1.08 | 1.00 |
| dynamic oracle | 0.29 | 0.44 | 0.59 | 0.70 | 0.95 | 1.04 | 1.01 |
Dynamic ~= static everywhere -> NO token-dynamic signal (oracle itself fails).
Note: L19 static12 (0.75) > static8 (0.48): verified vector cancellation between the
drop-4 set (heads 6,5,4,1) and the next-4 set (cos = -0.77). Non-monotonicity is real.

## 4. Millisecond model (A3: decode GEMV-bound, ms propto weight-MACs)
GDN 755.0 + FULL 272.6 = 1027.6 MMAC/token; 47.5 ms -> 0.0462 ms/MMAC.
VALIDATED vs phase6b measured TSC: qkv 48.9% vs 48.3%, GDN-gate 24.5% vs 25.3%,
full-QG 16.3% vs 16.0% (all within 0.8pp).
Full-attention projections = **12.60 ms** of the 47.5 ms (GDN qkv+gate = 73.6%).
Per-head-index (QG rows + O cols, all 10 layers): 15.73 MMAC = **0.727 ms**.
RAM bytes at A4: dense Q5_K ~= 0.69 B/param (phase7c attn/GDN Q5_K families).
Ceiling: erasing ALL full-attn projections -> 202.6 ms (4.94 tok/s), still 3x off goal.

## 5. Candidate verdicts (baseline 215.2 ms = 4.647 tok/s for all rows)

| candidate | kind | measured block-output err | ms saved | tok/s | RAM | difficulty | quality risk | verdict |
|---|---|---|---|---|---|---|---|---|
| gate rank-4/head | gate compression (U_h a_h) | 8.4% | 3.81 | 4.731 | -56.9 MB | MEDIUM (split fused QG in GGUF+llama.cpp; 2 GEMVs/head) | HIGH (6-13%/block unvalidated E2E; flat rank curve; no train recourse) | KILL |
| gate rank-8/head | gate compression | 7.9% | 3.74 | 4.729 | -55.8 MB | MEDIUM | HIGH | KILL |
| gate rank-16/head | gate compression | 7.3% | 3.60 | 4.726 | -53.8 MB | MEDIUM | HIGH | KILL |
| static 16->12 | query-head pruning | 68.9% | 2.91 | 4.710 | -43.4 MB | LOW (static reshape) | CATASTROPHIC | KILL_HARD |
| static 16->8 | query-head pruning | 67.1% | 5.82 | 4.776 | -86.8 MB | LOW | CATASTROPHIC (even 16->15 = 30%) | KILL_HARD |
| static 16->4 | query-head pruning | 93.8% | 8.72 | 4.843 | -130.2 MB | LOW | CATASTROPHIC | KILL_HARD |
| dynamic oracle top-8 | token-dynamic heads | 69.6% (oracle lower bound) | <=5.82* | <=4.776 | 0 | HIGH (per-token masked QG+O GEMV in llama.cpp) | CATASTROPHIC | KILL |
| dynamic oracle top-4 | token-dynamic heads | 95.4% (oracle lower bound) | <=8.72* | <=4.843 | 0 | HIGH | CATASTROPHIC | KILL |

*Optimistic: dispatch/gather overhead assumed 0 (unmeasured); deployable error >= oracle.
All errors are block-output-domain rel err, mean over L3/L19/L39, 32 decode queries,
REAL bf16 weights. Pruning errors are conservative (uniform-attention sim understates
head differentiation; peaked real attention differentiates heads more).

Why KILL across the board:
- Heads are incoherent-equals (shares 5-8%, Wo norms within 6-15%, share-std ~0.002):
  zero prunable heads, zero token-dynamic signal (dynamic oracle identical to static).
- Gates are near-constant (100% sigmoid in (0.1, 0.9), zero near-zero dims) yet
  low-rank gate maps still perturb blocks 6-11%: no clean operating point, and 1.7%
  end-to-end is below the 5% promotion bar for an approximate, unvalidated change.
- Structural ceiling: the full-attn head/gate attack surface is only 12.6 of 47.5
  attn ms; 73.6% of attn-projection work is GDN qkv/gate (sibling scope).

## 6. Reproducibility notes

- `results.json` bit-identical across reruns; L19 top-N curve independently
  reproduced line-for-line (0.7507/0.4805/...).
- Debugging trap encountered and resolved: `Tensor.norm(-1)` binds `-1` to the
  norm ORDER `p`, not `dim` (demo: same tensor gives 0.907 vs 2.6e-08). All probe
  files use keyword `dim=-1`; a throwaway one-liner using positional `-1` produced
  phantom 1e-8 norms and was discarded. Verified: only `.norm()` (full Frobenius)
  and `.norm(dim=-1)` appear in `02_analyze.py` / `04_corroborate.py`.
- No commits, no pushes. All artifacts under `probes/agent2_attn/`.
