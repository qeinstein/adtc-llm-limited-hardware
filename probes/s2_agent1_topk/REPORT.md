# S2 AGENT 1 — Expert conditionality: REPORT (parent-written; agent died on network)

Method: the agent (two attempts, both killed by transport errors) left
complete measurement code + data: `code/01_routing.py` (all-40-layer
routing), `code/02_fetch.py`, `code/03_forward.py` (R8/Rk forward engine),
`code/05_hfcheck.py`, `results/routing_all40.json`, `results/top8_all40.npz`,
`results/union_curve.json`, `results/hf_crosscheck.json`, and
`results/forward_{L00,L20,L39}_{proxy,gauss}.json` (N=320 each; proxy =
real-embedding-derived states, gauss = matched-RMS Gaussian replication).
The parent verified the JSONs, cross-checked proxy-vs-gauss agreement, and
wrote the verdicts. Real bf16/HF + GGUF weight bytes throughout.

## 0. Verdict: KILL k-reduction, KILL width reduction, KILL sparsity

Dropping even ONE of 8 experts rewrites ~1/4 of the routed output on every
measured layer. The router is diffuse (top-8 holds 9-21% of mass), every
rank carries material norm share, SwiGLU has zero block sparsity, and
width-cut energy arguments fail per-expert minima. Nothing promotable.

## 1. Top-k output error ||R8-Rk||/||R8|| (mean; proxy/gauss agree)

| layer | k=7 | k=6 | k=5 | k=4 | k=3 | k=2 | k=1 |
|---|---|---|---|---|---|---|---|
| L00 | 0.21 | 0.34 | 0.48 | 0.65 | 0.87 | 1.21 | 1.91 |
| L20 | 0.24 | 0.39 | 0.54 | 0.72 | 0.96 | 1.33 | 2.08 |
| L39 | 0.25 | 0.42 | 0.59 | 0.78 | 1.04 | 1.43 | 2.23 |

Gaussian replication matches within ~0.03 (slightly worse) on every cell:
the result is robust to activation direction, not a proxy artifact.

## 2. Why: diffuse router, flat ranks (L00/L20/L39 proxy)

- Cumulative top-8 mass: 0.213 / 0.110 / 0.094 (of 256 experts).
- Rank masses L00: 0.060, 0.038, 0.027, 0.022, 0.020, 0.017, 0.016, 0.014.
  Rank-8 still holds 6.7% of top-8 mass (L00 norm share 7.8%).
- Leave-one-rank-out output error: rank-1 0.58, rank-4 0.28, rank-8 0.21
  (L00). Every rank matters; there is no dispensable tail.
- Union curve: 320 tokens touch 214-252 of 256 experts per layer — routing
  is spread over nearly the whole pool, not concentrated.

## 3. SwiGLU z = silu(g)*u: no sparsity lever

- Near-zero fraction |z|<0.01: 4.0-9.0% across layers/models.
- Structured block sparsity (16/32/64): exactly 0.0 everywhere.
- Width energy (routed, mean/min over experts): 384 -> 0.958-0.999 /
  0.865-0.959; 256 -> 0.862-0.984 / 0.671-0.840. Mean looks plausible but
  worst experts collapse, and energy is not output error: no measured
  forward-error basis exists. KILL under the accuracy invariant.

## 4. Genuine curiosity (not a path): L00 is half-dead

L00 gate/up have 280/512 exactly-zero rows (dead_frac 0.437, live_dims 298
vs 511-512 elsewhere), verified IDENTICAL in HF bf16 and GGUF
(zero_pattern_agree 1.0). Real model artifact, not quant noise. But it is
one layer of forty: even a perfect L00-only width cut saves ~1% of expert
compute. Noted for a future micro-optimization pass, not this sprint.

## 5. Verdict table (S2 realistic baseline 172 ms / 5.8 tok/s; expert region
   ~110 ms of it; savings assume routed GEMV scales with k — generous)

| candidate | measured output err | ms saved | tok/s | RAM | difficulty | verdict |
|---|---|---|---|---|---|---|
| avg k 8->7 | 21-28% | ~8-10 | ~6.1 | 0 | LOW | KILL (catastrophic/quality) |
| avg k 8->6 | 34-45% | ~16-20 | ~6.5 | 0 | LOW | KILL_HARD |
| avg k 8->4 | 65-81% | ~33-40 | ~7.2 | 0 | LOW | KILL_HARD |
| avg k 8->2 | 121-145% | ~50-60 | ~8.0 | 0 | LOW | KILL_HARD |
| per-layer k (strict err<5%) | no layer admits k<8 (min err 21%) | 0 | 5.8 | 0 | — | KILL (no operating point) |
| width 512->384 | unvalidated fwd err; per-expert energy min 0.86 | ~0-8* | ~5.8-6.1 | small | MEDIUM | KILL (no error basis) |
| width 512->256 | min-expert energy 0.67 | ~0-15* | ~5.8-6.3 | small | MEDIUM | KILL |
| SwiGLU sparsity skip | 0% block sparsity | 0 | 5.8 | 0 | — | KILL (no signal) |

*Width savings assume a valid cut existed; it does not. All errors are
routed-output-domain relative error, far above any promotable threshold,
and every candidate that keeps quality (none found) would stack trivially.

## 6. Reproducibility

Rerun `code/03_forward.py {0,20,39} proxy,gauss` (needs the fetched weight
slices the agent used; see `code/02_fetch.py`). Proxy-vs-gauss agreement on
all headline numbers is the built-in cross-check. No training was done.
