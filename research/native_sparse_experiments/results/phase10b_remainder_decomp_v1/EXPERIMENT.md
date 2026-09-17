# Phase 10B — Non-expert remainder decomposition (evidence assembly, no new run)

## Staged 138 ms remainder (312.7 total, phase8c/9b)

Resident non-expert wall = 210.7 − 87 = 123.7 ms (phase5a). Staged remainder
138 = 123.7 + ~14 ms staging overhead (reader-thread contention; v11 proved
the mechanism; zero-copy removes the readers and tests this directly).

| component | ms/token | class | source |
|---|---|---|---|
| attention projections (QKV/O/gate) | ~48 | wall bypass | phase6b: 22.9% wall saving |
| LM head | ~25 | wall bypass | phase5a-v2: 12.11% wall saving |
| Gated DeltaNet recurrence/state | ~11 | op-fraction, NEEDS wall bypass | phase5a 5.37% summed ops |
| shared expert | ~8 | op-fraction, NEEDS wall bypass | phase5a 3.82% summed ops |
| attention scores/softmax/AV | ~10? | gap | unattributed |
| norms + SwiGLU + elementwise + router | ~12? | gap | unattributed |
| sampler + scheduler + barriers | ~5? | gap | unattributed |
| staged reader contention | ~14 | residual, tested by zero-copy | 138 − 123.7 |

Wall-measured: ~73 ms (53%). Op-fraction estimates: ~19 ms. Gaps: ~27 ms.
Staging residual: ~14 ms.

## ggml/runtime tax (source analysis, bounded)

- v10 adds 1 extra barrier per MUL_MAT_ID node per token (120 nodes); GOMP
  barrier on 4 threads ≈ 1–5 us → <1 ms/token. Negligible.
- Graph plan is precomputed and reused; per-token scheduler = node dispatch
  + threadpool wakeups over ~600 nodes. No per-token allocation in steady
  state (static graph buffers). Estimated low single-digit ms.
- Route-trace hook: 120 fprintf/event per token — sub-ms.
- Provisional: generic ggml tax <5% → do NOT rewrite runtime for this reason
  alone (per contract thresholds). A custom executor is justified only by
  kernel/representation fusion (Hypotheses B/D/G), not dispatch savings.
- Confirmation needed: one "empty-graph overhead" microbench or a bypass arm
  removing ALL matmuls (bounds scheduler+elementwise directly). Queued in the
  bypass kernel below.

## Key mined facts (phase8c raw)

- Serial cold pread: 11.70 s/rep; warm: 6.08 s/rep (same 8.13 GB). Warm copy
  rate ≈ 1.33 GB/s on Kaggle (vs 2.87 local) — copy term is BIGGER on Kaggle
  than phase9b's local bound assumed.
- Prompt is I/O-bound (3.12→4.71 t/s warm), decode is compute-bound. Staged
  helps prompt +33% but decode only +2.7%: decode offers no overlap
  opportunity (each token's experts needed NOW) — so zero-copy's copy-removal
  should land almost fully in decode wall.
- Rep spread is tight (staged cold 3.190/3.214/3.192): ≥5% effects resolvable
  with n=3; the Q4-bounded-v2 ±15% spread was anomalous noise.

## Falsifiable zero-copy prediction (recorded BEFORE kernel returns)

Staged-warm 301.1 = compute+runtime (~225.6, resident) + warm-copy + thread
mgmt + contention. Zero-copy removes warm-copy (~50–70) + pthread churn (~3)
+ reader contention (~5–15), adds faults+madvise (~5–10). Predicted ZC ≈
235–245 ms/token → **4.1–4.3 tok/s (+28–35%)** at RSS ≤ staged, exact
routes/hash. <3.5 tok/s (<10%) falsifies the copy-dominance model.

## Queued: remainder-bypass kernel (ONE build, FIVE measurement-only arms)

Amortize one clone/build/download across: (1) GDN-bypass, (2) shared-expert
bypass, (3) attention-score isolation (keep projections, skip QK^T/softmax/AV
... or vice versa), (4) all-dense-matmul bypass (bounds scheduler+elementwise
+sampler), (5) resident control. Each arm is exact-arithmetic-preserving
removal (zeros), measurement-only. Resolves the ~27 ms gap + GDN/shared wall
truth + ggml-tax bound in a single ~15 min kernel. Launch after zero-copy
retrieval (or alongside Q2_K if budget allows two kernels).
