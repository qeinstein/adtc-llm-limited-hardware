# Sprint close-out: architecture + hardware exploitation (2026-09-15)

Parent-authored synthesis. Finished agent reports speak for themselves;
`hw_profile` and `hw_agent3_memory` REPORTs are parent-run (agents stalled;
parent rebuilt harnesses, re-ran, verified, and wrote verdicts).

## What ran

Two sprints on the native-sparse Qwen3.5-35B-A3B path (baseline 215.2 ms/token,
~4.65 tok/s; goal >15 tok/s, <3 GiB RAM, accuracy invariant).
Box: Intel N100 (Gracemont, 4C/4T, no SMT, AVX2, no AVX512), KVM guest,
~2.7 GB RAM, no `perf`/PMU binary (used perf_event_open directly), no
`numactl`, THP `never`. End-to-end runs were impossible (model 9.9 GB >>
RAM); all harnesses used real weight bytes + real kernels and adopted the
parent's 215.2 ms frame for tok/s math.

## Finished reports

- `hw_agent1_kernel/REPORT.md` (agent) — CPU-specific expert kernel: KILL.
  Five exact-output AVX2 variants 0.27-0.90x vs shipped ggml IQ2 kernels.
  Expert GEMV is decode/uop-bound (~90% of block insns are IQ2 decode).
  Guardrail: pthread-create-per-GEMV inverts scaling; persistent threads only.
- `hw_agent2_topology/REPORT.md` (agent) — Core/cache/thread: KILL all levers,
  KEEP stock `-t 4`. Migration ~0, 4T scaling 3.48-3.50x, oversubscription
  collapses 4-100x, NTA/streaming ceiling <=0.2%, parity 139/139.
- `hw_profile/REPORT.md` (parent-run) — Hardware counters + bottleneck
  verdict: expert path is decode/dependency-latency-bound (IPC 1.08-1.51 of
  5-wide; LLC miss 92% but DRAM only ~3-6 GB/s; dTLB miss 0.006%; branch
  miss 0.07%; 0 migrations/ctx). Dense-region split corroborates the
  parent frame (~49 ms est. vs 47.5 ms). No hardware-only speedup.
- `hw_agent3_memory/REPORT.md` (parent-run) — TLB/staging verdicts. THE
  sprint's one surviving lever: **KEEP dropping `MADV_DONTNEED`** (exact-
  output, one line, ~5x sync-staging management cut, +14 MB bounded).
  KILL huge pages (THP never, zero effect), KILL_HARD Q8 transcode cache
  (12x slower than GEMV, unhidden, outputs DIFFER, 25-51 MB). KEEP 28-64
  slots + compact slot reuse (locality: noslot GEMV 4.89 vs 3.3 slotted).
- `agent2_attn/REPORT.md` (agent) — Attention gate/head reduction: KILL
  everything. Single-head ablation 0.31-0.42 rel err; 16->8 = 67% err;
  gate low-rank flat 6-11%/block; full-attn surface only 12.6 of 47.5 ms.

## Killed stale (no reports; partials salvaged, reusable)

- `agent1_expert/` — router concentration / expert conditionality probes
  (scripts 00-06 + `gguf_inventory.json`). Stalled ~100 min, killed.
- `agent3_funcmoe/` — functional MoE compression (calib assets + top-K/SVD/
  linear-probe/MLP scripts). Stalled ~90 min, killed.
- The two killed hardware agents died waiting on approval-gated tool calls;
  the parent re-ran both workstreams instead (REPORTs above).

## Hardware-sprint synthesis (parent, all four workstreams complete)

- Primary bottleneck diagnosis: expert GEMV is instruction-decode/uop-bound
  on Gracemont (agent-1 disassembly + profiler IPC ~1.1-1.5 of 5-wide), NOT
  bandwidth-bound (~3-6 GB/s vs roof), NOT TLB-bound in steady compute
  (0.006% dTLB miss), NOT scheduler-bound (0 migrations). Staging-path
  fault churn (214 minflt/miss under DONTNEED) is a separate, fixable cost.
- Best hardware-only speedup measured: `noDN` (drop MADV_DONTNEED) — the
  only exact-output lever with a strictly-positive gain; e2e magnitude =
  currently-exposed sync management on the 10 GB box (sync-replay upper
  bounds: ~100-185 ms/token). Everything else: stock already optimal.
- Combined best host configuration: stock llama.cpp `-t 4`, no affinity
  flags, persistent threads, `n_threads <= hardware_concurrency` guardrail,
  noDN staging, 28-64 staging slots.
- Realistic remaining hardware headroom on N100-class: noDN e2e gain (TBD)
  plus ~0% from kernel/topology/TLB levers. Untested (needs bigger box):
  many-core/NUMA paths, batch>=8 IQP path (speculative/MTP-style only).
- Next 3 hardware experiments only: (1) apply noDN, measure e2e on 10 GB
  box; (2) Q2_K-transcode quality gate (format change, ~2x hint); (3) rerun
  staging with a REAL production route trace to size slots for the true
  miss curve.

## Notes for rerun

- Weight blobs and derived binaries (*.pt, *.npy, *.npz, *.o, executables,
  tokenizer.json) are EXCLUDED from git; probe scripts re-fetch/rebuild them
  (see each REPORT's rebuild section and each dir's fetch scripts).
- Open arch questions (agents 1+3 killed): expert conditionality / K
  reduction / width reduction / dispensable branches; functional MoE
  compression (delta rank, dense-FFN distillation). Salvaged assets above.
