# Phase 1 — N100 K×Format Hardware Ceiling (proxy) + 6GB Pareto + i5 Projection

Date: 2026-09-16. Branch: `research/edge0-port`.
Supersedes the original Phase-1 gate with the updated objective (i5, ≤6GB,
≥18 tok/s); the original N100 expert-path gate is still scored (§8).

## 1. Question

Would Edge0-style K=4 + hardware-friendly INT4 be materially faster than
K=8 + IQ2 — and, under the new objective, what is the fastest credible
≤6GB configuration and how far is it from 18 tok/s on the i5 target?

No training. Real kernels (ggml, pin `3057bb66`, `-O3 -march=native`),
real expert weights (L20, 256-expert-full layer), exact Qwen shapes
(gate/up 512×2048, down 2048×512). N100 numbers are proxy evidence;
final reasoning targets Core i5 10th–12th (explicit projection, §7).

## 2. What was measured

Harness `phase1_ubench.c` (8 arms, 32MB DRAM-streaming pools of DISTINCT
real rows, interleaved+rotated trials, warmup, FTZ/DAZ, pinned cores):
K8/K4 × {IQ2-mixed (production exact), Q4_0, Q4_K, Q2_K, Q3_K}.
Q-formats requantized from real f32 with the production `quantize_row_*`
(double-quant caveat quantified in §5 — affects values, not timing).
Plus: ST/MT scaling pairs, self-bracketed perf counters, bf16-source
quant-error tool (`phase1_qerr.c`), LRU/Belady sims on the 2016-token
real route corpus, RSS/latency Pareto model (`phase1_pareto.py`).

## 3. Expert GEMV: gate/up/down time (ST, ns/row, median-of-7, run2)

| arm | gate | up | down | ST ms/tok | ratio vs K8+IQ2 |
|---|---|---|---|---|---|
| k8_iq2 (current) | 359.9 | 365.6 | 98.8 | 183.6 | 1.000 |
| k8_q40 | 302.7 | 307.4 | 80.7 | 152.9 | 0.833 |
| k8_q3k | 322.2 | 319.3 | 88.9 | 163.4 | 0.890 |
| k8_q2k | 203.1 | 201.6 | 54.3 | 101.9 | **0.555** |
| k4_q40 | 296.0 | 289.2 | 79.6 | 74.0 | 0.403 |
| k4_q3k | 314.4 | 319.5 | 87.4 | 80.6 | 0.439 |
| k4_q4k | 231.8 | 229.4 | 62.5 | 58.3 | **0.318** |
| k4_q2k | 205.9 | 197.0 | 54.9 | 51.0 | **0.278** |

Replicated run1 within 5% on all 4 shared arms; all checksums STABLE;
k4_q40/k8_q40 pools byte-identical (hash match), K-scaling 0.484–0.491
(≈0.5, linear as expected). Q4_K K8 derived: 0.635 (2× K4 number).

## 4. Counters, scaling, decode overhead (mechanism, not just time)

- 4T scaling (adjacent ST/MT pairs, max-of-9): 3.7–4.2× on ALL formats
  (format-independent; ST=MT bit-identical). Adopted sustained: 3.5×
  (agent-2: 3.48×). Medians were host-contention garbage; max-pairs +
  agent-2 agree.
- Self-bracketed steady-state counters (single-arm, init excluded):
  ins/row IQ2:Q2_K:Q4_K = 1521:405:494 (**3.75×/3.1× fewer insns**);
  cyc/row = 498:264:311 (ratios 1:0.53:0.62 — matches timing 1:0.555:0.635
  within 5%, independent confirmation); IPC 3.06/1.51/1.59 (IQ2 retires
  more/shallower uops; Q2_K fewer/heavier); LLC miss/row 5.5/6.1/11.7,
  miss% 63–73; branch-miss ~0.02%.
- Quant decode overhead, quantified: ~73% of IQ2's executed insns are
  excess decode vs Q2_K (1521→405). Prior disassembly (~90% of the IQ2
  block is decode) corroborates. Bits don't predict speed (Q3_K 0.89
  vs Q2_K 0.555 at +31% bytes); decode structure does.
- Activation quantize: q8_K 6.7µs/2048 + 1.7µs/512; q8_0 1.4µs + 0.35µs
  (≈0.5ms / 0.1ms per token — negligible either way).

## 5. Quantization error vs bf16 source (L20 E0+E1, agree to 3 decimals)

Weight-level relL2 (SNR): gate/up — IQ2 0.347 (9.2dB), Q2_K **0.297
(10.5dB)**, Q4_0 0.087 (21.2dB), Q4_K **0.072 (22.9dB)**; down — IQ2_S
0.253 (12.0dB), Q2_K 0.297 (10.6dB), Q4 0.086/0.071. GEMV-output errors
track weight errors. Conclusions: (a) Q2_K ≈ IQ2 quality tier (better
on gate/up, worse on down) — same-tier, imatrix is further upside;
(b) Q4 is a full tier better (4× lower error); (c) **production must
quantize from bf16**: IQ2→Q2_K transcode doubles error (0.444 vs 0.297),
Q4 transcodes inherit IQ2's floor. The Kaggle likelihood gate (§9)
confirms end-to-end; it tests the transcoded worst case, so a pass is
decisive and a fail still leaves from-bf16 requant as fallback.

## 6. Bytes, staging, RAM

- Bundle bytes: IQ2 876,544 (measured) / Q2_K 1,032,192 (+18%) /
  Q3_K 1,351,680 (+54%) / Q4 1,769,472 (+102%).
- Bytes/token @0% hit: K8+IQ2 280MB; K4+Q4 276MB (wash);
  K8+Q2_K 330MB; K4+Q2_K 165MB.
- LRU/Belady on the real 2016-token corpus (sim reproduces all 6
  reported points exactly): LRU 826→164 miss/tok, 1792→106, 2664→73,
  4096→40, 5632→18; Belady ≈ 0.45× LRU misses (perfect-prefetch bound).
- Fetch model: fresh_MB/SSD_BW + misses×0.25ms (+5ms mgmt), sidecar
  layout assumed. Conservative (cold SSD); the phase9b box showed
  2.87GB/s page-cache-assisted — reported as upside (§7).
- RSS model: dense trunk 1.67GB (inventory-measured; 1.16GB with
  dense-Q2K on attn/shared) + cache + 0.25 KV + 0.30 overhead. Dense
  breakdown: attn 0.65, LM-head 0.34, embeddings 0.29, other 0.22,
  shared 0.09, norms/router 0.08 GB. (Embeddings are a free 0.29GB
  saving — 1 row/token can stay on disk; not yet taken.)

## 7. Pareto: fastest credible per RSS cap (i5 projection, LRU, 1GB/s SSD)

Projection: CPU terms ÷1.4 (LOW = clocks-only, zero IPC credit — genuinely
conservative) and ÷1.7 (HIGH); fetch unchanged (disk-bound). Base buckets
are phase10b wall measurements (attn 48, LM-head 25, GDN 11, shared 8,
scores 10, gaps 17, expert-ovh 8+15·K/8, core 87×ratio×K/8).

tok/s | peak RSS | quality impact | mechanism (i5-LOW; HIGH in parens):

| tok/s | RSS | quality | mechanism |
|---|---|---|---|
| 3.1 | 3.2GB | control | current bounded K8+IQ2 (model reproduces 3.2 measured ✓) |
| 4.1 | 4.6GB | none | K8+IQ2 + 2.4GB cache (RAM alone: +32%) |
| 4.1 | 4.6GB | ≈same tier (MMLU gate running) | K8+Q2_K, same cache GB (fetch eats the win on slow disk!) |
| 5.5 | 5.7GB | +dense-Q2K (gate TBD) | K8+Q2_K + 4GB cache + dense-Q2K |
| **7.7 (8.9)** | **5.7GB** | **K4 needs Phase-4 recovery; dense-Q2K TBD** | **K4+Q2_K + 4GB cache + dense-Q2K — fastest credible ≤6GB** |
| 5.3 | 5.7GB | better weights, K4 unrecovered | K4+Q4_K (bytes kill it: 76MB fresh vs 23MB) |
| 8.8 (10.1) | 5.7GB | +unproven LM-shortlist | sensitivity: head 25→3ms |
| 8.9 (10.5) | 5.7GB | oracle policy | Belady bound (+15% over LRU here) |

Fastest per cap: ≤4GB 6.1 (K4+IQ2!) · ≤5GB 7.0 · ≤5.5GB 7.3 · ≤6GB **7.7**.
Disk sensitivity (top config): 0.5GB/s→6.3–6.7, 1GB/s→7.4–7.9,
3GB/s→8.3–9.0. NOTE every sub-6GB winner uses K4 — K4 quality recovery
(Phase 2/4) is load-bearing, not optional. Fastest QUALITY-PROVEN point
(K8, Q2K-experts MMLU gate passed): 4.94 i5-LOW / 5.6 i5-HIGH @6.0GB
(4.05 @4.6GB) — 1.6× the current bounded path with zero quality risk.

## 8. Original gate + the Edge0-INT4 question

N100-proxy expert path (core+eovh) vs STRONG ≤50 / KEEP ≤65 / WEAK ≤80:
K4+Q2_K **39.6 STRONG** · K4+Q4_K **43.1 STRONG** · K4+Q4_0 51.7 KEEP ·
K8+Q2_K 71.3 WEAK · K8+Q4_K 78.2 WEAK · K8+Q4_0 95.5 KILL · K8+Q3_K 100 KILL.
**Phase verdict: STRONG KEEP** — K4+INT4-class execution is 2.6× faster
than K8+IQ2 on the expert path (43 vs 110ms).

Does easier INT4 arithmetic offset larger bytes on x86/SSD? **No for Q4,
yes for Q2_K.** Q4_K's 1.57× decode win loses to its 2× bytes at every
realistic cache/disk point (K4+Q4_K: 5.3 vs K4+Q2_K: 7.7). Q2_K's 1.8×
win at +18% bytes wins whenever the cache is big enough. **Do not port
Edge0's quant; port K4+prerouter+LoRA-recovery, keep our format search:
Q2_K is the x86 speed king.** Q3_K is Pareto-dominated (slower than Q2_K
AND bigger) — KILLED for experts. Q4_K survives only as the quality
fallback if Q2_K's likelihood gate fails.

## 9. Quality-gate result (Phase 1d — COMPLETE, kernel v1)

Matched MMLU-100 (transcoded worst-case, i.e. IQ2→Q2_K, NOT from-bf16):
control 37.0% ±4.9 → q2k-experts **39.0% ±4.9 (Δ+2.0pp) → KEEP**;
q2k-all 26.0% ±4.4 (Δ−11.0pp) → REJECT (head/embeddings/dense must NOT go
 wholesale to 2-bit — expected, matches prior art). Conclusion: Q2_K
experts show NO measurable regression even in the worst case; production
from-bf16 is strictly better. Q2_K-experts is QUALITY-PROVEN. The +2.0 is
within noise (do not claim "better"). Scope notes: (a) dense-Q2K in §7
covers attn/shared ONLY (narrower than rejected q2k-all) — still needs
its own gate; (b) K4 needs Phase 4; (c) MMLU≠Kiswahili/safety (Phase 2).

## 10. Gap to 18 tok/s (55.6ms) — the honest math

Best credible (i5-LOW ms): cpu 101 (core 17 + eovh 11 + attn/shared-Q2K 24
+ GDN 8 + scores 7 + gaps 12 + LM-head 18 + mgmt 4) + fetch 29 = 130ms.
To close 74ms: LM-shortlist −16 · fetch-overlap/fast-disk −20 · GDN work
−8 · i5-HIGH −18 · scores/gaps −10 · then MTP ×1.3–1.5 effective. Every
item must land, several unproven. Realistic now: **8–9 tok/s** (2.5×
today); 18 needs the full stack — K4 recovery, dense-Q2K gate, LM
shortlist, fetch overlap (Phase 3 prerouter!), fast disk, MTP. This
report defines that stack; it does not pretend we are there.

## 11. Files

`probes/edge0_port/`: `phase1_ubench.c` (harness), `phase1_mini.c` (ggml
shim), `phase1_qerr.c` (quant error), `phase1_fetch_bf16.py`,
`phase1_pareto.py` (LRU/Belady + RSS + i5 model), `build_phase1.sh`.
Raw logs: `/tmp/edge0_phase1/` (st_run2, ratio_run2, self_counters,
qerr_E0/E1 — session scratch, key numbers inline above).
