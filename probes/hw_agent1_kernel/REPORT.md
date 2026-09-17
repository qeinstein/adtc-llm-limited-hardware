# AGENT 1 — CPU-specific expert-kernel specialization: REPORT (KILL)

## Verdict: KILL. ggml's AVX2 IQ2 kernels are already at the Gracemont optimum.

Five exact-output specialization directions tested head-to-head against the real
kernels on exact shapes with real weight bytes. All slower: 0.27x-0.90x.
No code change proposed. No RAM impact. Nothing to stack.

## 1. Method (reproducible)

- Kernels: REAL `ggml_vec_dot_{iq2_xxs,iq2_s,q5_K,q6_K}_q8_K` + `quantize_row_q8_K`
  from pinned llama.cpp 3057bb66 (`/tmp/llamacpp-pin`), file
  `ggml/src/ggml-cpu/arch/x86/quants.c`, compiled `-O3 -march=native`
  (alderlake: AVX2+FMA+F16C+AVX-VNNI+GFNI+BMI2, no AVX512) + tables shim.
- Dispatch check: `ggml_cpu_iqp_supports_mul_mat_id` requires batch>=8
  (`GGML_IQP_MIN_BATCH_ID`), so single-token decode uses these classic vec-dots.
- Weights: REAL bytes range-fetched from pinned
  `unsloth/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf` (L00 routed gate/up `IQ2_XXS`
  512x2048, down `IQ2_S` 2048x512; shared gate/up `Q5_K`, down `Q6_K`).
- Regime: 32-64 MB streaming pools (DRAM, decode-like), activation hot,
  FTZ/DAZ, pinned cores, interleaved A/B trials, medians.
- Build: `gcc -O3 -march=native` per file; see commands in section 5.
  Objects live in `/tmp/hw1` (scratch); sources + this report are the deliverable.

## 2. Falsification results (gate GEMV 512x2048, IQ2_XXS; ratios stable across 3 windows)

| variant | idea | parity (512 rows) | speedup vs baseline |
|---|---|---|---|
| v1 | 2-row interleave, stock decode | bit-exact 512/512, maxdiff 0.0 | 0.84-0.90x |
| v2 | GFNI in-register signs (verified vs LUT 128/128) | bit-exact | 0.42x |
| v2b | and/cmpeq signs, no GFNI (isolate) | bit-exact | 0.40-0.41x |
| v3b | vpgatherqq grid + stock signs (isolate) | bit-exact | 0.43x |
| v6 | pair-LUT grid x sign (256 KB) | — | KILLED analytically: maddubs is u8xi8, cannot consume signed mags; mask-only variant = same loads as baseline + L2 latency |

Why baseline wins (from disassembly of the shipped kernel):
- gcc lowers `_mm256_set_epi64x` to `vmovq`/`vpinsrq` with MEMORY operands +
  `vinserti128`: L1->lane directly, zero store-forwarding. Already optimal.
- Per 256-value block: 332 insns = ~200 scalar idx/address + 88 vector decode
  + 25 MAC + 27 data loads. ~90% of instructions are decode, not MAC.
- GFNI/isolation: parity-unpack scalar chain (24+ dependent ops/group) dwarfs
  the 8 loads + 3 combines it replaces. Gather ~40-50c each on Gracemont.
- Inner loop fully unrolled x4, zero stack spills, BMI2 shrx already used (S kernel).

Also checked, no lever: Q5_K/Q6_K can't use VNNI (per-position scales defeat
vpdpbusd: equal uop count); IQ2_S asm equally clean (L1+pinsrq, vector signs/scales).

## 3. Expert-region accounting (this box, phase10a-consistent window)

ST medians, pool-streaming: gate 535.0, up 525.8, down(IQ2_S) 151.9 ns/row;
shared Q5_K 492.8/456.4, Q6_K 120.5 ns/row; q8_K quant x+h 13.1 us.
Cross-check: gate 535 vs phase10a 583 (-8%); down 152 vs 142.6 (+7%). MATCH.

- ST experts-only x40 layers: 302.6 ms/token (no dispatch).
- At measured 4-thread efficiency 3.48x (persistent threads): 87.0 ms/token
  = EXACTLY phase5a's independently measured 87 ms expert core. Closed.
- Parent frame (~110 ms expert region) = 87 core + dispatch/quant/SiLU/router.
- Effective BW: XXS ~1.0, IQ2_S ~1.1, Q5/Q6 ~3 GB/s << DRAM => decode/uop-bound.
- Est IPC ~2-2.5 of 5-wide (shuffle-port + L1-latency bound, static mix based;
  no PMU on this KVM box). No `perf` available; honest proxy only.
- Cycles/output: gate 535 ns/row; TSC-cycles 344/row (TSC=806 MHz const);
  core-cycles freq-dependent (~1070 @2.0 GHz probe; VM hides cpufreq).
- MT warning: pthread_create-per-GEMV INVERTS scaling (measured 1T>2T>4T);
  persistent threads required (llama.cpp threadpool OK, verify n_threads=4).

## 4. End-to-end

Cannot measure e2e here (2 GB box RAM << 9.9 GB model; full load impossible).
Baseline frame adopted from parent: 4.65 tok/s, 215.2 ms/token, experts ~110 ms.
KILL => new = baseline. No accuracy/output change (no change at all).

## 5. Files & rebuild

- `bench_base.c` — first ST harness (cold + hot) + ref outputs.
- `ggml_tables_shim.c` — fp16 tables + exact `quantize_row_q8_K_ref` copy.
- `check_signs.c`, `keven.inc`, `keven_table.c` — GFNI sign verification + LUT copy.
- `spec_xxs_v1.c`, `spec_xxs_v23.c`, `spec_xxs_v2b3b.c` — variant kernels.
- `spec_xxs_v6_NOTE.c` — v6 kill record.
- `shoot_xxs.c`, `shoot_xxs_all.c` — parity + interleaved shootouts.
- `bench_final.c` — ST full-region accounting. `bench_mt.c` — MT scaling + freq probe.
- Rebuild: compile `x86/quants.c` from the pin with
  `-O3 -march=native -I ggml/src -I ggml/src/ggml-cpu -I ggml/include`,
  then each probe file + `shim.o [+ spec_*.o + keven.o] -lm [-lpthread]`.

## 6. Suggested follow-ups (for parent/other agents)

1. Do NOT spend more on IQ2 row-dot ISA tricks for Gracemont/N100.
2. Representation change (e.g. Q2_K transcode, ~2x per phase10a) needs its
   quality gate — different agent, format change, out of my scope.
3. Batch>=8 unlocks ggml's IQP path; only reachable via speculative/MTP-style
   batching, not single-token decode. MTP sidecar exists (+1.16 GB RAM).
4. Re-measure 4T efficiency on quiet bare metal (this VM is noisy: absolutes
   swing ~2x run-to-run; ratios are the robust signal). /4-ideal overstates
   throughput ~15% (measured 3.48x here).
5. Deployment-level: huge pages / THP for the 280 MB/token expert stream
   (TLB pressure in pool regime is real but second-order).
