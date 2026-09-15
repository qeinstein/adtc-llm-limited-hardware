# Hardware profiler REPORT (parent-run; agent stalled, no agent verdict delivered)

Method: rebuilt the stalled agent's `kreplay` from current `kreplay.c`
(its recompile died on approval) with a minimal link
(`ggml_mini.c`, new in this dir — see section 6), verified bit-identical
output (`checksum=000e53ec3972d4fa` on all runs), and collected
steady-state counters via the agent's `hwpstat` (perf_event_open wrapper,
HWP_SYNC=1) over the exact-kernel 40-layer x top-8 expert replay with real
weight bytes. All e2e tok/s adopts the parent frame (model 9.9 GB >> 2.7 GB
box RAM; full load impossible).

## 0. Verdict

The dominant expert path is **instruction-decode / dependency-latency-bound**
(mixed with streaming-memory traffic far below the bandwidth roof).
NOT compute-bound, NOT bandwidth-bound, NOT TLB-bound (steady compute),
NOT scheduler-bound. No hardware-only speedup: stock kernels + stock
scheduler are already optimal on this box. The one actionable hardware
finding in the whole sprint is the staging-path `noDN` result — see
`../hw_agent3_memory/REPORT.md` (parent-run).

## 1. The 12-item report

1. **CPU model + ISA + topology**: Intel N100 (Gracemont) under KVM; 4 vCPUs,
   1 thread/core, NO SMT; AVX2+FMA+F16C+AVX-VNNI+GFNI+BMI2, no AVX512
   (`gcc -march=native` = alderlake); L1d 32K + L1i 64K shared per pair
   (0-1, 2-3), L2 2 MB + LLC 6 MB shared 0-3; single NUMA node; 2.7 GB RAM;
   THP `never`, 0 huge pages; `perf`/`numactl` absent (used perf_event_open
   directly); TSC constant 0.807 GHz, guest MHz 806, host boost invisible.
2. **Baseline tok/s**: adopted **4.65** (e2e unmeasurable here).
3. **Baseline ms/token**: adopted **215.2** (experts ~110).
4. **Counter diagnosis** (steady-state, 4T, 2 reps, DRAM-streaming regime;
   two windows agree):
   | event | run A | run B | reading |
   |---|---|---|---|
   | cycles | 3.58G | 2.12G | — |
   | instructions | 3.86G | 2.82G | — |
   | **IPC** | **1.08** | **1.33** | of 5-wide: NOT compute-bound (4-run range 1.08-1.51) |
   | cache_references / misses | 333M / 308M | — | **92.4% LLC miss** |
   | llcref / llcmiss (raw 2e) | — | 182M / 167M | **91.9% LLC miss (independent event agrees)** |
   | L1 loads (d1:01/08) | — | 568M hit / 171M miss | **76.9% L1 hit** (decode tables hit L1) |
   | dTLB loads / misses | 997M / 64K | — | **0.006% miss: TLB healthy in compute** |
   | branch miss rate | 0.07% | — | negligible |
   | page faults (2 reps) | 11 (7 min/4 maj) | — | ~0 steady-state |
   | ctx switches / migrations | 0 / 0 | — | scheduler clean |
   | DRAM traffic (167M LL miss x 64 B) | — | ~10.7 GB / 3.4 s | **~3-6 GB/s << roof** |
   Unavailable in this guest (open OK but 0, or ENXIO): LLC_load(_miss),
   L1 miss, stalled_front/back, uops_retired (vec256 > instructions =
   virtualization garbage), IDQ/resource stalls, CPUID leaf 0x18. SIMD mix
   therefore static only: ~26% vector per agent-1 disassembly (88 of 332
   insns/block; ~90% of block is scalar IQ2 decode).
5. **Exact change**: none (diagnostic). Rebuilt kreplay binary from newer
   source; added `ggml_mini.c` link shim (no kernel/behavior change).
6. **New tok/s**: 4.65 (diagnostic; no change proposed).
7. **New ms/token**: 215.2.
8. **Expert ms/token**: ST DRAM-streaming kernel 146 ms/token measured
   (checksum-verified); production-faithful 4T number is agent-2's
   spin-barrier 63 ms, NOT kreplay's 4T (see harness artifact below).
9. **RAM impact**: zero.
10. **Parity**: all runs `checksum=000e53ec3972d4fa`, `ysum=0.090763`.
11. **KEEP/KILL**: diagnostic — no lever. KEEP stock kernels/scheduler.
12. **Stacks**: the diagnosis (decode-bound, ~1 GB/s eff.) says future gains
    must come from representation/compute-removal (arch sprint), not clocks.

## 2. Harness artifact (do not mistake for a hardware finding)

kreplay's thread pool uses blocking `pthread_barrier`: 4 threads achieve
only **1.06-1.44 CPUs** with **15-30k voluntary ctx switches/run**
(rusage), so kreplay-4T kernel times (143-200 ms) are pessimistic.
Production ggml spins (`ggml_barrier`); agent-2's spin 4T (63 ms) is the
faithful number. Counters above are unaffected (sleep retires no user
insns; task_clock/cycles exclude sleep).

## 3. Dense-region cross-check (cache-hot ST, EVICT=0; LM head skipped —
   its 163 MB fetch is truncated vs 286 MB needed)

Per-token ST: attn_q 9.34, k 0.55, v 0.54, o 4.24, gdn_qkv 28.7,
gdn_gate 13.0, ssm_out 12.2, shared (gate+up+down) ~6.8, router ~2.3
(serial approx). Attn+GDN = 68.6 ms ST-hot; /3.5 (4T eff.) x ~2.5
(streaming penalty) ~= **49 ms vs the parent's 47.5 ms frame. MATCH.**
Expert GEMV takes ~2x the time for ~1/3 the bytes of dense (IQ2 decode
tax) — consistent with the decode-bound verdict.

## 4. Frequency / throttling / noise

Guest sees fixed 806 MHz; TSC constant 0.807 GHz; host boost state is
invisible. Absolute wall times swing 2-5x run-to-run on this VM
(kernel_sum 146-898 ms across windows); ratios, checksums, and counter
rates are the robust signal. minflt/majflt steady-state ~0 (no paging).

## 5. Files

- `kreplay.c` (+`ggml_mini.c`, new) — replay harness + mini link shim.
- `hwpstat.c`, `rawprobe.c`, `rusage_chk.c`, `calib.c`, `t1.c`-`t5.c`,
  `fetch_dense.py`, `fetch_experts.py` — agent's tools (as found).
- `counters_expert_A1.txt`, `run_expert_*.txt`, `routes_tok60.txt` —
  agent's + parent's raw logs (parent counter runs: this REPORT only).

## 6. Rebuild (parent-verified)

```
P=/tmp/llamacpp-pin/ggml; I="-I. -I$P/include -I$P/src -I$P/src/ggml-cpu"
gcc -O3 -march=native -D_GNU_SOURCE $I -c kreplay.c
g++ -O3 -march=native $I -c $P/src/ggml-cpu/vec.cpp
gcc -O3 -march=native -D_GNU_SOURCE $I -c $P/src/ggml-quants.c
gcc -O3 -march=native -D_GNU_SOURCE $I -c ggml_mini.c
gcc -O3 -march=native -Dquantize_row_q8_K_ref=shim_unused_q8k_ref \
  -c ../hw_agent1_kernel/ggml_tables_shim.c -o shim.o
g++ -O3 -march=native -o kreplay kreplay.o ggml-quants.o ggml_mini.o \
  vec.o shim.o /tmp/hw1/x86_quants.o -lm -lpthread
gcc -O2 -o hwpstat hwpstat.c
```

`ggml_mini.c` implements exactly: `ggml_cpu_init` (noop: no SILU/GELU
tables on this path), `ggml_abort`, `ggml_row_size` (Q8/Q4/Q5/Q6_K sizes
verified against `ggml-common.h` static_asserts AND the dense byte table),
`ggml_quantize_init` (IQ2 init only), `ggml_type_size/name` (link-only).
Parity gate: `checksum=000e53ec3972d4fa`.
