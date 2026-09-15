# AGENT 2 — Core/cache/thread topology: REPORT (KILL all levers, KEEP stock -t 4)

## 0. Verdict

Nothing beats stock `llama.cpp -t 4` (defaults, no affinity flags) stably.
Thread count = logical CPUs is confirmed optimal (3.3-3.7x over ST, ~90%
efficiency); pinning, expert-affine partitioning, cluster placement,
non-temporal streaming hints, and oversubscription are all KILLED with
numbers. No RAM impact. Parity PASS (139/139 runs bitwise identical).
No code change proposed. Nothing to stack.

## 1. The 12-item report

1. **CPU model + ISA + topology**: Intel N100 (Gracemont E-cores) under KVM;
   4 vCPUs, **1 thread/core — NO SMT** (thread_siblings = 0/1/2/3 singly);
   ISA: AVX2+FMA+F16C+AVX-VNNI+GFNI+BMI2, no AVX512 (`lscpu` flags read);
   L1d 32K + L1i 64K shared per pair (cpus 0-1, cpus 2-3), L2 2 MB shared
   0-3, L3 6 MB shared 0-3 (sysfs cache topology read); single NUMA node;
   2.7 GB RAM; `taskset` present, **`numactl` and `perf`/PMU absent**.
2. **Baseline tok/s**: end-to-end UNMEASURABLE on this box (model 9.9 GB >>
   2.7 GB RAM — full load impossible, same constraint as agent 1). Adopted
   parent frame: **4.65 tok/s**. Harness-level ST baseline (this box):
   5.453 ms/layer median-of-medians (3 windows x 30 trials).
3. **Baseline ms/token**: adopted **215.2**. Harness: ST 218.1 ms/40-layers;
   stock-4T 63.1 ms/40-layers (this box is faster-clocked than Kaggle).
4. **Counter/scheduler diagnosis** (no PMU on KVM — `/proc` + `sched_getcpu`
   proxies, read every run): thread **migration ≈ 0** (0-11 transitions per
   90 trials even unpinned — Linux keeps spinning threads home); blocking
   barriers cost ~30-40 **voluntary** ctxt switches/trial vs **0** with
   spin barriers; **involuntary** ctxt spikes (up to 1462/run) coincide only
   with host-preemption storms (p90 blowouts); per-worker busy-time balance
   typically 1.03-1.3, ~2.7 worst in stall windows, 5.5 oversubscribed.
  Conclusion: the
   scheduler is not the bottleneck; sync *mechanism* is.
5. **Exact winning change**: `-t 4` (= `hardware_concurrency`), **no**
   affinity mask, **no** `--cpu-mask`/`--cpu-strict`/`--poll` flags —
   i.e. stock defaults. Outer `taskset 0xf` ≡ default. Nothing to set.
6. **New tok/s**: 4.65 (same as baseline — nothing beats stock stably).
7. **New ms/token**: 215.2 (same).
8. **Expert ms/token**: **59.5-63.1 ms/40-layers measured** (4T spin,
   gate+up+down+SiLU+quant for K=8/layer, real bytes+kernels, DRAM-streaming
   regime); ST 218.1 ms. Kaggle frame says 110 ms — the gap is box clock
   (N100 boost vs Xeon 2.2 GHz), not a win. Cross-check: +10% vs agent-1's
   same-window ST number after workload normalization. MATCH.
9. **RAM impact**: ZERO. Peak RSS 107.3 MB (1T) vs 107.5 MB (8T) — thread
   count/affinity/partition change nothing (VmHWM read per config).
10. **Parity**: PASS. 139/139 runs, maxdiff 0.0 / nbad 0 (bitwise
    identical): 108/108 stream-regime (96 main + 12 fill) + 28/28
    smoke/calibration + 3/3 hot-regime vs hot ref — across ALL thread
    counts (1-8), partitions, affinities, taskset masks, pool regimes,
    and barrier types. Scheduling never changes outputs.
11. **KEEP/KILL**: KEEP stock `-t 4` (confirmed optimum). KILL pinning,
    expert-affine partitioning, NTA/streaming hints, oversubscription,
    L1-cluster placement, per-op thread counts (unsupported by runtime).
12. **Stacks with architecture changes? NO** — nothing to stack. One
    guardrail for any future executor: keep workers <= logical CPUs and
    keep spin (not blocking) sync; anything else is for a many-core/NUMA
    box, not this one.

## 2. Method (reproducible)

- Workload: one full MoE layer per trial — K=8 experts x (gate 512x2048
  IQ2_XXS + up 512x2048 IQ2_XXS + SiLU/mul + down 2048x512 IQ2_S), REAL
  weight bytes (`/tmp/agent1_raw` L00 E000-E007, unsloth
  Qwen3.5-35B-A3B-UD-IQ2_XXS) + REAL `ggml_vec_dot_{iq2_xxs,iq2_s}_q8_K`
  from pinned llama.cpp 3057bb66 (`/tmp/hw1/*.o`), `-O3 -march=native`,
  FTZ/DAZ, 3x32 MB streaming pools (decode-like DRAM regime).
- Harness `topo_sweep.c`: persistent pthreads + barriers, ggml style.
  Arms: nth 1-8; partition row-split (ggml-like, 40 barrier rounds/trial)
  vs expert-split (1 barrier/trial); affinity free vs `pthread pin`
  (respects outer taskset, like `--cpu-strict`); pool stream vs hot;
  barrier pthread/futex vs sense-reversing spin (ggml-`ggml_barrier`
  analog); outer `taskset` 0xf/0x3/0x5/0x1.
- Stability: 35 configs x 3 rotated windows x 30 trials; report
  median-of-medians ("best STABLE", immune to the 15 ms VM-stall outlier
  in `4 expert free spin` w2 and the w1 fill-in host storm).
- Runtime grounding (bodies read, cited below): intra-graph node sync in
  stock ggml ALWAYS spins (`ggml_barrier`, ggml-cpu.c:576-607), so the
  spin arm is the production-faithful one; `--poll` only gates graph-
  kickoff waits (ggml-cpu.c:3208-3240). Affinity: `-C/--cpu-mask`,
  `--cpu-range`, `--cpu-strict` (arg.cpp:1534-1560, ggml-cpu.c:2715+).
  Thread counts: ONLY `-t` (decode) and `-tb` (prompt/batch) exist —
  NO per-op expert/attn/LM-head knobs, so that question is answered by
  the partition experiment instead (closest analog) + this note.
- LLC probe `llc_pollute.c`: hot AVX2-reduction victim (32K/512K/4M)
  around a 32 MB real-kernel expert stream, +/- NTA prefetch, 30 trials.

## 3. Full sweep table (median-of-medians, ms/layer; lower better)

mask nth part aff pool bar | wins | med | per-window medians | worstCV | nbad
---|---|---|---|---|---|---
0xf 4 row free hot spin | 3 | 1.470 | 1.01,1.69,1.47 | 1.75 | 0*
0xf 4 expert pin stream spin | 3 | **1.487** | 1.01,1.67,1.49 | 1.69 | 0
0xf 4 expert free hot spin | 3 | 1.500 | 1.01,1.57,1.50 | 1.68 | 0*
0xf 4 expert free stream spin | 3 | 1.551 | 0.99,15.23!,1.55 | 1.06 | 0
0xf 4 row free stream spin | 3 | **1.578** | 1.06,1.65,1.58 | 2.31 | 0
0xf 8 expert pin stream futex | 3 | 1.709 | 1.07,1.71,1.85 | 1.22 | 0
0xf 4 row pin stream spin | 3 | 1.716 | 1.05,1.92,1.72 | 1.93 | 0
0xf 4 expert pin stream futex | 3 | 1.810 | 1.12,2.37,1.81 | 1.33 | 0
0xf 6 expert free stream futex | 3 | 1.903 | 1.44,2.42,1.90 | 0.84 | 0
0xf 4 expert free stream futex | 3 | 2.023 | 2.02,1.50,3.13 | 0.84 | 0
0xf 8 expert free stream futex | 3 | 2.053 | 1.44,2.05,3.91 | 1.26 | 0
0xf 3 expert pin stream futex | 3 | 2.210 | 2.14,2.21,2.61 | 0.66 | 0
0xf 3 row free stream futex | 3 | 2.231 | 1.58,2.23,2.63 | 1.94 | 0
0xf 4 row pin stream futex | 3 | 2.385 | 1.43,2.76,2.38 | 1.47 | 0
0xf 3 row pin stream futex | 3 | 2.476 | 1.69,2.48,2.55 | 1.65 | 0
0xf 3 expert free stream futex | 3 | 2.487 | 2.38,2.49,3.07 | 1.23 | 0
0x3 2 row free stream spin | 3 | 2.534 | 1.96,3.24,2.53 | 1.09 | 0
0xf 6 expert pin stream futex | 3 | 2.695 | 1.60,3.00,2.69 | 1.52 | 0
0x3 2 expert pin stream spin | 3 | 2.729 | 1.81,3.07,2.73 | 0.72 | 0
0xf 2 expert free stream futex | 3 | 2.790 | 1.81,2.79,2.95 | 0.09 | 0
0x3 2 row pin stream spin | 3 | 2.800 | 1.82,3.20,2.80 | 1.40 | 0
0x5 2 expert pin stream spin | 3 | 2.827 | 1.98,3.06,2.83 | 1.10 | 0
0x5 2 row pin stream spin | 3 | 2.828 | 1.94,3.06,2.83 | 0.93 | 0
0xf 2 expert pin stream futex | 3 | 2.849 | 1.79,2.85,3.05 | 0.25 | 0
0xf 8 row pin stream futex | 3 | 3.004 | 3.60,3.00,2.89 | 1.44 | 0
0xf 6 row free stream futex | 3 | 3.020 | 2.81,4.30,3.02 | 1.66 | 0
0xf 2 row free stream futex | 3 | 3.049 | 2.01,3.05,3.37 | 0.33 | 0
0xf 8 row free stream futex | 3 | 3.287 | 2.27,3.29,4.30 | 1.22 | 0
0xf 4 row free stream futex | 3 | 3.308 | 2.52,3.31,3.39 | 0.69 | 0
0xf 6 row pin stream futex | 3 | 3.355 | 3.35,3.70,3.18 | 1.33 | 0
0xf 2 row pin stream futex | 3 | 3.383 | 3.38,2.90,3.42 | 1.33 | 0
0x1 1 row pin stream spin | 3 | 5.175 | 3.51,5.80,5.18 | 0.87 | 0
0xf 1 row free stream futex | 3 | 5.453 | 3.34,5.56,5.45 | 0.31 | 0
0xf 1 row free hot spin | 3 | 5.462 | 3.38,5.79,5.46 | 0.10 | 0*
0xf 8 expert pin stream spin | 3 | 6.219 | 6.22,6.99,6.00 | 0.68 | 0

`*` hot arms use different expert bytes by construction (copies of other
experts), so they mismatch the stream ref EXPECTEDLY; re-verified nbad=0
vs a dedicated hot ref (3/3 runs, `hot_verify.csv`). All 96 main-sweep
stream runs nbad=0 vs stream ref.
`!` = VM stall outlier absorbed by median-of-medians.

Fill-ins (clean window w2; w1 hit a host-preemption storm, kept in CSV):
2T row spin 2.45, 2T expert spin 2.54, 3T row spin 1.88, 3T expert spin
2.09, 6T expert spin 10.7 (bad), 6T row spin 180 (catastrophic, 1303
involuntary ctxt). All nbad=0.

Scaling (production arm, spin; single clean window `scale_calib.log`,
30 trials each, all nbad=0): 1T 3.599 -> 2T 1.849-1.902 (1.89-1.95x,
95-97% eff.) -> 3T 1.291-1.412 (2.55-2.79x, 85-93%) -> 4T 1.029-1.034
(3.48-3.50x, 87%). Multi-window med-of-med agrees: 4T 3.3-3.7x (w1:
3.34/1.01; w3: 5.45/1.49). Row vs expert in the SAME window: 2T 3%,
3T 9% (row ahead), 4T 0.5% — no stable direction => partition is noise.
6T/8T spin: 4-100x COLLAPSE (6.2-180 ms, worse than ST).

## 4. Kill notes (each falsified with numbers)

- **Pinning** (`pin` ~= `--cpu-mask 0xF --cpu-strict 1`): spin 4T row free
  1.578 vs pin 1.716 (pin 9% WORSE); expert free 1.551 vs pin 1.487 (pin
  4% better); 3T/2T mixed both ways. No stable direction => noise.
  Migrations ~0 either way. Matches phase1 Kaggle (+1.7% ~= noise). KILL.
- **Expert-affine partition** (not expressible in stock runtime — would
  need a scheduler change): 4T spin expert 2-13% ahead, but 2T/3T clean-
  window row AHEAD 4-11%. No stable win under spin; only wins big with
  blocking sync (futex 4T: 40%), which production doesn't use. KILL.
- **Oversubscription** (SMT analog; box has no SMT): 6T spin 10.7-180 ms,
  8T spin 6.2 ms vs 4T 1.5 ms. Catastrophic with spin sync; merely
  pointless with futex. `-t` must equal logical CPUs. KILL (guardrail).
- **L1-cluster placement** (0x3 same-pair vs 0x5 split-pair, 2T): 2.73-2.83
  both, identical. KILL.
- **NTA / streaming hints** (llc probe): 32 MB expert stream slows hot
  re-access +51% (32K), +17% (512K), +12% (4M) — but absolute penalty is
  1.7/8.5/56 us vs 25-31 ms stream (<=0.2%). NTA prefetch ahead of the
  kernel: NO improvement (kernel 27.4/25.9/28.0 ms vs kernel+nta
  29.7/26.5/26.8 and 29.3/31.4/25.4 ms — wash to worse); memcpy+nta
  saves only ~2-9% of re-access cost (~16-24% of the already-tiny
penalty). True fix (movntdqa
  kernel rewrite) has a <=0.2% ceiling. KILL.
- **hot vs stream weights**: ST 5.462 vs 5.453 (identical); 4T within 7%.
  Decode/uop-bound (stream = 1.1-1.3 GB/s eff., matches agent 1) => cache
  residency of weights doesn't matter. KILL.
- **Per-op thread counts** (experts vs attn vs LM head): NOT supported by
  stock runtime — single `-t` for the decode graph, `-tb` is prompt/batch
  only (arg.cpp:1515-1528). No measurement possible; partition experiment
  is the analog (KILL above).
- **`--poll`**: only gates graph-kickoff waits; intra-graph node sync
  always spins (ggml_barrier). Expect <=2% per phase1. Not re-tested e2e
  (can't load model); no claim beyond phase1's numbers.

## 5. Files & rebuild

- `topo_sweep.c` — sweep harness (build: `gcc -O3 -march=native -o
  topo_sweep topo_sweep.c /tmp/hw1/x86_quants.o /tmp/hw1/shim.o -lm
  -lpthread`; needs `/tmp/agent1_raw/L00_E00{0..7}_{gate,up,down}.*`).
- `llc_pollute.c` — LLC probe (same link line).
- `run_sweep.sh` — 35 configs x W windows x T trials driver (used 3x30).
- `ref/layer_ref.{bin,u64}`, `ref/layer_hot.{bin,u64}` — parity refs.
- `results/sweep_*.csv/.log`, `results/sweep_fill_*.csv`,
  `results/llc_*.csv` — raw data (105 + 12 + 18 rows).
- `results/scale_calib.log` — single-window 1T/2T/3T/4T scaling (7 runs).
- `results/hot_verify.csv/.log` — hot-regime parity re-verification (3/3).
- Binaries `topo_sweep`, `llc_pollute` kept for re-runs.

## 6. Suggested follow-ups (for parent/other agents)

1. Do NOT spend more on topology for 4-core/N100-class targets; the
   stock `-t ncpu` + spinning intra-graph sync is already optimal.
2. If a custom MoE executor is ever built: expert-granular work items +
   spin sync (fewer sync points, robust under noise) — but budget <=5%,
   not a lever, and it is a scheduler change needing its own parity gate.
3. Add a run-script guardrail: `n_threads = min(requested,
   hardware_concurrency)` — oversubscribing spinners collapses 4-100x
   (measured), worse than single-threaded.
4. Revisit ONLY on many-core/NUMA hardware (ggml NUMA
   distribute/isolate/numactl paths + `--cpu-mask` per-node split are
   untested here — single node, no numactl). Needs the 10 GB+ box anyway
   for e2e (`-t/-tb/--cpu-mask/--poll` matrix, cf. phase1 on Kaggle).
5. `numactl`/interleave and THP/huge-pages for the expert stream were out
   of scope (no NUMA here); agent 1's TLB note stands as second-order.
