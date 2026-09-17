# GATE B result: KILL the 35B single-GGUF path (measured)

Run: `moe35-bench.yml` on GH ubuntu-latest, scalar (no-SIMD) audit-parity build.
llama.cpp @ c32d1dabe. Model: `unsloth/Qwen3.5-35B-A3B-GGUF`,
`Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`, 10,656,955,008 bytes.
Artifacts: `moe35-gateb-bench` (run 34480130283). First attempt (34469260568)
died post-bench with no artifacts; workflow now uploads bench data immediately.

## MEASURED (profiler-exact command: llama-bench -p 512 -n 128 -ngl 0)

- Prompt processing: **1.91 tok/s** (512 tokens)
- Decode: **1.71 tok/s** (128 tokens, exit 0, correct completion)
- Wall clock: **33 min** for the single bench invocation
- Peak RSS (`/usr/bin/time -v`): **10,683,744 KB ≈ 10.4 GB**
- RSS trace (0.5s sampling, 3930 samples): **10.4 GB within ~40s of start**,
  flat for the full 33 min. Not gradual expert faulting — full residency.
- Page faults: 2 major (I/O), 181,731 minor. CPU: 199% (~2 cores).

## Score math (profiler formulae, S_perf=min(tps/15,1)*100)

- S_perf = 1.71/15*100 = **11.4** (current Jamii: 77.53)
- S_eff = max(0, (7-10.43)/7) = **0** (current: 92.72)
- Even at S_acc = 95: total = 47.5 + 3.4 + 0 = **50.9 < 66.04**.
- Worse: 10.4 GB peak exceeds the 8 GB reference laptop → likely **OOM
  disqualification** on real audit hardware, regardless of score.

## WHY (both mechanisms source-proven in llama.cpp, not inferred)

1. **Loader prefetches the whole file.** `llama_model_loader::init_mappings`
   is called with `prefetch=true` unconditionally (`src/llama-model.cpp:1707`);
   `llama_mmap` then maps with `MAP_POPULATE` plus a touch-all-pages thread
   (`src/llama-mmap.cpp:480,500`). RSS = file size ~40s after launch, before
   routing matters. Sparse compute cannot fix loader-level residency, and the
   profiler controls the loader.
2. **Scalar MoE decode is ~11x too slow.** 1.71 vs 15 tok/s cap. Active FLOPs
   (~2 GFLOP/token incl. K=8 experts) look comparable to dense 0.6B, but
   IQ2_XXS scalar dequant + MoE gather + 40-layer GQA + a 248k-vocab output
   head (~1 GFLOP/token for the head alone) dominate. This ratio roughly holds
   across MoE sizes on the scalar path.

## What this kills and what it doesn't

- KILLED: any GGUF whose file size approaches/exceeds the RAM budget, and any
  plan relying on demand-paged expert residency under stock llama.cpp.
- NOT killed by this run: SMALL MoE GGUFs (file ≪ 7 GB) score S_eff > 0, but
  the break-even bar is brutal — e.g. a 3.5 GB / 5 tok/s MoE needs S_acc ≈ 100
  to tie Jamii (50 + 10 + 10 = 70 vs 66). Any smaller-sparse probe must clear
  that arithmetic BEFORE burning runner hours, and must measure accuracy first
  (accuracy is the only lever that can pay for the S_perf/S_eff deficit).

## Process fixes (my bug, owned)

- `llama-cli` now auto-enables conversation mode for chatted models and loops
  on stdin after turn one; the smoke test spun forever printing empty prompts
  (user-caught, run cancelled on request). Fixed with `--single-turn`,
  `</dev/null`, `timeout`, and a line-count tripwire in `moe35-bench.yml`.
