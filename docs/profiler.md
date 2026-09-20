# Profiler contract (frozen)

How the official ADTC profiler measures this submission, and what each
number means. Authoritative numbers: REPORT.md BENCHMARKS.

## Environment (required)

The bounded runtime is env-gated. The profiler's `llama-bench` child must
inherit the frozen `bounded_3gb` env (from `configs/final_runtime.json`
via `src/sparse.py`): GGML_MOE_K1=4, GGML_MOE_K2=16,
GGML_PHASE6_BOUNDED_CACHE=1, GGML_PHASE6_SLOTS=755, GGML_PHASE6_ASYNC=1,
GGML_PHASE6_PINS=<absolute pins_3.0.txt>, LLAMA_ARG_LAZY_MODE=on.
Without it, the run silently profiles the resident path (~15 GB).
CI exports it in `.github/workflows/official-profiler.yml` and preflights
the `PHASE6_BOUNDED_CACHE slots=755 pins=80` marker before the full run.
llama-bench's custom parser needs our compat patch
(`BENCH_LAZY_ENV_ANCHOR` in `probes/edge0_port/join4_apply.py`) to honor
the lazy-mode variable; stock server/CLI honor it via common_arg.

## Two paths, never merged

- Throughput + memory: our patched `llama-bench` (K1K2 + bounded
  executor). Final headline: 2502.49 MB peak, 16.0 tok/s (16.46 and 15.5
  tok/s observed, rounded).
- Accuracy: stock `llama-cpp-python` in-process (native K8, full mmap).
  It cannot express K4/16. Official: arc_easy 0.72 (50 samples).
  Report the 0.72 with this caveat, always.

## Commands

- Local self-check (fast, skips accuracy): `make profiler`.
- Full official run (CI, ~25 min):
  `gh workflow run "Official ADTC profiler (Q2K submission)" --ref main`
- Raw JSON of the green run: `benchmarks/final/result.json`.
