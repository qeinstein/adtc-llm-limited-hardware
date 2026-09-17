# Native MoE profiler contract (llama.cpp source-verified)

Profiler: `Africa-Deep-Tech-Foundation/adtc-profiler @ ac2e137`.
llama.cpp: `ggml-org/llama.cpp @ e5a8d43` (master; the audit builds master at
audit time — revision is NOT pinned, see UNKNOWN-1).
No Edge0 code is used or copied anywhere in this path.

## PROVEN FROM SOURCE

1. **S_perf cap.** Profiler README (pinned commit):
   `S_perf = min(TPS / TPS_REFERENCE, 1.0) * 100`, `TPS_REFERENCE = 15.0`.
   **15 tok/s generation = S_perf 100 = performance DONE.** Chasing >15 tok/s
   buys zero score. (Our `src/score.py` already mirrors this.)
2. **Qwen3.5-MoE is a first-class arch.** `LLM_ARCH_QWEN35MOE / "qwen35moe"` in
   `src/llama-arch.cpp`; tensor mappings exist in `src/llama-model.cpp`.
3. **Only selected experts are multiplied (CPU).** The MoE graph builds
   `ggml_mul_mat_id(ctx0, w, cur, ids)` with the routed top-k `ids`
   (`src/llama-graph.cpp:1550,1570`); the CPU forward kernel
   (`ggml_compute_forward_mul_mat_id_one_chunk`, `ggml/src/ggml-cpu/ggml-cpu.c`)
   iterates the id→row mapping and dots ONLY selected expert rows.
   Unselected experts cost zero FLOPs. (The earlier "effectively dense" GATE A
   aside is retracted: file size ≠ compute, and file size ≠ RSS.)
4. **mmap is ON in the exact profiler path.** `throughput.measure()` builds a
   fixed command — `llama-bench -m <gguf> -p 512 -n 128 -ngl 0 --output json` —
   with no `--load-mode`/`--no-mmap`/`--mlock` flags. `llama-bench` defaults to
   `load_mode=AUTO`, which memory-maps on CPU backends (`use_mmap=true` unless
   the device lacks mmap support). Neither the profiler nor our audit-parity
   scripts disable mmap anywhere.
5. **Quantized weights stay quantized during matmul.** CPU `vec_dot` kernels
   dequantize on the fly from the mmap'd quantized blob; there is no upfront
   dequantized copy. An IQ2_XXS expert page fault brings ~2.5-bit data, not fp16.
6. **Memory scoring counts touched mmap pages.** `memory.py` sums
   `psutil` RSS of the profiler process + recursive children. Untouched mmap
   pages and page cache are NOT counted; faulted-in expert pages ARE.
   Metric recorded is peak over 100ms samples; S_eff = 100*(7-peak_gb)/7.
7. **IQ-quant MoE CPU path exists.** `ggml_cpu_iqp_supports_mul_mat_id` /
   `ggml_compute_forward_mul_mat_id_iqp` (`ggml/src/ggml-cpu/iqp.cpp`).

## MEASURED (this repo, prior runs)

- Dense Qwen3-0.6B Q4_0: 19.1 tok/s scalar x86 (quant-sweep.yml) — the small
  baseline the MoE candidate must beat on TOTAL, not speed.

## INFERRED (same-core reasoning, not yet directly observed)

- `llama-cpp-python` (accuracy stage) mmap-defaults like the core library
  (its `Llama` defaults to `use_mmap=True`); exact pinned version's behavior
  not re-verified here.
- Prefill (`-p 512`) processes 512 tokens through all layers and likely faults
  a large expert working set; decode (`-n 128`) then faults incrementally.
  Whether steady-state RSS stabilizes or grows is the experiment below.

## UNKNOWN (decided by measurement, not argument)

- U1. Resident RSS over tokens for a 35B MoE GGUF under `-p 512 -n 128`.
- U2. Linux readahead amplification around 1.7MB expert slices.
- U3. Whether routing touches enough unique experts over 128 tokens that
  residency explodes, or stabilizes under ~2GB.
- U4. Scalar-CPU decode tok/s for K=8 35B MoE (prefill time also unknown —
  512 prompt tokens × 35B params on scalar CPU may dominate wall-clock).
- U5. Exact llama.cpp master revision the auditors will build (moving target).

## The one experiment (GATE B)

`.github/workflows/moe35-bench.yml` — scalar build + single-file
`unsloth/Qwen3.5-35B-A3B-GGUF` download + exact profiler bench command with
RSS-over-time tracing + generation smoke test. Results decide GO/MODIFY/KILL.
