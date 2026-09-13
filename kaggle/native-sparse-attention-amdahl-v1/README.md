# Native-sparse attention Amdahl v1

This is a disjoint, measurement-only Kaggle harness for the next wall-clock
Amdahl question after `native-sparse-amdahl-v2`: how much decode time is spent
in attention projection `MUL_MAT` nodes when the exact Qwen3.5-35B-A3B resident
control is otherwise unchanged?

The script pins llama.cpp at
`3057bb66c86c46d5781e50e85462a760ba7d1feb`, verifies the same IQ2_XXS GGUF
size and SHA-256 as the Phase 0/5A controls, applies the existing routed-expert
lazy-loader patch, and adds only an opt-in attention profiler/bypass in
`ggml/src/ggml-cpu/ggml-cpu.c`.

Arms are:

- `resident_exact_control`: no bypass, 3 repetitions of 64 decode tokens.
- `resident_attention_profile`: same workload with per-family TSC work
  counters enabled.
- `attention_projection_bypassed`: every attention-named `MUL_MAT` destination
is zeroed before its compute; this is an invalid-output wall-clock bound, not
an inference mode.

The bypass does not skip `GGML_OP_FLASH_ATTN_EXT`, normalization, routed MoE,
or any non-attention `MUL_MAT`. It logs the actual tensor names and family
labels to `attention_names.<arm>.jsonl`. The committed floor inventory shows
these exact families:

- linear/DeltaNet: `blk.<layer>.attn_qkv.weight`,
  `blk.<layer>.attn_gate.weight`, `blk.<layer>.attn_norm.weight`;
- full attention: `blk.<layer>.attn_q.weight`,
  `blk.<layer>.attn_k.weight`, `blk.<layer>.attn_v.weight`,
  `blk.<layer>.attn_output.weight`, `blk.<layer>.attn_q_norm.weight`, and
  `blk.<layer>.attn_k_norm.weight`.

In the distinct-name inventory there are 30 linear layers with each of the
first three families and 10 full-attention layers with each of the six full
attention families.

Only the projection families are eligible because the hook requires
`GGML_OP_MUL_MAT`, an `attn_` source name, and excludes the `*_norm` names.
Norms are not bypassed. The
existing Phase 5A classifier uses the same `attn_` predicate for its
`attention_matmul` bucket; `FLASH_ATTN_EXT` is a separate category.

Source/result anchors are pinned `src/models/qwen35moe.cpp::load_arch_tensors`
(the `LLM_TENSOR_ATTN_QKV`, `LLM_TENSOR_ATTN_GATE`, `LLM_TENSOR_ATTN_OUT`,
`LLM_TENSOR_ATTN_Q_NORM`, and `LLM_TENSOR_ATTN_K_NORM` loader calls),
`src/llama-model.cpp` (the corresponding `attn_q*`, `attn_k*`,
`attn_output`, and `attn_gate` tensor-name regex families),
`ggml/src/ggml-cpu/ggml-cpu.c::ggml_compute_forward` (the operation boundary
patched by this harness),
`research/native_sparse_experiments/results/phase5h_floor_decomp_v1/tensor_inventory.jsonl`
(runtime-observed names), and
`research/native_sparse_experiments/results/phase5a_amdahl_v2/ANALYSIS.json`
(the prior aggregate Amdahl split).

Each Kaggle run writes command/configuration, build patch, model identity,
stdout/stderr, `/proc` RSS/fault/I/O samples, profile lines, generated output,
and `result.json` below `/kaggle/working/native-sparse-attention-amdahl-v1-results`.

The bypass changes activations and may change later native routes, so its
speedup is an upper-bound measurement rather than a controlled same-route
microbenchmark. The result records this limitation explicitly. No Kaggle
submission is performed by this checkout.
