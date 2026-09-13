# Phase 6B attention-projection wall-clock Amdahl bound v1

## Hypothesis

Attention and Gated-DeltaNet projection `MUL_MAT` nodes are the next material
resident decode bottleneck after the routed-expert and LM-head bounds, and
their wall-clock share must be measured before selecting a dense kernel.

## Configuration

- Model: Qwen3.5-35B-A3B IQ2_XXS, 10,656,955,008 bytes,
  SHA-256 `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- Model revision: `unsloth/Qwen3.5-35B-A3B-GGUF@bc014a17be43adabd7066b7a86075ff935c6a4e2`
- llama.cpp: `3057bb66c86c46d5781e50e85462a760ba7d1feb`
- Research source: `ba0047b`
- Kaggle: 4-vCPU Intel Xeon, AVX2, 4 threads, inherited CPUs, CPU-only,
  `--poll 0`, resident mmap, lazy mode off
- Workload: fixed clinical prompt, 64 generated tokens, 3 repetitions,
  llama-bench JSON output
- Control and profile use the same exact native top-8 model. The bypass arm
  zeroes attention projection destinations and is intentionally invalid.

## Wall-clock result

| arm | mean tok/s | samples tok/s | ms/token | peak RSS |
|---|---:|---|---:|---:|
| exact resident control | 4.36535 | 4.34031, 4.32561, 4.43012 | 229.077 | 10,530.3 MiB |
| attention projections bypassed | 5.66183 | 5.64407, 5.63629, 5.70514 | 176.621 | 10,529.9 MiB |

The bypass saved 52.456 ms/token, or 22.899% of control wall time. The
attention-free measurement ceiling is 5.662 tok/s. This is a conservative
system bound for the selected projection family, not an exact route-preserving
speedup: zeroing activations can change later routes and graph behavior.

The exact correctness smoke returned the established hash
`a3c3c73a067cbb5fab9e68bd7d06c03d3c488e74467d33c40fae339836006c8c`. The
bypass output is not a quality result.

## Family profile

The opt-in profile recorded summed worker TSC intervals (not wall time):

| family | calls | summed cycles | share of profiled projection cycles |
|---|---:|---:|---:|
| `attn_qkv` | 23,160 | 42,262,282,257 | 48.3% |
| `attn_gate` | 23,160 | 22,101,960,813 | 25.3% |
| `attn_q` | 7,720 | 14,035,150,213 | 16.0% |
| `attn_output` | 7,720 | 6,827,596,467 | 7.8% |
| `attn_v` | 7,720 | 1,276,725,100 | 1.5% |
| `attn_k` | 7,720 | 1,026,360,484 | 1.2% |

The profile’s percentages overlap in interpretation with the wall result only
as operator-work evidence; worker TSC is summed across threads. The dominant
families to attack first are `attn_qkv`, then `attn_gate`, then full-attention
`attn_q`. The inventory identifies these as Q5_K projections.

## Conclusion

Attention projection optimization is worthwhile but cannot independently meet
the throughput target: even an impossible free projection path measured only
5.66 tok/s, below 10 tok/s. Routed experts remain the larger measured wall
term (42.10% in Phase 5A), while the current non-MoE remainder is broad. The
next dense experiment is therefore a Q5_K repack/donor A/B, followed by a
component-specific optimization only if it produces a measured win.

## Artifacts

- Raw Kaggle output and runtime patch: `raw/`
- Harness: `kaggle/native-sparse-attention-amdahl-v1/`
- Kaggle kernel: `toheebogunade/jamii-native-sparse-attention-amdahl-v1`
