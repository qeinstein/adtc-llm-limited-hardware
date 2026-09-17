# ik_llama.cpp external baseline probe (v1)

This is a disjoint external-baseline probe for the Jamii native-sparse Qwen
track. It does not change the existing Phase 0--5 reports, the pinned
mainline runtime, or any model representation. The Kaggle script downloads
the model only at run time into `/tmp`; no model binary belongs in this
repository.

## Question and status boundary

Question: can the official `ikawrakow/ik_llama.cpp` CPU build load and run the
exact Qwen3.5-35B-A3B `IQ2_XXS` checkpoint used by Phase 0, on the same Kaggle
AVX2 machine and single-stream interactive configuration?

The source inspection supports the following limited statements:

- The upstream README lists **Qwen3.5-MoE PR 1288** as model support. The
  merged PR also documents a one-sequence caveat for Qwen3.5/Qwen3-Next.
- The current source has a Qwen3.5-MoE graph in `src/graphs/build_qwen35.cpp`
  and a Qwen3.5-MoE tensor loader in `src/llama-load-tensors.cpp`. The loader
  uses the routed tensor families `ffn_gate_exps`, `ffn_up_exps`, and
  `ffn_down_exps`.
- The current IQK CPU dispatch includes `GGML_TYPE_IQ2_XXS` in
  `ggml/src/iqk/iqk_mul_mat.cpp`; the quantized representation and reference
  quantization are in `ggml/src/ggml-quants.c`. The parameters document lists
  `IQ2_XXS` among supported i-quants.
- None of those facts proves that this exact 10,656,955,008-byte Unsloth
  checkpoint loads, routes identically, or produces equal output in ik. The
  probe is the compatibility test.
- No target-model throughput number is imported from upstream. The PR's
  published table is for a different 397B `IQ4_XS` model and different
  hardware, so it is not a Qwen3.5-35B-A3B `IQ2_XXS` result.

## Upstream pins and source links

The probe pins the upstream repository to the observed commit
`3bb386eb68ffee0a5dc7db21da0735d594929eeb`. The script verifies the checked
out commit and records it in the Kaggle result. If that immutable commit
cannot be fetched, the result is a reproducible fetch blocker rather than an
implicit substitution of `main`.

- Repository at the pin:
  <https://github.com/ikawrakow/ik_llama.cpp/tree/3bb386eb68ffee0a5dc7db21da0735d594929eeb>
- Qwen3.5-MoE support PR #1288:
  <https://github.com/ikawrakow/ik_llama.cpp/pull/1288>
- Qwen3.5-MoE graph:
  <https://github.com/ikawrakow/ik_llama.cpp/blob/3bb386eb68ffee0a5dc7db21da0735d594929eeb/src/graphs/build_qwen35.cpp>
- Tensor loader and Qwen3.5-MoE tensor construction:
  <https://github.com/ikawrakow/ik_llama.cpp/blob/3bb386eb68ffee0a5dc7db21da0735d594929eeb/src/llama-load-tensors.cpp>
- IQK CPU matrix/vector dispatch:
  <https://github.com/ikawrakow/ik_llama.cpp/blob/3bb386eb68ffee0a5dc7db21da0735d594929eeb/ggml/src/iqk/iqk_mul_mat.cpp>
- IQ2 reference format implementation:
  <https://github.com/ikawrakow/ik_llama.cpp/blob/3bb386eb68ffee0a5dc7db21da0735d594929eeb/ggml/src/ggml-quants.c>
- Build/run and quantization notes:
  <https://github.com/ikawrakow/ik_llama.cpp/blob/3bb386eb68ffee0a5dc7db21da0735d594929eeb/README.md>
  and <https://github.com/ikawrakow/ik_llama.cpp/blob/3bb386eb68ffee0a5dc7db21da0735d594929eeb/docs/parameters.md>

The current upstream source layout differs from the Phase 0 mainline layout:
Phase 0 patched `src/models/qwen35moe.cpp`, while this ik probe inspects the
loader/graph split above. The probe also searches the checked-out tree for the
actual `GGML_OP_MUL_MAT_ID`, `ggml_mul_mat_id`, and routed-name anchors rather
than assuming a CPU source filename. The upstream PR Files-changed view was
not treated as authoritative when it could not be fetched; no PR path list is
guessed here.

## Exact target checkpoint

- Repository: `unsloth/Qwen3.5-35B-A3B-GGUF`
- Immutable repository revision: `bc014a17be43adabd7066b7a86075ff935c6a4e2`
- File: `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf`
- Size: `10,656,955,008` bytes
- SHA-256: `2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b`
- URL:
  <https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/bc014a17be43adabd7066b7a86075ff935c6a4e2/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf>

Phase 0's pinned mainline control was llama.cpp
`3057bb66c86c46d5781e50e85462a760ba7d1feb`, with exact native top-8 routing,
40 layers, and 256 experts. The ik arm must use the same checkpoint, raw
prompt, `-ngl 0`, four threads, context 512, seed 1234, temperature 0, and
single-stream execution. It must not use `-rtr`, GPU offload, `--cpu-moe`,
expert overrides, K changes, or tensor substitutions.

## Kaggle execution plan

Run `ik_llama_probe.py` as the Kaggle kernel. It will:

1. Clone `https://github.com/ikawrakow/ik_llama.cpp`, fetch the immutable
   `IK_COMMIT`, and record `git rev-parse`, `git status`, and `git log`.
2. Record source matches for architecture, routed tensor names, IQ2_XXS,
   IQK dispatch, `GGML_OP_MUL_MAT_ID`/`ggml_mul_mat_id`, mmap/load-mode, and
   one-sequence handling. This is observation only; it does not patch ik.
3. Build CPU-only with:

   ```text
   cmake -S ik_llama.cpp -B ik_llama.cpp/build-native \
     -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
     -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_METAL=OFF \
     -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF
   cmake --build ik_llama.cpp/build-native --config Release -j4 \
     --target llama-cli llama-bench
   ```

   The result records compiler, CMake, CPU feature, and target failures. The
   probe does not silently switch to a different compiler or commit.
4. Verify model size and SHA-256 before executing it.
5. Capture `llama-cli --version` and `--help`, then run a cold/advisory-cache
   smoke and an immediate warm smoke. It uses `-lm mmap` only if the built
   CLI advertises that option; otherwise the result records a load-mode
   compatibility blocker. `-rtr` is deliberately absent.
6. If `llama-bench` and its expected options exist, run a three-sample
   resident CPU decode probe (`p0`, `n32`, `t4`, `ngl0`). A bench parser failure
   is recorded as unsupported output format, not converted into a number.

The commands are written to the result JSON and per-arm stdout/stderr files.
The model and clone remain outside the repository.

## Measurements and correctness

Each child process is sampled at approximately 20 ms. The sampler records:

- `/proc/<pid>/status`: `VmRSS`, `VmHWM`, `RssAnon`, and `RssFile`;
- `/proc/<pid>/smaps_rollup`: `Rss`, `Pss`, clean/dirty shared/private pages,
  `Anonymous`, `AnonHugePages`, `Referenced`, `Swap`, and related fields;
- `/proc/<pid>/io`: `read_bytes`, `rchar`, and related counters;
- `/proc/<pid>/stat`: minor and major faults;
- a final/peak smaps snapshot where available, plus model-backed VMA rows.

RSS is the process resident total. PSS divides shared pages and is the better
process-attributable comparison when mappings are shared. `RssFile`/PSS of a
file-backed model VMA are operational residency measurements, not proof of
physical SSD reads. `read_bytes` is a kernel I/O counter and mmap faults may
not be represented as one read per logical tensor. `posix_fadvise(DONTNEED)` is
best effort and is reported as an advisory control, never as proof of a cold
NVMe cache.

The probe's exact checks are:

- immutable model size and SHA-256;
- checked-out upstream commit;
- successful CPU binary build and `--version`;
- no GPU layers and one process/one sequence;
- non-empty deterministic output across two identical seeded smokes;
- no `nan`/`inf` marker in the captured logs;
- parsing of architecture/expert-count metadata when the runtime emits it;
- benchmark row/config consistency when JSON is available.

The known mainline response hash is recorded for reference, but a hash
difference is not called an ik routing failure: tokenization, template,
recurrent-state, and floating-point ordering may differ. Stock ik has no
Phase 0 route hook in this probe, so exact route equality is **not measurable**
unless the output explicitly exposes routes. The result therefore reports
route equality as `unmeasured`, not `true`.

## Concrete blockers and decision gates

1. **Exact checkpoint compatibility:** Qwen3.5-MoE and IQ2_XXS are declared
   source capabilities, not a completed test of this exact GGUF. Any load,
   shape, tensor-type, or runtime failure is a direct blocker.
2. **Sparse-storage equivalence:** Phase 0's `TENSOR_READ_LAZY` expert marking
   and `-lzm on` behavior are mainline research changes. The current ik loader
   source does not provide an assumed equivalent. A successful stock ik run
   is therefore a resident external CPU baseline, not an expert-lazy result.
3. **Route observability:** stock output equality cannot establish the 40 x
   top-8 route sequence. An observation-only future patch must first identify
   the actual ik `MUL_MAT_ID` dispatch anchor from the recorded source probe;
   it must not alter IDs, K, weights, or math.
4. **Hybrid recurrent semantics:** upstream PR #1288 documents a one-sequence
   caveat. This probe intentionally uses one sequence and does not claim
   concurrent/batched correctness.
5. **Representation changes:** `-rtr` row-interleaves tensors at load time;
   it is excluded because it would no longer be a clean external baseline.
   No quantization, expert count, routing, or model tensor is changed.
6. **Build/CLI drift:** target names and flags are verified from the checked
   out binary. Missing `llama-cli`, `llama-bench`, `-lm`, or required smoke
   flags are concrete compatibility findings.

## Changed paths in this repository

This investigation adds only the following new, disjoint paths:

- `kaggle/native-sparse-ik-llama-v1/README.md`
- `kaggle/native-sparse-ik-llama-v1/ik_llama_probe.py`
- `kaggle/native-sparse-ik-llama-v1/kernel-metadata.json`

No existing research report, source file, model file, or other Kaggle kernel is
modified by this probe.
