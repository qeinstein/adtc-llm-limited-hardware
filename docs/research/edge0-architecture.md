# Edge0-35B architecture notes (Phase 1, source-read + stock-config verified)

> GATE A1 (Edge0 custom multi-file runtime + sidecars + custom engine):
> KILLED under the current profiler contract — recorded, do not resume.
> GATE A2 (large sparse-MoE single-GGUF on stock llama.cpp): OPEN — the live
> direction. This doc is preserved as prior art; its portable lessons feed A2.

Sources: `Edge0-AI/Edge0 @ 0700e65`, stock `Qwen/Qwen3.5-35B-A3B` config.json
(HF, `model_type: qwen3_5_moe`). No weights downloaded; no upstream numbers
reproduced (see §8). Apache-2.0; NOTICE names mlx-lm (MIT) + Ling MLX (Apache-2.0)
vendored code. No Edge0 code copied into this repo.

## 1. Stock model facts (verified from config.json, not docs)

- hidden 2048, 40 layers, 256 routed experts, **native top_k = 8**,
  moe_intermediate = shared_intermediate = 512, `norm_topk_prob` on.
- Hybrid attention: 30× linear-attention + 10× full-attention layers
  (`full_attention_interval: 4`), GQA 16q/2kv, head_dim 256.
- Vocab 248320 → embeddings ≈ 248320×2048 ≈ **509M params** (~1B if LM head untied).
- MTP block present (`mtp_num_hidden_layers: 1`); Edge0 does not use it.
- Routed-expert params: 40 × 256 × 3 × (512×2048) ≈ **32.2B**. Shared experts:
  40 × 3.15M ≈ 126M. Dense remainder (attention/norms/router/embeddings/head)
  ≈ 2–2.5B. Total ≈ 35B ✓.

## 2. What Edge0 changes (K=8 → K=4 tier)

- Decode routes **K=4** instead of stock 8. Halving active experts halves
  per-token expert FLOPs and SSD bytes, but stock weights at K=4 lose quality —
  hence **Recover-LoRA (r=16, α=32)** + a **trained prerouter**, both shipped as
  `.safetensors` adapter files. Stock weights + K=4 with no recovery is NOT viable.
- Experts stored 4-bit affine, group 64, SEPARATE gate/up/down stacked tensors
  under `language_model.model.layers.{N}.mlp.switch_mlp`.

## 3. Dataflow

**A. Prefill.** Chunked (2048). 35B tier uses the on-demand expert path
(`full_layer_prefill=False`, `prefill_full_layers=0`) — same path as decode —
so peak memory matches deployment. (The 8B tier instead bulk-loads whole layers.)
After prefill: drop working set, refresh hot pins, prime staged slots.

**B. One decode token.** (1) Prerouter heads fired at token t−1 already filled
this step's staged slots. (2) Each MoE block gathers its 4 experts from slots
by table lookup — indices never leave GPU, zero host sync per layer.
(3) Attention → shared expert (always resident) → routed experts → norm.
(4) At the step boundary the stager commits predictions for token t+1 into
double-buffered slots and swaps state. (5) Sample (temp/top-k/top-p/rep-penalty,
one host-sync categorical draw; first token forced greedy — a production fix
for `!`-loop derailment).

**C. Always resident.** Attention weights, norms, router, shared expert,
embeddings/LM head, prerouter heads (33 × fp16, tiny), LoRA, KV/linear-state,
MLX cache (capped ~256MB). Reported peak active ≈ **3.3 GB** (M4 Pro).

**D. On SSD.** All 256×40 routed-expert tensors (quantized), paged in per expert.

**E/F. Cache.** `SharedExpertCache`: ONE global cross-layer LRU
(`cache_slots=64` bundles); `PrefetchBuffer` (cap 48) stages not-yet-consumed
prefetches (over-cap evictions counted as `prefetch_wasted`). Hot-pinning is
OFF for 35B (`hot_per_layer=0`); the 35B tier relies on staging, not residency.

**G. mmap.** `SafetensorsMmap`: plain `mmap(ACCESS_READ)` + header parse;
`raw(name)` returns a zero-copy uint8 view of one stacked tensor; per-expert
slices are byte ranges. `madvise(WILLNEED)` + one sequential `seq_read` pass to
warm pages (macOS madvise only warms ~half the file). Portable concept; the
file format is just safetensors.

**H. Bytes/token (own arithmetic from §1).** Per expert: 3×512×2048×0.5B ≈
1.5MB + bf16 scales/biases ≈ 192KB → ~1.7MB. K=4 × 40 layers ≈ **~270MB/token
at 0% cache hit.** At 15 tok/s with no hits that is ~4GB/s — physically
impossible on commodity SSD random reads. The design only works at high
effective hit/predict rates; every mechanism below exists to buy hits.

**I. Routing.** Softmax → top-k → renormalize, bit-identical copy of the
vendored model's math (`moe/routing.py`, parity-tested).

**J/K. Prerouter.** 33 heads (owners = layers 6..38, first consumer = layer 7),
hidden 512, fp16. Features = concat[layer-N MoE input @t, executed top-k
one-hot @t, one-hot @t−1] (double shift: prev-layer + prev-token). Head =
fc1 → erf-GELU → fc2 + linear skip on the same features. Prediction made at
t−1 IS the routed set (`staged_replace` semantics → zero drop by construction).
Fill overlaps the next forward; one cheap sync point per step.

**L. Staging slots.** 4 fixed slots + overflow zero slot. Slot table turns
per-token variable expert sets into fixed gather indices → no graph rebuilds
(`asm_cache`), no `mx.stack` nodes (`incr_stack` row writes), no host sync.
Misses map to the zero slot (contribution dropped) or fall back to exact path.

**M. Recover-LoRA.** Parallel LoRA (r=16) recovering quality lost to K=4 +
4-bit quantization. Training path is NOT in the repo (adapters ship as
artifacts; only a legacy-format converter script exists) — the expensive
bit (distillation recipe) is undisclosed.

**N/O. Why K=4.** Halves the two binding constraints (bytes/token, active
FLOPs). Cost: quality loss requiring recovery training; stock K-reduction
without it is a quality cliff, not a free knob.

## 4. Portability (P/Q/R/S/T)

**P. MLX-specific:** `mx.gather_qmm` (quantized gather-matmul), `mx.compile`
graph caching, `mx.stack`/`put_along_axis` staging tricks, `mx.eval` lazy-eval
overlap model, unified-memory assumption (GPU reads SSD-mmap directly; no
PCIe transfer accounting), Apple-only install target. The vendored
`qwen3_5_moe`/`qwen3_next` model code is MLX.

**Q. Portable:** safetensors byte-range mmap loader (pure Python+numpy —
directly reusable), global-LRU + prefetch-buffer design, slot-table staging
concept, prerouter head math (3 tiny matmuls + GELU — trivially portable),
routing math, LoRA-addition concept. `backends/` facade already isolates MLX
behind `core/nn/io/quant`; a CPU backend would implement that surface with
ggml-style Q4 kernels.

**R. Throughput features:** prerouter overlap, staging (no rebuild/sync),
`gather_qmm` (dequant-fused matmul — never materialize fp16 experts),
sorting path for large sets, compile caching.

**S. Memory-only features:** LRU cap, prefetch cap, on-demand prefill,
MLX cache cap, full-layer load+release (8B prefill).

**T. x86 bottlenecks (estimated, not measured):** (1) Random-read IOPS for
1.7MB expert slices ×160/layer-token — needs io_uring/batched pread + layout
work; page-fault-driven mmap stalls are the #1 risk. (2) No `gather_qmm`
equivalent in scalar llama.cpp — needs ggml Q4 gather path or dequant+GEMM
(dequant alone ≈ 270MB/token memory traffic). (3) Audit runs a SCALAR
(no-SIMD) build — CPU matmul throughput there is far below NEON/M4.
(4) 40-layer × 33-head prerouter CPU overhead per step is small but nonzero.
(5) KV/state for hybrid attention on CPU is unproven in this design.

## 5. What "reproduce" would require (Phase 2, NOT done)

~20GB checkpoint (Edge0-35B-A3B-preview adapters + base) + Apple-Silicon runner
for MLX correctness, or a from-scratch x86 backend. Neither fits current
resources (local disk 100% full; standard GH runners ~14GB free).

## 6. Competition verdict (GATE A — measured against profiler @ ac2e137)

The profiler loads **one file** (`_runtime.model_path`, default `model.gguf`)
directly: `llama-bench` for throughput, `llama-cpp-python` in-process for
accuracy, process-tree RSS for memory. No entrypoint runs; sidecar files
(experts, prerouter, LoRA) are never opened. Consequences:

- An Edge0-style multi-file SSD-streaming runtime scores **nothing** — its
  weights would sit unopened while `llama-bench` fails on (or OOMs over) the
  single GGUF.
- A 35B-equivalent single GGUF (≈19GB Q4) executed densely by `llama-bench`
  touches every page per token → peak RSS ≈ file size ≫ 7GB → OOM/disqualify
  on the 8GB reference laptop.
- S_perf is capped at a fixed 15 tok/s reference, so even a miraculous port
  banks at most S_perf=100 — the same ceiling dense 0.6B already approaches —
  while S_eff collapses under resident weight pages the profiler counts.

**Decision: KILL the Edge0 port under the current scoring contract.**
Revisit only if (a) the profiler gains custom-runtime support, or
(b) organizers confirm an alternative scoring path. Preserved option value:
the portable pieces (§4Q) remain valid research if the contract changes.
