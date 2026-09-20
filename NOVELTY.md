# Novelty & Prior Art — Jamii Afya Sparse MoE System

This note exists to keep our claims honest. We reviewed the directly
relevant CPU/NVMe MoE inference systems before writing any novelty
language. Bottom line: **every ingredient we use has prior art**; what
appears distinctive — to the best of our knowledge — is the **validated
composition at the demonstrated operating point**: real-router K4/16
execution of Qwen3.6-35B-A3B with routed-experts-only Q2_K weights under
an explicit bounded expert-slot executor, running CPU-only at
2502.5 MB official peak RSS (2301.2 MB dev point) with bit-exact outputs.

## 1. What is claimed

1. **The composition.** We found no prior work demonstrating this exact
   combination: Qwen3.6-35B-A3B + real (unmodified) routing + K4/16
   narrowed execution + routed-experts-only Q2_K transcode + explicit
   bounded expert-slot staging with `pread` + CPU-only commodity-laptop
   execution at 2502.5 MB official peak RSS (2301.2 MB dev point).
2. **Bounded-slot executor inside llama.cpp with deterministic memory
   accounting.** Fixed 755-slot staging pool + 80 globally-pinned hot
   experts, background async fill, and an RSS design that is a *budget*,
   not an observation — vs OS-paging or heuristic-cache approaches whose
   footprint is emergent.
3. **Bit-exact validation methodology at this envelope.** Bounded arm
   proven bit-identical (outputs AND routes) against the resident arm,
   plus sim-predicted vs on-device hit-rate validation — so the low-RAM
   number is a guarantee about the same answers, not a different system.
4. **Routed-experts-only Q2_K transcode recipe.** Exactly the 120 routed
   expert tensors requantized to Q2_K with every other tensor keeping
   base types — a quality/size trade aimed at the experts that dominate
   both, byte-verified and layout-verified (120/120).

## 2. What is NOT claimed (explicit non-novelty)

We did **not** invent: SSD/NVMe expert streaming, expert caching or
prefetching, MoE itself, quantization, K4/16 reference-mass
renormalization (paper 2609.04575, Eq. 2 — implemented, not invented),
CPU offload, hot-expert pinning, or the underlying model, runtime, or
quantization formats (Qwen, llama.cpp, Unsloth/ik_llama quants). Any
sentence in this repo that could read as claiming those is a bug —
please file it as one.

## 3. Prior-art table

| system | idea in one line | target / envelope | closest overlap | key difference from Jamii |
|---|---|---|---|---|
| LLM in a Flash (Apple, arXiv:2312.11514) | flash offload + windowing/bundling; 2× DRAM models | phones, dense LLMs | SSD streaming itself | dense-focused; windowing mechanism, not bounded MoE slots or routing narrowing |
| AirLLM (lyogavin) | layer-by-layer streaming; 70B on 4 GB VRAM | consumer GPUs | streaming under a fixed budget | GPU-centric, layer granularity; no expert-slot cache design or K-narrowing |
| Fiddler (ICLR'25, arXiv:2402.07033) | CPU computes cold experts, GPU hot; Mixtral-8x7B >3 tok/s on 24 GB GPU | 24 GB GPUs | CPU participation in MoE inference | requires a GPU; placement optimization, not a bounded-RAM CPU-only envelope |
| DAOP (arXiv:2501.10375) | per-sequence expert placement + predictive CPU pre-calc | GPU+CPU hybrids | activation-aware placement | GPU-required; prediction machinery vs our fixed slots |
| MoE-Infinity (arXiv:2401.14361) | batch-1 sparsity tracing + prefetch + GPU expert cache | personal GPUs | activation-aware caching | GPU cache over host/SSD; heuristic cache, not deterministic slot budget |
| SwapMoE (arXiv:2308.15030) | tunable-memory-budget MoE serving | GPU serving | the *budget knob* idea | server/GPU serving frame; no narrowed routing or CPU-only laptop point |
| FlashMoE (arXiv:2601.17063; danveloper/flash-moe + forks) | SSD expert streaming + ML cache replacement (+51% hits vs LRU) | Apple Silicon, 24–48 GB RAM | SSD expert streaming on Qwen MoE | Metal/GPU compute; 10–20× our RAM; learned policy vs fixed slots; no K4/16 or experts-only Q2_K composition |
| HotPin (lozzkappa/hotpin-llm) | llama.cpp patches; mlock top-K experts; 120B lossless at 3.84 tok/s / 19 GB, CPU+NVMe | consumer CPU, 19 GB | llama.cpp-based, lossless, pinning | pinning-only over OS paging vs our explicit bounded executor; 8× our RSS; no routing narrowing |
| TokenQL (eiomra/tokenql) | bounded-memory SSD-backed Qwen runtime; whole expert slots, frequency-gated | low-RAM Qwen | bounded slots for Qwen MoE | independent implementation; ours adds K4/16 + experts-only Q2_K + 2.44 GB officially-profiled point (detail-limited comparison — see §6) |
| llama.cpp upstream MoE offload (`-ot`, `--n-cpu-moe`; ik_llama.cpp `exps=CPU`+mmap; issue #20757 two-tier cache proposal) | tensor→device placement; OS-paged CPU experts | hybrid CPU/GPU | expert placement inside our own runtime | coarse placement + emergent paging; no bounded executor, no routing narrowing; our base quants build on ik/Unsloth work |
| KTransformers | experts on CPU, attention on GPU | GPU + big CPU RAM | hybrid MoE execution | requires a GPU |
| vLLM expert cache (`--moe-expert-cache-size`, LFRU) | GPU cache over CPU-pinned experts | server GPUs | hot-expert caching | datacenter frame; no CPU-only envelope |
| PowerInfer / FlexGen / Pre-gated MoE / eMoE / EdgeMoE / MoE-Lens / MoEpic | hot/cold split, 3-tier placement, pre-gating, task-aware reuse, edge SoCs, split caching | various | individual mechanisms | each overlaps one mechanism at most; none matches the composition + operating point |

## 4. Experimental evidence / ablations

- **Bit-exactness:** bounded arm == resident arm on outputs AND routes
  (exact-IQP executor; `configs/final_runtime.json`).
- **Routing cost:** MMLU-200 native K8 42.0 → K4/16 41.0 (−1.0pt);
  Q2_K experts 38.5 (−2.5pt further); deterministic loglik, n=200.
- **Cache reality vs model:** sim-predicted hit rates validated against
  on-device staging behavior (bounded_3gb arm); see ARCHITECTURE.md §6.
- **Runtime ablations (phase35):** resident vs K2 vs locality floor vs
  scheduler/P1/P2 variants — rejected directions documented in
  ARCHITECTURE.md §10 (exact inference beat everything approximate).
- **Transcode verification:** byte-identity + 120/120 Q2_K layout asserted
  in the publishing kernel before upload.

## 5. Authoritative hardware evidence (official profiler)

The development operating point above (2301.2 MiB @ ~2.9 tok/s, Kaggle
4-vCPU) is NOT the audit number. The authoritative measurement is the
official ADTC profiler run:

- Status: PASS (workflow `.github/workflows/official-profiler.yml`, run
  35514252643 on the evidence source commit @ `c454f1a`; artifacts preserved).
- Result: peak_rss 2502.49 MB, steady 2436.26 MB; 16.0 tok/s headline
  generation (16.46 and 15.5 tok/s observed, rounded; pp512/tg128, 2 threads),
  TTFT 26394.46 ms; arc_easy 50-sample
  0.72 acc_norm; AMD EPYC 7763 4-core / 15.6 GB / Ubuntu 22.04 CPU-only;
  model 12262341600 bytes @ `0f3698ae…c7603b`; no throttling.

Do not replace the official number with the development RSS, or vice
versa; they answer different questions (audit hardware vs dev hardware).

## 6. Links / citations

- Apple, LLM in a Flash: https://arxiv.org/abs/2312.11514
- AirLLM: https://github.com/lyogavin/airllm (aka tamdrew/airllm)
- Fiddler: https://arxiv.org/abs/2402.07033
- DAOP: https://arxiv.org/abs/2501.10375
- MoE-Infinity: https://arxiv.org/abs/2401.14361
- SwapMoE: https://arxiv.org/abs/2308.15030
- FlashMoE paper: https://arxiv.org/abs/2601.17063; engine: https://github.com/danveloper/flash-moe
- HotPin: https://github.com/lozzkappa/hotpin-llm
- TokenQL: https://github.com/eiomra/tokenql
- llama.cpp MoE offload: PR ggml-org/llama.cpp#11397; issue #20757;
  ik_llama.cpp PR #232/#239
- K4/16 renormalization: paper 2609.04575, Eq. 2 (implemented, not invented)
- vLLM expert offload: vllm-project/vllm PR #37190
- KTransformers: https://github.com/kvcache-ai/ktransformers

## 7. Limitations of the novelty search

- Web search conducted 2026-09-20 over papers, repos, and technical
  write-ups; non-English and paywalled sources were not systematically
  covered.
- Fast-moving 2026 systems (FlashMoE forks, TokenQL, contemporary
  llama.cpp PRs) were compared from public READMEs/issues/surveys, not
  from running their code — behavioral details may be mischaracterized;
  corrections welcome.
- "No prior work demonstrating this exact combination" is a statement
  about what we found, not a proof of absence. The composition claim is
  deliberately narrow so it stays checkable.
