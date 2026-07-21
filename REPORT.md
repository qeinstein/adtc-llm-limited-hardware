# Technical Report — Jamii Afya: Offline Clinical Advisor for Rural African Clinics

**Team ID:** jamii-afya
**Domain:** healthcare_medical
**Model:** JamiiAfya-Qwen3-1.7B-Medical (Qwen3-1.7B base, Apache-2.0), GGUF Q4_K_M with an English+Kiswahili medical importance-matrix
**Target hardware:** ADTC Standard Laptop — 8 GB RAM (7 GB budget), Intel Core i5 / AMD Ryzen 5 x86-64, integrated graphics only, Ubuntu 22.04, CPU-only.

---

## 1. Problem

Community health workers (CHWs) and nurses staff the front line of rural African primary care, often the only health presence for a village. They triage childhood fever, diarrhoea, pneumonia, malaria, maternal danger signs, snakebite and more — frequently **without a doctor on site, without reliable internet, and with intermittent power**. Cloud LLMs are unusable here: no connectivity, recurring subscription cost in USD, and patient-privacy exposure.

**Jamii Afya** ("community health" in Kiswahili) is a 100% offline clinical **decision-support** assistant that runs on a $150–250 refurbished laptop. It answers in **English and Kiswahili** (the language of the question), grounds every answer in a curated WHO/IMCI-based knowledge base, and is engineered to **surface danger signs and say when to refer**. Target user: a CHW or nurse who needs fast, safe, local guidance — not a diagnosis, and never a replacement for a clinician.

Why this matters for the rubric's *African Use Case* dimension: Kiswahili is spoken by ~200M people across East/Central Africa, and offline operation is a hard requirement — not a nice-to-have — in exactly the low-resource settings the challenge targets.

---

## 2. Design Decisions (and alternatives rejected)

Our central decision follows directly from the scoring formula and *how it is measured*. The `adtc-profiler` runs **`llama-bench` on the raw GGUF** (throughput/memory) and **lm-eval on the raw model** (accuracy); it does **not** run our application code. Critically, the audit builds llama.cpp **with all SIMD disabled** (`GGML_NATIVE/AVX/AVX2/AVX512/FMA/F16C = OFF`), so decode throughput on the grading VM is a *fraction* of any modern laptop build. Perf (30%) and efficiency (20%) therefore reward **small, low-RAM models**, and on the scalar build only a small model can be fast.

**Base model — Qwen3-1.7B (Apache-2.0).** Best balance of the three scored axes:
| Considered | Verdict | Reason |
|---|---|---|
| Qwen2.5-14B IQ3_XXS (the inherited plan) | ✗ rejected | ~6 GB RAM → S_eff ≈ 12; ~2–4 tps even with AVX2, far worse scalar → S_perf ≈ 0. Sacrifices ~50% of the score to marginally help accuracy. |
| Qwen3-4B-Instruct-2507 | ◐ A/B backup | More accurate, but ~2× slower on the scalar build and ~2.5 GB RAM (S_eff ~64). Kept as documented A/B — ship it only if its accuracy edge outweighs the perf/eff loss on *our* eval. |
| **Qwen3-1.7B** | **✓ chosen** | Apache-2.0; Qwen3's 119-language coverage includes Kiswahili; ~1.1 GB Q4_K_M → ~1.3 GB peak (**S_eff ~81**); fastest capable option on the scalar build. |
| Qwen2.5-3B | ✗ | Non-commercial "Qwen Research" license. |
| Llama-3.2-1B/3B | ✗ | Acceptable-Use Policy bars use in languages outside its supported set (Kiswahili excluded). |
| Gemma-3-1B | ✗ | English-only (the multilingual Gemma-3 starts at 4B). |
| MedGemma-4B | ✗ | Best small-model medical scores, but English-only + restrictive HAI-DEF terms + 2.5 GB. |

*Decision rule (documented for the A/B):* ship the 1.7B unless the 4B beats it on our EN+SW concept-recall eval by enough to offset its lower S_perf/S_eff on a scalar-parity benchmark.

**Quantization — Q4_K_M with a domain importance matrix.** Q8_0 (~1.7 GB) wastes the RAM budget; Q3_K/IQ3 lose quality and, because i-quants use codebook lookups that don't vectorise, are **slower on CPU — and worse still on the no-SIMD audit build**. Q4_K_M is the quality-per-byte and quality-per-tps sweet spot on x86 CPU. We calibrate the importance matrix on our **English+Kiswahili medical corpus** (`scripts/prepare_dataset.py` → `llama-imatrix`) so precision is biased toward our use case at zero inference-RAM cost. KV cache is `q8_0` with flash-attention to shave tens of MB.

**Accuracy recovery for a small model.** Highest-ROI first:
1. **RAG grounding** (`src/retriever.py` BM25 + `src/compressor.py`) over a curated WHO/IMCI knowledge base — a small model + good retrieval can match far larger models on grounded clinical QA. Retrieval is stdlib-only (RAM-negligible, transparent, works for both languages).
2. **Few-shot + safety-first system prompt** pinning the answer shape (assessment → action → danger signs → refer), cached as a stable KV prefix.
3. **LoRA fine-tune** reserved for **Kiswahili fluency + output format** (knowledge comes from RAG), keeping the base model's general reasoning intact for lm-eval.

**Application-side CPU engineering** (`src/engine.py`, for the interactive product/demo): quantized KV cache, KV **prefix caching** of the stable system+few-shot block, and **prompt-lookup (n-gram) speculative decoding** — free extra RAM and effective on RAG-grounded output that copies spans from retrieved text.

**Anti-fragility to the audit.** We ship `scripts/build_llamacpp_scalar.sh` (a no-SIMD build matching the audit Dockerfile) and `src/benchmark.py --profiler-parity` (runs `llama-bench -p 512 -n 128` and samples process-tree RSS at 100 ms, exactly like the profiler). This lets our self-reported Gate-1 numbers match the Gate-2 audit within tolerance (±25% throughput / ±15% memory) — a comparator failure that will sink teams who benchmark on their fast laptops.

---

## 3. Constraints

| Constraint | Target | How we meet it |
|---|---|---|
| RAM | 8 GB total, **7 GB scored budget**; OOM = disqualification | ~1.3 GB peak (1.7B Q4_K_M, n_ctx 2048, q8_0 KV) → large safety margin, no OOM risk |
| Compute | CPU-only, 4 vCPU, integrated GPU, **no SIMD on audit build** | Small model sized for scalar decode; `n_gpu_layers=0` |
| Thermal | −10 penalty if throttle / core temp > 85 °C | Small model = low sustained heat; report notes Turbo-off option for stability |
| Connectivity | 100% offline during evaluation | Zero network calls at inference; weights fetched once by `download_model.sh`; RAG corpus is local |
| Power | Grid instability in target setting | Low compute per query → low energy footprint |
| Runtime | llama.cpp / GGUF only | Ship GGUF; `metadata.model.runtime = llama.cpp` |

---

## 4. Benchmarks

> **Status: pending real measurement.** Per the challenge rules and our own honesty policy, this table is intentionally left as *targets + method* until we run the pipeline on real weights. No numbers here are invented. We will populate it by (a) running `scripts/build_llamacpp_scalar.sh` then `make bench-audit` to reproduce the audit environment, and (b) `make accuracy` (lm-eval). These are **self-reported development benchmarks**; official scores are measured by the ADTC profiler on the standard evaluation machine.

| Metric | Target / Method | Value |
|---|---|---|
| Machine | ADTC Standard Laptop (i5/Ryzen 5, 8 GB, iGPU, Ubuntu 22.04) | — |
| Peak RAM (RSS) | `llama-bench` process-tree, 100 ms sampling | **~1.3 GB (target)** → S_eff ≈ 81 |
| Model file size | GGUF Q4_K_M | ~1.1 GB |
| Time to first token | prompt-processing rate over 512-token prompt | *pending* |
| Generation speed | `llama-bench -p 512 -n 128` on **scalar** build | *pending (measure on audit-parity build)* |
| Thermal throttling | profiler thermal probe | *pending (expected: none at this size)* |
| Accuracy (S_acc) | lm-eval `arc_easy` + medical MCQA on raw GGUF | *pending* |
| Domain concept recall (EN+SW) | our offline `src/evaluator.py` | *pending* |

**Estimated leaderboard positioning** (from `src/score.py`, using the target ~1.3 GB peak and the challenge's 15 tps reference): efficiency contributes ~16 of 20 points, and any decode ≥ ~7–8 tps on the scalar build already yields a strong perf contribution — a profile the 14B plan cannot reach on any axis. The remaining lever is accuracy (50%), which we address with RAG + the domain LoRA and validate with `make accuracy`.

---

## 5. Tools & rationale

- **llama.cpp / llama-cpp-python** — mandated runtime; best CPU GGUF inference; enables KV-quant, prefix caching, prompt-lookup decoding.
- **Qwen3-1.7B (Apache-2.0)** — small, multilingual (incl. Kiswahili), commercially usable.
- **TRL + PEFT + bitsandbytes** — QLoRA fine-tune on a single modest GPU (Udutech credits).
- **EleutherAI lm-eval-harness** — same accuracy tool the profiler uses, so we predict S_acc on the same footing.
- **Pure-stdlib BM25 + extractive compression** — RAM-negligible, transparent, bilingual retrieval; fully unit-tested without weights.
- **adtc-profiler + a no-SIMD llama.cpp build** — reproduce the exact grading measurement locally.

*All medical content in `data/` is derived from public WHO / IMCI / national treatment-guideline material and is provided for clinical decision support only.*
