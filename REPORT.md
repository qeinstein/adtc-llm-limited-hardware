# Technical Report — Jamii Afya: Offline Clinical Advisor for Rural African Clinics

**Team ID:** jamii-afya
**Domain:** healthcare_medical
**Model:** JamiiAfya-Qwen3-0.6B-Medical — fine-tuned from Qwen3-0.6B-**Base** (Apache-2.0), GGUF **Q4_0** with an English+Kiswahili medical importance matrix
**Target hardware:** ADTC Standard Laptop — 8 GB RAM (7 GB budget), Intel Core i5 / AMD Ryzen 5 x86-64, integrated graphics only, Ubuntu 22.04, CPU-only.

---

## 1. Problem

Community health workers (CHWs) and nurses staff the front line of rural African primary care, often the only health presence for a village. They triage childhood fever, diarrhoea, pneumonia, malaria, maternal danger signs, snakebite and more — frequently **without a doctor on site, without reliable internet, and with intermittent power**. Cloud LLMs are unusable here: no connectivity, recurring subscription cost in USD, and patient-privacy exposure.

**Jamii Afya** ("community health" in Kiswahili) is a 100% offline clinical **decision-support** assistant that runs on a $150–250 refurbished laptop. It answers in **English and Kiswahili** (the language of the question), grounds every answer in a curated WHO/IMCI-based knowledge base, and is engineered to **surface danger signs and say when to refer**. Target user: a CHW or nurse who needs fast, safe, local guidance — not a diagnosis, and never a replacement for a clinician.

Why this matters for the rubric's *African Use Case* dimension: Kiswahili is spoken by ~200M people across East/Central Africa, and offline operation is a hard requirement — not a nice-to-have — in exactly the low-resource settings the challenge targets.

---

## 2. Design Decisions (and alternatives rejected)

Our central decision follows directly from the scoring formula and *how it is measured*. The `adtc-profiler` runs **`llama-bench` on the raw GGUF** (throughput/memory) and **lm-eval on the raw model** (accuracy); it does **not** run our application code. Critically, the audit builds llama.cpp **with all SIMD disabled** (`GGML_NATIVE/AVX/AVX2/AVX512/FMA/F16C = OFF`), so decode throughput on the grading VM is a *fraction* of any modern laptop build. Perf (30%) and efficiency (20%) therefore reward **small, low-RAM models**, and on the scalar build only a small model can be fast.

**Base model — Qwen3-0.6B (Apache-2.0).** Best balance of the three scored axes:
| Considered | Verdict | Reason |
|---|---|---|
| Qwen2.5-14B IQ3_XXS (the inherited plan) | ✗ rejected | ~6 GB RAM → S_eff ≈ 12; ~2–4 tps even with AVX2, far worse scalar → S_perf ≈ 0. Sacrifices ~50% of the score to marginally help accuracy. |
| Qwen3-4B-Instruct-2507 | ◐ A/B backup | More accurate, but ~2× slower on the scalar build and ~2.5 GB RAM (S_eff ~64). Kept as documented A/B — ship it only if its accuracy edge outweighs the perf/eff loss on *our* eval. |
| **Qwen3-0.6B-Base** | **✓ chosen** | Apache-2.0; Qwen3's 119-language coverage includes Kiswahili; 364 MB Q4_0 → **527 MB measured peak** (S_eff 92.65); fastest capable option on the scalar build. Base rather than Instruct: the profiler ranks answer choices by raw loglikelihood with no chat template, a regime where base checkpoints outperform instruct-tuned ones. |
| Qwen2.5-3B | ✗ | Non-commercial "Qwen Research" license. |
| Llama-3.2-1B/3B | ✗ | Acceptable-Use Policy bars use in languages outside its supported set (Kiswahili excluded). |
| Gemma-3-1B | ✗ | English-only (the multilingual Gemma-3 starts at 4B). |
| MedGemma-4B | ✗ | Best small-model medical scores, but English-only + restrictive HAI-DEF terms + 2.5 GB. |

*Decision rule (documented for the A/B):* ship the 0.6B unless the 4B beats it on our EN+SW concept-recall eval by enough to offset its lower S_perf/S_eff on a scalar-parity benchmark.

**Quantization — decided empirically, not by convention.** The usual advice ("Q4_K_M is the sweet spot") assumes a normal-size model; independent research on Qwen3 specifically found that **compression damage scales inversely with model size** — a 14B model loses ~1% accuracy at 4-bit while a 0.6B loses ~10-12%, a roughly 10x amplification. Meanwhile 8-bit is close to lossless for this model, and at 0.6B the RAM cost of going from Q4_K_M (~460 MB) to Q8_0 (~767 MB) is well under 1 point of the efficiency score. So for our size class the "obvious" choice may be wrong. We resolved this empirically via `.github/workflows/quant-sweep.yml` across Q4_0/Q4_K_M/Q5_K_M/Q6_K/Q8_0 on the scalar build, and then re-measured accuracy on the **fine-tuned** model for Q4_0/Q5_K_M/Q8_0. **Result: the accuracy differences are not real.** The spread was 3.0 points on arc_easy and 2.5 on medmcqa, while the 95% confidence interval at n=200 is +/-5.6 and +/-6.7 respectively — every gap is smaller than its own error bar. Since precision buys nothing measurable, the decision falls to speed margin, and **Q4_0 wins outright**: 20.33 tok/s versus a 15 tok/s cutoff (a 35% margin), where Q8_0 sits near the line. This is worth stating plainly because an earlier reading of the same data suggested higher precision was worth ~1.9 points; computing the intervals showed that conclusion was noise. Q3_K/IQ3 are excluded regardless of the sweep result: i-quants use codebook lookups that don't vectorise and are measurably **slower without SIMD**, so they lose on both axes at once. We calibrate the importance matrix on our **English+Kiswahili medical corpus** (`scripts/prepare_dataset.py` → `llama-imatrix`) so precision is biased toward our use case at zero extra inference-RAM cost.

**Accuracy recovery for a small model.** Highest-ROI first:
1. **RAG grounding** (`src/retriever.py` BM25 + `src/compressor.py`) over a curated WHO/IMCI knowledge base — a small model + good retrieval can match far larger models on grounded clinical QA. Retrieval is stdlib-only (RAM-negligible, transparent, works for both languages).
2. **Few-shot + safety-first system prompt** pinning the answer shape (assessment → action → danger signs → refer), cached as a stable KV prefix.
3. **LoRA fine-tune** reserved for **Kiswahili fluency + output format** (knowledge comes from RAG), keeping the base model's general reasoning intact for lm-eval.

**Application-side CPU engineering** (`src/engine.py`, interactive product only): quantized KV cache and KV **prefix caching** of the stable system+few-shot block. Prompt-lookup (n-gram) speculative decoding was implemented and then **disabled**: testing showed this llama-cpp-python version raises `could not broadcast input array from shape (N,) into shape (0,)` on long chat-formatted prompts (system + few-shot + RAG context). It works on short completions, which is exactly what a smoke test would use — only an end-to-end test with the real prompt caught it.

**Anti-fragility to the audit.** We ship `scripts/build_llamacpp_scalar.sh` (a no-SIMD build matching the audit Dockerfile) and `src/benchmark.py --profiler-parity` (runs `llama-bench -p 512 -n 128` and samples process-tree RSS at 100 ms, exactly like the profiler). This lets our self-reported Gate-1 numbers match the Gate-2 audit within tolerance (±25% throughput / ±15% memory) — a comparator failure that will sink teams who benchmark on their fast laptops.

---

## 3. Constraints

| Constraint | Target | How we meet it |
|---|---|---|
| RAM | 8 GB total, **7 GB scored budget**; OOM = disqualification | **527 MB measured peak** (0.6B Q4_0, n_ctx 2048, q8_0 KV) → 7.5% of budget, no OOM risk |
| Compute | CPU-only, 4 vCPU, integrated GPU, **no SIMD on audit build** | Small model sized for scalar decode; `n_gpu_layers=0` |
| Thermal | −10 penalty if throttle / core temp > 85 °C | Small model = low sustained heat; report notes Turbo-off option for stability |
| Connectivity | 100% offline during evaluation | Zero network calls at inference; weights fetched once by `download_model.sh`; RAG corpus is local |
| Power | Grid instability in target setting | Low compute per query → low energy footprint |
| Runtime | llama.cpp / GGUF only | Ship GGUF; `metadata.model.runtime = llama.cpp` |

---

## 4. Benchmarks

All numbers below are **measured**, not estimated. Throughput and memory come from a real
`adtc-profiler 0.1.0` participant-mode run against a **scalar (no-SIMD)** llama.cpp build;
accuracy comes from `scripts/mcq_eval.py`, which reads raw logits directly (see the bug note
in §6). Nothing in this table is projected.

### 4.1 Gate-1 profiler run (official tool)

Environment: AMD EPYC 7763, 4 cores, 15.6 GB RAM, no GPU, Ubuntu 24.04, scalar build.

| Metric | Measured | Score contribution |
|---|---|---|
| Generation throughput | **20.33 tok/s** (`-p 512 -n 128`) | S_perf **100.0 / 100** → 30.00 of 30 |
| Peak process-tree RSS | **526.91 MB** (0.515 GB) | S_eff **92.65 / 100** → 18.53 of 20 |
| Steady-state RSS | 495.97 MB | — |
| First-token latency | 12 104 ms | not scored (prefill, not decode) |
| Thermal throttling | `throttled: false` | P_thermal **0.00** |
| Parameter count | 596 049 920 vs "0.6B" claimed | `params_match: true` |

**Banked from speed + efficiency alone: 48.53 / 50.** Total therefore resolves to
`S_total = 48.53 + 0.50 x S_acc`.

### 4.2 Accuracy (held-out test/validation splits only)

Training used **train splits exclusively**; evaluation uses test/validation splits. No overlap.

| Task | Pre-fine-tune baseline | Shipped model | n |
|---|---|---|---|
| arc_easy (`acc_norm`) | 51–57 | **79.5** | 200 |
| arc_easy (`acc_norm`) | — | 78.7 | 300 |
| medmcqa (`acc_norm`) | — | 36.0 | 300 |
| openbookqa (`acc_norm`) | — | 51.5 | 200 |

The fine-tune moved arc_easy from roughly random-plus to 79.5, which is the single largest
score movement in the project. Generalisation is **uneven**: medmcqa at 36.0 is only 11 points
above the 25% chance floor for a 4-option task, and we report it rather than quoting the best
number alone. At n=300 the 95% CI is about +/-4.7 points, so differences smaller than that
between model revisions are noise.

### 4.3 Estimated leaderboard position

With the measured 48.53 banked, S_total lands between **66.5** (if graders' tasks resemble
medmcqa) and **88.3** (if they resemble arc_easy). We deliberately do not quote a single
figure: the automated portion of S_acc depends on which tasks are used, and that is not
published.

## 5. Tools & rationale

- **llama.cpp / llama-cpp-python** — mandated runtime; best CPU GGUF inference; enables KV-quant, prefix caching, prompt-lookup decoding.
- **Qwen3-0.6B (Apache-2.0)** — small, multilingual (incl. Kiswahili), commercially usable.
- **PEFT + bitsandbytes + a custom `Trainer`** — QLoRA fine-tune. Not TRL's `SFTTrainer`: the listwise ranking objective needs full control of `compute_loss`.
- **EleutherAI lm-eval-harness** — same accuracy tool the profiler uses, so we predict S_acc on the same footing.
- **Pure-stdlib BM25 + extractive compression** — RAM-negligible, transparent, bilingual retrieval; fully unit-tested without weights.
- **adtc-profiler + a no-SIMD llama.cpp build** — reproduce the exact grading measurement locally.

---

## 6. Known limitations and bugs found

We report these because a system used for clinical decision support should be judged on what
it gets wrong, not only on its best numbers. Every item below was found by testing, and each
is reproducible.

**Kiswahili generation degrades on long or weakly-grounded answers.** When a retrieved
guideline strongly anchors the answer the model produces correct Kiswahili — e.g. *"Hesabu
mipumuo kwa dakika moja (kupumua haraka ni ishara ya nimonia kwa watoto)... ISHARA ZA HATARI...
peleka haraka kituo cha rufaa."* When it must generate novel Kiswahili it degenerates into
repetition and coins non-words (observed: `kifua kunyonya`, a fusion of `kushindwa kunyonya`
and `kifua kubonyea`). Three fine-tuning rounds, including one with 1,182 distilled Kiswahili
clinical rows, did not resolve this. Our reading is that this is a **base-capability limit**:
LoRA adapts existing ability, and Qwen3-0.6B-Base carries too little Kiswahili to adapt.
Mitigations shipped: a repetition guard and a script-leakage trim in `src/engine.py`.

**Uneven accuracy across MCQA families.** arc_easy 79.5 versus medmcqa 36.0. The fine-tune
generalised far better to general-science ranking than to clinical MCQA.

**The model can still invent clinical specifics.** Observed: recommending an antihistamine for
a pre-eclampsia presentation, and suggesting sugar be added to ORS (dangerous — it alters
osmolarity). The system prompt forbids guessing doses; a 0.6B model does not reliably comply.
This is the strongest argument for the RAG grounding and for the product being framed as
decision support with mandatory clinician confirmation.

**No structured-output guarantee.** Asked for JSON, the model returns prose. The correct fix is
grammar-constrained decoding (GBNF) in llama.cpp, which makes malformed output structurally
impossible; that is not yet wired in.

**Prefill dominates latency on long prompts.** First-token latency is 12.1 s for a 512-token
prompt on the scalar build. Decode throughput (the scored metric) is unaffected, but it is felt
in the demo.

### Bugs found by testing, and what they taught us

- **The accuracy checker was wrong before the model was.** `mcq_eval.py` originally used
  llama-cpp-python's `create_completion(echo=True, logprobs=N)`, whose echoed token logprobs are
  misaligned. It ranked "banana" above "Paris" for the capital of France. Rewritten to read raw
  logits. Had we trusted it, we would have "fixed" a model that was not broken.
- **A weak retrieval match caused wrong clinical advice.** For a pre-eclampsia query BM25
  correctly returned Maternal Danger Signs and Pre-eclampsia, then padded to `top_n` with
  Diabetes Basics — and the model answered about blood glucose. A weak match is worse than no
  match, because a small model cannot tell which part of its context to ignore. `retrieve()` now
  drops hits below 45% of the top score, a threshold chosen by sweeping real queries.
- **The generation failure was in our data, not the architecture.** The first run trained on
  ~93,000 ranking rows against 80 chat rows. It learned to rank and never learned to write, and
  since we fine-tuned from a Base checkpoint those 80 rows were also the only thing teaching it
  to emit EOS. Correcting the ratio fixed the rambling.
- **A train/inference chat-template mismatch exists.** Training rendered prompts with
  `enable_thinking=False`, appending `<think>\n\n</think>` before the answer; llama.cpp does not
  pass that flag, so the model sees a prefix at inference it never saw in training. We tested
  whether this explained the quality problems — it does not — so it is documented rather than
  presented as a fix.

---

*All medical content in `data/` is derived from public WHO / IMCI / national treatment-guideline material and is provided for clinical decision support only.*
