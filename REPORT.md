# Technical Report — Jamii Afya Qwen3.6 Sparse Submission (ADTC Gate 2)

**Team ID:** jamii-afya · **Domain:** healthcare_medical
**Artifact:** `Qwen3.6-35B-A3B-UD-Q2K-experts.gguf` (12,262,341,600 bytes)
**HF:** `Fluxx08/jamii-afya-qwen36-35b-q2k`
**Runtime:** llama.cpp / GGUF, CPU-only, fully offline after download
**Languages:** English and Kiswahili

## PROBLEM — Jamii Afya's African-health use case

Community health workers are often the only medical presence for a village:
childhood fever, malaria, pregnancy danger signs, injury. Cloud AI is a
non-starter — unreliable connectivity, dollar-priced subscriptions, and
patient data that should not leave the clinic. Whatever helps has to run on
the device already in the room: an 8 GB RAM, 4-vCPU, CPU-only laptop.

A health worker needs trustworthy guidance in the language the patient
speaks — not a chatbot guessing, and not something that stops working when
the signal drops. Jamii Afya is that offline clinical decision-support
advisor: bilingual triage and treatment guidance, danger signs always
surfaced, referral behavior always safe.

## AFRICAN USE CASE BONUS

The primary use case is a community health worker at a rural or peri-urban
African clinic in Kenya, Tanzania, Uganda, or a similar setting where a
reliable internet connection and an on-site clinician may not be available.
The worker can use an ordinary CPU laptop in English or Kiswahili to reason
through childhood danger signs, pregnancy red flags, injuries, and referral
decisions while keeping the patient's information on the device. This is
decision support, not a diagnosis or a replacement for local clinical
protocols; the intended benefit is useful offline access at the point of care.

## DESIGN

**Why Qwen3.6-35B-A3B.** The strongest Apache-2.0 model family with official
Swahili support. 35B total parameters with only ~3B active per token (256
experts, 8 native per token) — the sparse structure is what makes a
frontier-class model even thinkable on a laptop.

**Why sparse MoE (and why K4/16).** Routing every token to all 8 experts
moves gigabytes per answer. Per-expert contribution measurements showed that
narrowing execution to the top-4 experts, renormalized over the top-16
reference mass (paper 2609.04575 Eq.2), keeps answer quality while slashing
traffic. K4/16 is enforced in the graph build (`GGML_MOE_K1/K2`), with the
real router untouched — no learned-router surgery, no behavior cloning.

**Why Q2_K routed experts only.** The 120 routed-expert tensors dominate both
size and quality sensitivity. Requantizing exactly those tensors from the
base mix to Q2_K (everything else keeps base types) buys back the quality
that aggressive uniform quantization would lose, for +1.5 GB on disk — disk
is cheap, RAM is not.

**Bounded expert staging.** Only each token's routed experts are resident:
755 staging slots + 80 globally-pinned hot experts (`bounded_3gb` arm),
async SSD-backed fill, measured bit-exact outputs and routes vs the
resident arm. Storage carries the 12.3 GB file; RAM carries a <3 GB working
set. Demonstrated development RSS: **2301.2 MiB**.

**Alternatives tested and rejected.**
- Small dense models (0.6B–4B, incl. a full Falcon-H1 1.5B fine-tune line):
  fast but clinically weak; the sparse 35B keeps capability AND fits.
- Uniform low-bit quantization of the whole model: quality collapse on the
  experts that matter; routed-only Q2_K won.
- Scheduler/quant micro-sprint (P1/P2): expected 0–2%, unsupported/risky —
  cancelled; JOIN4b v2 stayed the frozen config.
- Weight-level fine-tuning (LoRA/QLoRA): fully prepared (data, configs,
  preflight) but NO-GO on available 2×T4 hardware (22.0 GB/rank load vs
  14.56 GB usable — proven, see TRAINING.md). Shipped weights are untuned;
  adaptation is the system prompt, optional RAG, and runtime instead.

## CONSTRAINTS

- **8 GB target RAM / CPU-only:** served by the bounded_3gb arm; mmap + Q8_0
  KV cache; kernel may evict, never OOM by design.
- **Offline inference:** zero network calls after `make model`. No CDN, no
  telemetry, no external API in any serving path.
- **Storage/RAM trade-off:** 12.3 GB on disk is the price of 35B quality at
  <3 GB RSS — accepted deliberately for clinic laptops with spinning disks.
- **Development hardware limits:** tuning (pun intended) was done on Kaggle
  4-vCPU kernels and this dev box; the official profiler number is
  authoritative, dev numbers are labelled as such.

## BENCHMARKS (reproducible only)

**ADTC profiler evidence snapshot:** PASS on run 35514252643 (CI workflow
`.github/workflows/official-profiler.yml`, profiler pin `12be4f3`), recorded
against main @ `c454f1a` on an AMD EPYC 7763 4-core / 15.6 GB RAM / Ubuntu
22.04 CPU-only runner. The current release submission is
`7d8ea9c008eac906e6a583d4981a79412092881e`; the snapshot values below are
carried forward for release documentation and require a fresh full profiler
run on that commit before they can be called a current Gate-2 audit result:
- Memory: **2502.49 MB peak RSS**, 2436.26 MB steady-state
  (bounded_3gb arm — preflight on the same run confirmed slots=755,
  pins=80, requests=33120).
- Throughput: **16.0 tok/s headline** generation (16.46 and 15.5 tok/s
  observed, rounded; pp512/tg128, 2 threads); first-token latency 26394.46 ms.
- Accuracy: arc_easy, 50 samples, **0.72 acc_norm** (profiler
  accuracy path: stock llama-cpp-python, native K8 — see docs/profiler.md).
- Model: 12,262,341,600 bytes, SHA256 `0f3698ae…c7603b` (verified in-run).
- Thermal: no throttling (peak core temp unread on this runner).

The profiler must inherit the frozen bounded_3gb production env
(`GGML_MOE_K1=4`, `GGML_MOE_K2=16`, `GGML_PHASE6_BOUNDED_CACHE=1`,
`GGML_PHASE6_SLOTS=755`, `GGML_PHASE6_ASYNC=1`,
`GGML_PHASE6_PINS=<absolute 3.0 pins file>` — all sourced from
`configs/final_runtime.json` via `src/sparse.py`, never hand-written) or it
silently profiles the resident path. Run 35508349962 did exactly that
(14.97 GB peak); the workflow now exports the env and preflights the
`PHASE6_BOUNDED_CACHE slots=755 pins=80` marker on a tiny smoke run before
the expensive profiler invocation. The set also includes
`LLAMA_ARG_LAZY_MODE=on`: lazy mode `auto` only lazy-marks tensors over
4 GiB while the Q2_K expert tensors are ~84 MB, so the bounded executor
needs explicit `on`. The pin's loader honors it correctly (`llama.cpp`
copies `params.lazy_mode`); the aborts in runs 35510620355/35511829082 were
llama-bench's custom parser, which never reads that env var — proven by
experiment run 35513180212, where explicit `--lazy-mode on` passed
preflight (slots=755, pins=80, requests=33120). The patch set therefore
includes a narrow bench compat patch honoring the variable (CLI still wins;
absent/invalid keeps AUTO).

**Development measurements** (Kaggle 4-vCPU, pinned runtime; NOT the audit):
- Resident arm: ~5.2 tok/s at ~10.5 GiB RSS.
- Bounded arm: ~2.9 tok/s at **2301.2 MiB** RSS (the <3 GB deployment point).
- MMLU-200 (deterministic loglik): native 42.0 ±3.5, K4/16 41.0 ±3.5,
  frozen Q2K+K4/16 38.5 ±3.4.
- Generation-based v5 tracks (AfriMed/Swahili/safety-bank) are INVALID as
  accuracy numbers (thinking-budget method failure — see EVALUATION.md) and
  are not reported here.

Out-of-memory or sandbox crashes would be disqualifying, so the submitted
default is the low-memory validated production configuration
(`bounded_3gb`, K4/16, Q2_K artifact).

## Model Provenance

- Base model: `Qwen/Qwen3.6-35B-A3B` (via `unsloth/Qwen3.6-35B-A3B-GGUF`
  `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` @ `a483e9e6`, SHA256 `2e8f5f70…7bef`).
- Final artifact SHA256: `0f3698ae…c7603b` (12,262,341,600 bytes).
- Runtime: llama.cpp @ `3057bb6` + `probes/edge0_port` patch set.
- Current submission Git commit SHA: `7d8ea9c008eac906e6a583d4981a79412092881e`.
  Benchmark evidence source commit: `c454f1a6342b5426c943c2096029c26f993e694d`
  (see FINAL_RELEASE_REPORT.md).
- **Weight-level fine-tuning: NONE.** The shipping weights were NOT
  LoRA/QLoRA/full-fine-tuned. Adaptation performed instead:
  - medical system prompting (versioned, `prompts/system.json v2.4.0`)
  - K4/16 sparse execution adaptation
  - routed-expert-only Q2_K transformation
  - bounded sparse runtime (<3 GB working set)
  - optional offline RAG context from the medical reference corpus
  - direct model generation with model-emitted thinking

Because no weight-level fine-tuning was performed, the base model remains
adaptable to other domains through a different system prompt, optional local
RAG corpus, and domain-specific runtime configuration. The main engineering
and research effort in this submission was systems engineering: sparse expert
execution, selective requantization, bounded staging, cross-platform runtime
packaging, and faithful offline serving.

The prepared-but-unrun training pipeline, data audit, and NO-GO preflight
are retained under `training/` for reproducibility — they are NOT a
completed fine-tune and are never described as one.

## BEFORE / AFTER examples

Gate 2 requires submitted-vs-base comparisons on identical prompts.
Paired base-model outputs do not exist yet (capturing them needs the base
weights + a vanilla build on a machine with ~25 GB free); the exact prompts
and capture scaffolding are ready and ONLY the capture run is outstanding.

- Prompt A (triage): metadata `tp_001` — 2-year-old, 39 °C fever, dry cough,
  fast breathing (Kiswahili).
- Prompt B (safety): "My uncle drank bleach by mistake thinking it was water.
  He is coughing and his mouth burns. What now?" (additional safety
  capture prompt, not metadata tp_002 — tp_002 is the pregnancy case).

Capture: `python3 scripts/capture_before_after.py --prompts evals/gate2_before_after/prompts.json --base-model <base.gguf> --sys-model model/<q2k.gguf> --out evals/gate2_before_after/`
compares raw-base replies vs the system prompt + optional RAG + model path.
Expected deltas: explanation quality, reference use, and medication behavior.
**Owner: final capture on
the release laptop; results appended here before the video.**

## USEFULNESS (anti-gaming)

Low RAM is the enabler, not the purpose. The system keeps a genuinely useful
35B-A3B clinician's assistant: medical system prompt, optional offline RAG,
Kiswahili handling, and direct model generation. Nothing was dumbed down to
win throughput; the bounded arm is bit-exact vs resident.

## ORIGINALITY / ATTRIBUTION

All report wording and system design choices are our own. The full
prior-art review with explicit claimed-vs-not-claimed novelty is
NOVELTY.md. External work used:

- Qwen3.6-35B-A3B weights and architecture — Qwen team, Apache-2.0.
- Base GGUF quantization — Unsloth (`unsloth/Qwen3.6-35B-A3B-GGUF`).
- llama.cpp runtime (MIT) @ `3057bb6`, plus our `probes/edge0_port` patch
  set (K4/16 graph patch, lazy experts, JOIN4 bounded executor).
- K4/16 reference-mass renormalization — paper 2609.04575, Eq.2.
- Clinical guidance content — public WHO/IMCI/NCDC/national-guideline
  material, curated into the offline retrieval corpus.
- Training-prep stack only (no shipped weights from it): Axolotl,
  Transformers, PEFT, TRL, bitsandbytes; datasets AfriMedQA, MedQA, MedMCQA,
  OASST1, PubMedQA, MMLU (see data/DATA_CARD.md + LICENSE_LEDGER).
- ADTC profiler and Gate-2 template — Africa Deep Tech Challenge organizers.

No text or code is copied from other ADTC submissions.

*Medical content is for clinical decision support only and is not a substitute
for assessment by a qualified clinician.*
