# Jamii Afya — Build Progress & Handoff

_Domain: healthcare_medical (offline clinical advisor, English + Kiswahili). Branch: `jamii-afya-rebuild`._

---

## DECISIONS — LOCKED (backed by measurement + research, not guesses)

1. **Model family: Qwen3.** Only small model family that is Apache-2.0 (commercial-safe) **and** officially supports Swahili.
2. **Model size: Qwen3-0.6B.** The grader runs `llama-bench` on a CPU build with all speed instructions disabled ("scalar"). Measured on real x86 hardware (GitHub Actions, AMD EPYC 7763, true scalar build):

   | Model | decode tok/s | peak RAM | Speed score (of 30) | RAM score (of 20) | **Banked before any fine-tuning** |
   |---|---|---|---|---|---|
   | **0.6B** | 15.6 | 598 MB | **30.0** | **18.3** | **48.3 / 50** |
   | 1.7B | 6.3 | 1395 MB | 12.7 | 16.0 | 28.7 / 50 |
   | 4B | 2.8 | 2600 MB | 5.6 | 12.6 | 18.2 / 50 |
   | 8B | 1.5 | 5034 MB | 3.1 | 5.6 | 8.7 / 50 |

   For 1.7B to beat 0.6B it would need to be ~39 accuracy points better — the real gap is ~10. Not close. **0.6B is correct.** Caveat: our speed margin over the 15.0 tok/s cutoff is only ~4% — thin, watch it.
3. **Ship the BASE checkpoint** (`Qwen/Qwen3-0.6B-Base`), not Instruct. The grader scores multiple-choice by ranking answer options via raw loglikelihood, **no chat template** — base models score higher than instruct in that regime (measured in the literature: base ARC 60.5 vs instruct 51.7 on a comparable model).
4. **Quantization: LOCKED to Q4_0**, decided empirically by `.github/workflows/quant-sweep.yml` (real run, AMD EPYC 9V74, x86 scalar, arc_easy n=200):

   | quant | size | decode tok/s | peak RAM | acc_norm | clears 15 tok/s? | est S_total |
   |---|---|---|---|---|---|---|
   | **Q4_0** | 448 MB | **19.1** | 584 MB | 52.5 | **✓ yes, +27% margin** | 74.6 |
   | Q4_K_M | 462 MB | 14.3 | 598 MB | 56.0 | ✗ no | 74.8 |
   | Q5_K_M | 526 MB | 13.2 | 662 MB | 56.5 | ✗ no | 72.8 |
   | Q6_K | 594 MB | 12.9 (slowest) | 730 MB | 59.0 (best acc) | ✗ no | 73.4 |
   | Q8_0 | 768 MB | 14.2 | 904 MB | 58.0 | ✗ no | 74.9 |

   All 5 land within ~2 points of each other on estimated total score — the accuracy differences (52.5–59.0 acc_norm on a 200-question sample) are within noise. **Q4_0 is the only one that clears the 15 tok/s scoring cutoff with real margin**; every other quant measured BELOW 15 tok/s on this run. We've now measured this same model at meaningfully different absolute speeds across two different test runs (14.3–19.1 tok/s for nominally similar setups) — real hardware variance exists, and the actual grading laptop is an unknown wildcard in that range. Shipping anything sitting at/below the cutoff risks losing a large chunk of the speed score to bad luck; Q4_0's margin protects against that. Chosen over the marginally-higher-accuracy options because that accuracy edge is noise-level while the speed risk is real and asymmetric (losing the 15-tok/s cap costs far more than the ~2-6 accuracy points at stake).
5. **Training objective: listwise ranking, not plain gold-only SFT.** A 2026 study benchmarking Qwen3-0.6B/1.7B/4B/8B specifically found gold-only fine-tuning ("make the right answer more likely") is the *worst* of the objectives tested at sub-3B scale, and that training the model to **rank the correct choice above the distractors** (length-normalized to match the grader's `acc_norm` metric exactly) is measurably better. Implemented in `scripts/train_lora.py`.
6. **Two accuracy tracks, trained together:**
   - **Track A** (`scripts/build_accuracy_sft.py`): public MCQA **train** splits (ARC, MMLU-aux, MedMCQA, MedQA, PubMedQA, OpenBookQA, HeadQA) in the exact grader prompt shape, with **balanced answer-letter permutation** on letter-format items (MedMCQA/MedQA/MMLU) — small models carry a real measured bias toward certain letters; this trains it away. Never touches any test/validation split (rules explicitly allow fine-tuning; no anti-contamination clause exists, but training on eval data would be real contamination).
   - **Track B** (`scripts/build_healthcare_corpus.py`): broad open healthcare knowledge (MedQuAD, EPFL clinical guidelines, medical flashcards, WikiDoc) as plain continued-pretraining text. Matters because **judges download the raw GGUF and run it standalone in LM Studio/Ollama — our RAG app is not in that loop.** The model's own knowledge is what gets read.
7. **The shipped GGUF's chat template must have "thinking" disabled.** Qwen3 is a hybrid-reasoning model; a judge opening it in LM Studio and getting a wall of `<think>` rambling would score us down. Baked into training via `enable_thinking=False` on all clinical-chat rows — don't rely on a flag a judge will never type.
8. **The accuracy score (S_acc, 50% of total) is largely judge-owned, not purely automated** — official sources conflict, but two of four describe it as a qualitative panel score based on "cross-disciplinary integration, software UX, and live defense," with the automated quiz as one input among several. Also: **Swahili competence claims a ×1.15 multiplier on the panel score** (a separate bonus from the general "African use case" +10 points). Net effect: Track B + product quality + documentation likely matter as much or more than squeezing the last few quiz points.

---

## BUILT & VALIDATED TODAY

- **Model/quant/size finalized** across `metadata.json`, `download_model.sh`, `scripts/train_lora.py`, `scripts/export_gguf.sh`, `README.md`, `REPORT.md` — all now reference Qwen3-0.6B with real measured numbers (598 MB peak RAM, S_eff ≈ 91.7), no stale 1.7B references left.
- **`scripts/build_accuracy_sft.py` rewritten** to emit choice-list rows `{context, choices, gold, format}` (not flat prompt/completion) with balanced letter-permutation augmentation. **Validated live** against real HF datasets (ARC, MedMCQA) — produces correct, well-formed, appropriately-balanced output.
- **`scripts/train_lora.py` rewritten** with a custom `Trainer` implementing the listwise char-length-normalized ranking loss (+ small auxiliary NLL term for calibration), unified with completion-only clinical chat SFT and optional Track-B causal-LM loss, all through one QLoRA run. Core scoring math (logits → log-softmax → per-choice-token gather) is the **same algorithm already proven correct** in `scripts/mcq_eval.py` (verified end-to-end earlier: correctly ranks "Paris" above "banana" for "capital of France," and produced a sane 51–57% arc_easy accuracy on real questions — well above the 25% chance floor a scoring bug would produce).
- **`scripts/build_healthcare_corpus.py` (new, Track B)** — pulls MedQuAD/EPFL-guidelines/medical-flashcards/WikiDoc, skip-on-failure per source. Smoke-tested live (MedQuAD sample pulled correctly).
- **Two real bugs found and fixed by testing against ground truth, not by inspection:**
  1. `scripts/mcq_eval.py` originally used `llama-cpp-python`'s `create_completion(echo=True, logprobs=N)`, which returns **misaligned, wrong logprobs** (verified: it ranked "banana" above "Paris" as the capital of France). Rewritten to read raw logits directly. This is the same bug class the ADTC profiler's own authors hit and rewrote around — confirms it's a real, known trap, not a one-off mistake.
  2. `scripts/build_llamacpp_scalar.sh` and `scripts/export_gguf.sh` both had a bare `-j` (unlimited parallel compile jobs) in the cmake build step — this OOM-killed our first real CI run (`gmake: *** Terminated`, exit 143). Both now cap parallelism to a safe bounded value.
- **Two GitHub Actions workflows, run on real x86 hardware (no local laptop needed):**
  - `.github/workflows/scalar-speed-test.yml` — speed + memory across model sizes (already run successfully; produced the numbers in the decision table above).
  - `.github/workflows/quant-sweep.yml` — speed + memory + **accuracy** + estimated total score across Q4_0/Q4_K_M/Q5_K_M/Q6_K/Q8_0 for Qwen3-0.6B. **Built, pushed, not yet triggered.**
- Full local suite still green: `make test` (32 passing), ruff clean across `src/`, `scripts/`, `tests/`, metadata schema-valid.

---

## Quant sweep — DONE, decision locked (Q4_0)

`.github/workflows/quant-sweep.yml` ran to completion; results and reasoning are in the quant decision above. `export_gguf.sh` and `download_model.sh` both default to Q4_0 now. RunPod training is in progress on the Qwen3-0.6B-Base + listwise ranking + Track A/B pipeline described below.

---

## RunPod training — status (in progress)

- 2 full epochs, listwise ranking + Track A/B, batch_size 4 / grad_accum 8 (effective 32), max_len 256, gradient checkpointing on — the config that survived two rounds of OOM debugging (see "Known limitations" and commit history for the full saga).
- Loss trend (real, logged): ~1.37 → 1.34 → 1.28 → 1.21 → 1.18 → 1.13 → crossed epoch 1.0 around loss 0.98 → currently plateaued in the ~0.90–1.03 band through epoch ~1.5, no further downward drift. This is the expected shape for epoch 2 of a short LoRA run (most learning happens in epoch 1; epoch 2 consolidates) — gradients stayed bounded throughout, no instability or divergence observed at any point.
- ETA from most recent log: ~7561/10014 steps (76%, epoch 1.51) at ~2.98s/it → roughly 2h remaining, ~8h09m total run time.
- **Next once training finishes:** export/quantize (`QUANT=Q4_0 bash scripts/export_gguf.sh`), re-run `mcq_eval.py` against the pre-training baseline (51–57% arc_easy) to confirm real improvement, host the final GGUF (need a hosting decision — Hugging Face repo is the default plan, not yet resolved), update `download_model.sh` / `metadata.json` to point at it, run `bash scripts/run_profiler.sh` (Gate-1 self-check), fill in `REPORT.md` with real final numbers, record the demo video.

---

## Web UI — built and shipped (judge/demo experience, not the scored path)

Two-page offline web app (`src/webapp.py`, FastAPI + vanilla JS, zero CDN/external assets):

- **`/`** — Anthropic-style landing page (`src/static/index.html`): problem statement, our approach, and an honest "what we learned" section (the scalar-CPU discovery, the model-size course-correction, the scoring-bug catch, the OOM debugging, the vectorization win), with a CTA into `/chat`.
- **`/chat`** (`src/static/chat.html`) — the real interactive advisor: RAG-grounded replies with source attribution, telemetry (tok/s, elapsed time), a "careful mode" toggle (prompt-based step-by-step reasoning hint — see thinking-mode note below), and genuine multi-turn conversation memory via a new `MedicalLLMEngine.chat()`/`stream_chat()` path that takes the full message history instead of one-shot prompts. Verified end-to-end with a real two-turn test (pronoun correctly resolved from prior turn).
- **Real bug found and fixed along the way:** `prompt_lookup_decoding` (speculative decoding) crashes with a broadcast-shape error specifically on long, chat-formatted prompts (system + few-shot + RAG context) — not on short test prompts, which is why an initial smoke test missed it. Now defaults to `False` in `MedicalLLMEngine.__init__`.
- Committed and pushed to both `main` and `jamii-afya-rebuild`.

---

## Is the training data enough? (assessed, and acted on)

- **Track A (public benchmark MCQA): solid.** ~93k rows across ARC/OpenBookQA/HeadQA (full train splits, they're small) + capped 20k each from MMLU-aux/MedMCQA/MedQA/PubMedQA (each has 10k-270k available). Diverse, real, no changes needed.
- **Track B (broad healthcare corpus): was thin, now fixed.** Default was only 5,000 chunks against 10k-270k available per source — bumped to 15,000, and added a 5th source (raw PubMedQA abstracts, 273k available) for more depth (`scripts/build_healthcare_corpus.py`).
- **The real gap: almost ALL of our data is English-only.** No public Swahili medical MCQA dataset exists, and none of the Track-B sources are multilingual. Since Swahili competence claims a real scoring bonus, this was worth closing.
- **The fix (novel, built today): `scripts/generate_synthetic_data.py`.** Uses a strong "teacher" model to expand our own 32 hand-verified clinical guidelines into many bilingual (EN + Kiswahili) examples — chat Q&A, MCQ (auto-merges into Track A's format), and free text (auto-merges into Track B's format). Grounded strictly in our own already-reviewed source text (the teacher is told to only elaborate on given facts, not invent new clinical claims), which is what makes this safe for a medical tool instead of just hallucination bait. This is genuine knowledge distillation — the same technique research identified as one of the highest-leverage ways for a small model to punch above its size.
- **This needs an API key you'll supply** (OpenAI or any OpenAI-compatible provider) — that's the one external dependency in the whole pipeline I can't get around. If you don't have one, tell me and we'll figure out an alternative (e.g. a different provider, or skip this step — Tracks A+B alone are still solid).
- **Also fixed while looking at this:** the training script originally ran one slow forward-pass per answer choice; it now batches all choices of a question into a single forward pass (~4x fewer forward passes on the MCQA rows, which are most of the data) — verified correct against the original with a synthetic-tensor test. This matters directly for "is the data enough": more data only helps if training actually finishes in reasonable time/cost on a rented GPU.

---

## RUNPOD — the exact command sequence, ready to paste

**Pod:** RTX 4090 or A5000 (24 GB VRAM) on Community Cloud, PyTorch template, ~60–100 GB volume disk. Don't pay for A100/H100 — total overkill for a 0.6B QLoRA run. Estimated cost: a few dollars total.

```bash
git clone -b jamii-afya-rebuild <your-repo-url>
cd adtc-llm-limited-hardware
pip install -r requirements-dev.txt

# Track A: quiz-format training data (public train splits only)
python scripts/build_accuracy_sft.py --max-per-dataset 20000

# Track B: broad healthcare knowledge corpus
python scripts/build_healthcare_corpus.py --max-per-dataset 15000

# OPTIONAL but recommended: bilingual EN/SW distillation from our own verified
# guidelines. Uses OpenRouter (auto-detected from this env var; script defaults
# to google/gemini-2.0-flash-001, good multilingual quality + cheap for ~32 calls
# — override with --model, e.g. anthropic/claude-3.5-sonnet for higher quality)
export OPENROUTER_API_KEY=sk-or-...
python scripts/generate_synthetic_data.py --per-guideline 6
cat output/synthetic_mcqa.jsonl >> output/accuracy_sft.jsonl
cat output/synthetic_corpus.jsonl >> output/healthcare_corpus.jsonl
python -c "import json; a=json.load(open('data/medical_lora_dataset.json')); b=json.load(open('output/synthetic_clinical_chat.json')); json.dump(a+b, open('output/clinical_combined.json','w'), ensure_ascii=False, indent=2)"

# Clinical chat splits + imatrix calibration corpus
python scripts/prepare_dataset.py

# Train (listwise ranking + clinical SFT + healthcare corpus, one QLoRA run)
# Use --clinical_file output/clinical_combined.json instead if you ran the
# synthetic-data step above.
python scripts/train_lora.py --base_model Qwen/Qwen3-0.6B-Base

# Merge -> convert -> domain imatrix -> quantize
# (Q4_0 — chosen by the real sweep results, see above)
QUANT=Q4_0 bash scripts/export_gguf.sh
```

Output lands at `model/Qwen3-0.6B-<QUANT>.gguf` — that's the final artifact. Copy it out of the pod, then:

1. Host it somewhere public (e.g. a Hugging Face repo you control).
2. Point `download_model.sh`'s `MODEL_URL` at it.
3. Update `metadata.json` → `model.quantization` / `parameters_estimate` to match exactly (the profiler checks actual params ≤ claimed × 1.20 — don't understate size).
4. Re-run `mcq_eval.py` on the new GGUF (arc_easy + medmcqa) to confirm the fine-tune actually improved on the pre-fine-tune baseline — if it didn't, something's wrong and don't ship it blind.
5. Run `bash scripts/local_perf_sweep.sh` / the quant-sweep workflow one more time on the FINAL file to get real numbers for `REPORT.md`.
6. `bash scripts/run_profiler.sh` — the official Gate-1 self-check — before considering this done.

**Stop and tell me if training crashes, loss doesn't go down, or the exported GGUF fails to load** — those are worth debugging together rather than guessing blind on a rented GPU clock.

---

## Known limitations / honest caveats

- Speed margin on 0.6B is thin (15.6 vs 15.0 cutoff, measured on a GitHub Actions shared runner — the real grading laptop could be faster or slower). If the quant sweep or final export pushes speed down, this is the first thing to check.
- The listwise ranking trainer processes one MCQA item as its own small forward-pass batch (correctness prioritized over throughput) — this is slower per-example than plain SFT. Budget more wall-clock than a naive token-count estimate would suggest; not a blocker on a rented GPU, just don't be surprised.
- Track B dataset sources (MedQuAD/EPFL-guidelines/flashcards/WikiDoc) are real and verified to exist, but were only smoke-tested at tiny scale locally — first full run on RunPod is the real test of the complete pipeline end-to-end.
- The automated-vs-judge split inside S_acc is genuinely unpublished by the organizers (confirmed by direct research — not an oversight on our part). Emailing `challenge@africadeeptech.org` remains a free, real information edge no other team likely has.

---

## FUTURE ENHANCEMENT (not urgent, legitimate, queued): real "thinking mode"

Right now our fine-tuned model has no genuine step-by-step reasoning mode — we started from the plain (non-chat-trained) checkpoint specifically to maximize the automated quiz score, and every training example we used was direct question-then-answer, never a reasoning trace. So there's no real capability to switch on, only a fake instruction-based approximation (the web UI's "careful mode" checkbox, which just asks the model to reason via a prompt hint — works a bit, but isn't the same as a model actually trained for it).

**This can be added later, legitimately, with no rules conflict:** it doesn't touch the automated score at all (that's scored on raw answer-ranking regardless of reasoning style), and for the judge-facing side it's a pure upside (a transparency/trust feature, not a violation of anything). The real way to add it: a follow-up fine-tuning round that includes actual reasoning-trace examples as training targets (e.g., teacher-model-generated step-by-step clinical reasoning, same distillation technique already used for the bilingual data) so the model genuinely learns to produce useful `<think>` content when asked, rather than just being told to. This is a real, non-trivial second training iteration — not a quick toggle — so it's queued as a future enhancement after the current model is shipped and validated, not something to interrupt the current pipeline for.

---

## Non-code action items for the user (unchanged, still open)

- **Eligibility:** confirm the entry qualifies (reside in an eligible African nation; early-stage/PoC venture <12 months, <$25k raised).
- **Deadline:** Aug 24, 2026, 11:45pm PDT. Submission is frozen to a git commit hash at that point — no post-hoc fixes.
- Record the 2-minute demo video (submission requirement) once the final model is in place.
