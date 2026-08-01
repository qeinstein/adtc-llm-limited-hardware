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
4. **Quantization: NOT yet locked — decided empirically by `.github/workflows/quant-sweep.yml`.** Research finding: compression damage is inverted against small models — Qwen3-14B loses ~1% accuracy at 4-bit, **Qwen3-0.6B loses ~10-12%** (measured 4-bit ARC-Challenge ≈ random chance). 8-bit is ~lossless. At 0.6B the RAM cost of Q8_0 vs Q4_K_M is <1 point of score, so this is a run-it-and-see decision, not a convention to follow. Default until the sweep runs: Q4_K_M.
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

## WHAT'S LEFT BEFORE THE RUNPOD FINE-TUNE — one action needed from you

**Trigger `.github/workflows/quant-sweep.yml`** (repo → Actions tab → "Quant sweep — accuracy vs speed vs RAM" → Run workflow → defaults are fine). Takes roughly 1–2 hours unattended (builds the scalar engine, downloads 5 quantized copies of the model, benchmarks each). This is the one remaining unlocked decision (quant level) and needs a GitHub click I can't do myself. Everything else is ready to go without waiting on it — the default (Q4_K_M) is a safe placeholder if you want to start the fine-tune before the sweep finishes; the final `export_gguf.sh` quant is a one-line env var change either way.

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
python scripts/build_healthcare_corpus.py --max-per-dataset 5000

# Clinical chat splits + imatrix calibration corpus
python scripts/prepare_dataset.py

# Train (listwise ranking + clinical SFT + healthcare corpus, one QLoRA run)
python scripts/train_lora.py --base_model Qwen/Qwen3-0.6B-Base

# Merge -> convert -> domain imatrix -> quantize
# (use whatever quant scripts/quant-sweep.yml recommended; Q4_K_M is the default)
QUANT=Q4_K_M bash scripts/export_gguf.sh
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

## Non-code action items for the user (unchanged, still open)

- **Eligibility:** confirm the entry qualifies (reside in an eligible African nation; early-stage/PoC venture <12 months, <$25k raised).
- **Deadline:** Aug 24, 2026, 11:45pm PDT. Submission is frozen to a git commit hash at that point — no post-hoc fixes.
- Record the 2-minute demo video (submission requirement) once the final model is in place.
