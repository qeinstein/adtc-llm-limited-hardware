# Jamii Afya — Build Progress & Handoff

_Snapshot of where this ADTC 2026 submission stands and exactly what remains to win.
Domain: healthcare_medical (offline clinical advisor, English + Kiswahili)._

---

## TL;DR — the winning strategy (evidence-based, replaces the inherited 14B plan)

The grader (`adtc-profiler`) does **not** run our app. It runs **`llama-bench` on the raw GGUF** (throughput/RAM) and **lm-eval MCQ on the raw GGUF** (accuracy), on a llama.cpp build with **all SIMD disabled** (scalar). Score = `0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal`.

So the win is a **small Qwen3 model**, fine-tuned on **two complementary data tracks** (the legitimate 50%-accuracy lever), quantized to GGUF:
- **Family = Qwen3** (only small model that is Apache-2.0 **and** officially supports Swahili).
- **Ship the BASE model** (base beats instruct on template-free loglikelihood MCQ).
- **Track A — benchmark-format SFT** (already built): MCQ-completion training on ARC/MedMCQA/PubMedQA/MMLU **train** splits, in the exact lm-eval prompt shape. Targets the automated scoring *format* directly.
- **Track B — broad healthcare knowledge corpus (NOT yet built — see "What remains" #0):** pull as much legitimate open medical/clinical text as we can (open QA datasets, clinical guideline text, consumer-health references, biomedical abstracts) and continue-pretrain / SFT on it so the model is genuinely knowledgeable, not just quiz-shaped. This is what makes answers actually accurate — MCQ-format training alone risks a model that's good at picking A/B/C/D but shallow on real clinical content, which hurts both the hidden-medical-MCQ subset (harder questions need real knowledge, not format tricks) and the judge-qualitative half of S_acc.
- Rules explicitly allow fine-tuning; no anti-contamination clause; never train on any test/validation split.
- **Quant = Q4_K_M** (Qwen is quant-robust) — A/B vs **Q4_0** (simpler unpack → faster on the scalar audit build).
- **Size = the largest Qwen3 that still clears ~15 tps on the scalar build.** RAM is NOT binding (even 4B ≈ 3 GB). Decision hinges on measured accuracy gap vs scalar tps — see "What remains".
- RAG app + Kiswahili + rural-clinic story drive the **judge-qualitative** half of S_acc and the **African-use-case bonus** (separate from the automated metric).

Full rationale + sources are in the agent memory files and REPORT.md.

**Our stated goal: maximize the score the official profiler tool would give us — stretch target >95%.** The scoring formula mixes accuracy + speed + memory efficiency, so pushing all three to the max simultaneously is a real balancing act, not a single number to grind up — but the profiler-driven testing loop below is how we actually track whether we're getting close, instead of guessing.

**Official grading tool (confirmed live):** https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler — this is not just documentation, it's the literal code the organizers run to grade every submission (loads the model, times generation speed, measures RAM, checks thermal throttling, runs the quiz accuracy test, emits a JSON score report). **This becomes the central validation loop for picking our final model** — not vendor benchmarks, not our own approximations:
1. Build a copy of their crippled test CPU setup locally (`scripts/build_llamacpp_scalar.sh`).
2. Install and run their actual profiler tool (`scripts/run_profiler.sh`) against each candidate model — the real grading code, not a guess.
3. Get real numbers: speed, memory, quiz accuracy.
4. Plug into `src/score.py` (mirrors their exact formula) and compare candidates head-to-head.
5. Only then commit to a final model size before the full fine-tune run.

**Independent cross-check on the size-vs-speed risk (2026-08 web research, non-vendor sources):** going from CPU fast-path instructions to the crippled "scalar" mode the grading machine uses typically costs **4–8x speed**, not a minor slowdown. Applying that range to our own measured numbers (Apple Silicon, fast path): 0.6B (~150 tok/s) → an estimated ~19–38 tok/s on the crippled grading machine; 1.7B (~60 tok/s) → an estimated **~8–15 tok/s — right at or below the 15 tok/s cutoff.** This is a rough estimate, not the real number — it's exactly why step 1–2 above (test on an actual matching setup) is non-negotiable before locking the model size. Confirmed official accuracy gap between sizes (Qwen3 Technical Report, arXiv 2505.09388, Base models): MMLU 0.6B=52.8 / 1.7B=62.6 / 4B=73.0 — each size step is a genuine ~10-point accuracy jump, so this is a real trade-off to measure carefully, not obviously won by either side.

**On "just pick the biggest model that fits" — deliberately rejected, with numbers:** a 14B model, compressed as small as reasonably possible, still needs roughly ~8.5GB just to load — already over the 7GB scoring budget and risking the full 8GB physical limit. Going over triggers a crash, which is an **automatic zero / instant disqualification** — not a small point loss. Compressing it further to fit collapses its speed on the crippled CPU (extreme low-bit formats are known to fall apart without fast CPU instructions). So oversizing loses on two independent failure paths (crash-to-zero, or crawl-to-zero-speed), while a well-chosen small/mid model has no such cliff. The likely sweet spot is 1.7B–4B, to be confirmed empirically via the profiler loop above — not assumed.

---

## DONE (committed)

- **Repo fully rebuilt & de-risked.** Deleted the inherited **fake** stack (`inference.py`, `attention.py`, `turboquant_numpy.py`, `memory_manager.py`, `gguf_loader.py`, `pruning.py`, `benchmark_compare.py`, `legacy/`, `plan.md`) and all **fabricated** benchmark numbers.
- **Fixed submission-breaking `.gitignore`** (`*.json` was excluding the RAG corpus/eval/dataset from the repo judges clone).
- **Clean, tested codebase (32 passing tests, ruff-clean, runs without model weights):**
  - `src/config.py`, `src/retriever.py` (stdlib BM25 + **bilingual stopword removal** — fixed a real Swahili mis-retrieval), `src/compressor.py`, `src/rag.py` — the offline RAG stack.
  - `src/engine.py` — llama-cpp-python CPU serving (KV prefix cache, prompt-lookup speculative decoding) for the interactive product/demo.
  - `src/main.py` — bilingual CLI advisor (runs in "RAG-preview" mode even without weights).
  - `src/evaluator.py`, `src/accuracy.py`, `src/score.py`, `src/manifest.py` — eval + score-estimation + schema self-check.
  - `src/benchmark.py` — honest bench with a **profiler-parity** mode (scalar `llama-bench` + RSS sampling that matches the audit, so Gate-1 numbers won't fail the ±25%/±15% Gate-2 variance check).
- **Turnkey model pipeline (`scripts/`)** — runs on free Colab (T4, ~1–3 h) or Udutech GPU:
  - `build_accuracy_sft.py` — builds the MCQ SFT set from public **train** splits in the exact lm-eval prompt shapes. **Validated live** (produces correct `{prompt, completion}` rows).
  - `prepare_dataset.py` — clinical chat splits + imatrix calibration corpus.
  - `train_lora.py` — QLoRA on **Qwen3-1.7B-Base**, completion-only loss, MCQ + clinical mix.
  - `export_gguf.sh` — merge → convert → **domain (EN+SW) imatrix** → Q4_K_M.
  - `build_llamacpp_scalar.sh` — no-SIMD llama.cpp matching the audit; `run_profiler.sh` — official Gate-1 self-check.
  - `mcq_eval.py`, `local_perf_sweep.sh`, `local_accuracy_sweep.sh` — local measurement tools.
- **Rich, medically-QC'd data:** `data/medical_guidelines.json` (32 bilingual WHO/IMCI guidelines), `swahili_eval_set.json` (18 cases), `medical_lora_dataset.json` (80 items).
- **Honest docs:** `README.md`, `REPORT.md` (official template, benchmarks marked *pending measurement*), `LICENSE` (MIT), `Makefile`, `requirements*.txt`, correct `metadata.json` (schema-valid; email set to ogunadetoheeb4@gmail.com).

---

## Measured so far (real, this machine — Apple Silicon, CPU `-ngl 0`, ARM-NEON proxy)

`llama-bench -p 512 -n 128`:

| Model | decode tg128 | prefill pp512 |
|---|---|---|
| Qwen3-0.6B Q4_0 | 151.0 t/s | 769 |
| Qwen3-0.6B Q4_K_M | 149.9 t/s | 590 |
| Qwen3-1.7B Q4_K_M | 59.6 t/s | 204 |

Reads: **0.6B decodes ~2.5× faster than 1.7B**; Q4_0 prefill ~30% faster (favors Q4_0 on the compute-bound scalar build). **These are NEON numbers — x86 scalar will be ~5–8× lower**, so 1.7B likely lands near/below 15 tps and 0.6B above. Ordering transfers; absolute scalar tps must be confirmed on x86.

**Accuracy: NOT yet measured** (was the very next step). Local env is fully ready to measure it (see below).

---

## WHAT REMAINS (ordered — the checklist to finish and win)

0. **Build Track B: broad healthcare knowledge corpus (NOT yet started).** Goal: make the model deeply accurate on real clinical content, not just MCQ-shaped. Plan:
   - **Sources to pull (open/legitimate only, check license per source before use):**
     - Open medical QA: MedQuAD (NIH-derived Q&A), HealthSearchQA, iCliniq/HealthCareMagic-style open dialogues, LiveQA-Med.
     - Reference text: MedlinePlus consumer health topics, WHO fact sheets & guideline documents (many CC-BY / public domain), CDC public health guidance, StatPearls (NCBI Bookshelf, free-access), Wikipedia medical articles (CC-BY-SA).
     - Biomedical literature: PubMed/PMC **open-access subset** abstracts (respect OA license per article), for grounding on mechanisms/terminology (not for verbatim regurgitation).
     - Our own domain: expand `data/medical_guidelines.json`-style WHO/IMCI content further (already have 32; can grow), plus any African MoH public treatment guidelines (Kenya/Tanzania/Nigeria) if openly published.
   - **Method:** a `scripts/build_healthcare_corpus.py` (new) to fetch/clean these into free-text + instruction-style pairs; feed into `train_lora.py` as a third data component (alongside Track A MCQ rows and the existing bilingual clinical chat data), OR run a short continued-pretraining (causal LM, not completion-only-loss) pass on the free-text portion before the SFT stage.
   - **Guardrails:** dedupe against all benchmark test/validation splits (contamination risk), keep a manifest of exact sources for the report (judges/organizers may ask for data provenance), and keep safety framing (danger-signs/refer/"not a diagnosis") consistent — don't let raw scraped text override the safety-tuned behavior from `data/medical_lora_dataset.json`.
   - **Priority order once building:** Track A (small, fast, directly targets the scored format) should still run; Track B is additive and can be scaled to available GPU time/budget — even a partial corpus (MedQuAD + WHO fact sheets + StatPearls summaries) meaningfully deepens real-world accuracy.

1. **Measure accuracy (the decisive number).** Run `mcq_eval.py` (arc_easy + medmcqa, ± pubmedqa/openbookqa) on **Qwen3-0.6B, 1.7B, 4B** (Q4_K_M) to get the real accuracy gap. Env is ready:
   ```
   PY=$HOME/adtc_models/venv311/bin/python
   $PY scripts/mcq_eval.py --model $HOME/adtc_models/gguf/qwen3-0.6b-q4km.gguf --task arc_easy --limit 100
   $PY scripts/mcq_eval.py --model $HOME/adtc_models/gguf/qwen3-1.7b-q4km.gguf --task arc_easy --limit 100
   # repeat for --task medmcqa, and for 4B once downloaded
   ```
2. **Pick model size with data + `src/score.py`.** Rule: ship the largest Qwen3 that clears ~15 tps on the **scalar** build; if 1.7B is well under 15 tps scalar, 0.6B likely out-scores it (perf cap + low RAM > ~10-pt accuracy gap). Feed measured (acc, tps, RAM) into `estimate_total()` under both fixed and relative TPS interpretations.
3. **Confirm scalar tps on x86.** This Mac can't produce x86-scalar numbers. Run `bash scripts/build_llamacpp_scalar.sh` then `make bench-audit` on an x86 box (or accept the audit measures it at Gate 2). This resolves the size decision definitively.
4. **Run the fine-tune (Track A + Track B combined) on GPU** (free Colab T4 ~1–3 h for Track A alone; Track B adds time proportional to corpus size — budget accordingly, or use Udutech credits for a bigger run):
   ```
   pip install -r requirements-dev.txt
   python scripts/build_accuracy_sft.py --max-per-dataset 20000      # Track A: MCQ format
   python scripts/build_healthcare_corpus.py                         # Track B: broad knowledge (to build — step 0)
   python scripts/prepare_dataset.py
   python scripts/train_lora.py --base_model Qwen/Qwen3-1.7B-Base   # (or 0.6B-Base if size flips)
   bash scripts/export_gguf.sh
   ```
   Then **re-measure** base vs base+SFT with `mcq_eval.py` (Track A gain) and with our own `data/swahili_eval_set.json` concept-recall evaluator (Track B gain) to confirm both tracks actually help before committing to a final model.
5. **Quant A/B:** measure acc + scalar tps for Q4_K_M vs Q4_0 on the chosen model; pick the higher S_total.
6. **Finalize:** put the winning GGUF at `model/…` (host it publicly, e.g. HF), point `download_model.sh` at it, set `metadata.json` `model.name`/`parameters_estimate` to match (the profiler checks actual params ≤ claimed×1.2), and **fill REPORT.md benchmark tables with the real measured numbers** (currently marked pending). Correct the RAM figures (1.7B ≈ 1.8 GB peak / S_eff ~74; 0.6B ≈ 1.0 GB / ~86).
7. **Edge-case hardening:** run `make profiler` (official Gate-1 self-check), verify no OOM headroom issues, confirm schema, re-run `make test`.
8. **Record the 2-minute demo video** (a submission requirement) showing the offline advisor answering EN + SW.

---

## How to resume (local environment)

- Repo venv (tests, stdlib code): `./venv` (Python 3.14; `pytest`, `numpy`). `make test` works.
- Measurement env: `$HOME/adtc_models/venv311` (Python 3.11 with `torch`, `lm_eval`, `llama-cpp-python`, `datasets`).
- Built llama.cpp binaries: `$HOME/adtc_models/llama.cpp/build/bin/` (`llama-bench`, `llama-cli`, `llama-server`, `llama-quantize`, `llama-imatrix`).
- Downloaded GGUFs (scratch, NOT in repo): `$HOME/adtc_models/gguf/` — currently `qwen3-0.6b-q4km`, `qwen3-0.6b-q4_0`, `qwen3-1.7b-q4km`. Still to fetch: 4B Q4_K_M, and Q4_0 variants for the quant sweep (bartowski repos; `-Base` GGUFs are NOT on bartowski → convert via `export_gguf.sh`/`convert_hf_to_gguf.py`).
- **Known gotcha:** macOS has no `timeout` command; lm-eval's `--model gguf` server backend is broken (`zip strict` mismatch) → use `scripts/mcq_eval.py` instead (llama-cpp-python direct, validated approach).

---

## Non-code action items for the user

- **Eligibility:** confirm the entry qualifies (reside in an eligible African nation; early-stage/PoC venture <12 months, <$25k raised). Entering via a company that fails these could disqualify regardless of code quality.
- **Optional:** email organizers to resolve two ambiguities — is S_perf fixed-at-15 or relative-to-fastest, and is the African bonus +10 pts or a +15% multiplier. (Our strategy wins under either, but it affects how hard to push size.)
- Host the final GGUF publicly and update `download_model.sh` before the submission commit (the download link is pinned to the commit hash).
