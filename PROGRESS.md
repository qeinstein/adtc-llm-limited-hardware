# Jamii Afya — Build Progress & Handoff

_Snapshot of where this ADTC 2026 submission stands and exactly what remains to win.
Domain: healthcare_medical (offline clinical advisor, English + Kiswahili)._

---

## TL;DR — the winning strategy (evidence-based, replaces the inherited 14B plan)

The grader (`adtc-profiler`) does **not** run our app. It runs **`llama-bench` on the raw GGUF** (throughput/RAM) and **lm-eval MCQ on the raw GGUF** (accuracy), on a llama.cpp build with **all SIMD disabled** (scalar). Score = `0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal`.

So the win is a **small Qwen3 model**, fine-tuned on **public benchmark train splits** (the legitimate 50%-accuracy lever), quantized to GGUF:
- **Family = Qwen3** (only small model that is Apache-2.0 **and** officially supports Swahili).
- **Ship the BASE model** (base beats instruct on template-free loglikelihood MCQ) + our **MCQ-completion SFT** on ARC/MedMCQA/PubMedQA/MMLU **train** splits (rules explicitly allow fine-tuning; no anti-contamination clause; never train on test/val).
- **Quant = Q4_K_M** (Qwen is quant-robust) — A/B vs **Q4_0** (simpler unpack → faster on the scalar audit build).
- **Size = the largest Qwen3 that still clears ~15 tps on the scalar build.** RAM is NOT binding (even 4B ≈ 3 GB). Decision hinges on measured accuracy gap vs scalar tps — see "What remains".
- RAG app + Kiswahili + rural-clinic story drive the **judge-qualitative** half of S_acc and the **African-use-case bonus** (separate from the automated metric).

Full rationale + sources are in the agent memory files and REPORT.md.

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

1. **Measure accuracy (the decisive number).** Run `mcq_eval.py` (arc_easy + medmcqa, ± pubmedqa/openbookqa) on **Qwen3-0.6B, 1.7B, 4B** (Q4_K_M) to get the real accuracy gap. Env is ready:
   ```
   PY=$HOME/adtc_models/venv311/bin/python
   $PY scripts/mcq_eval.py --model $HOME/adtc_models/gguf/qwen3-0.6b-q4km.gguf --task arc_easy --limit 100
   $PY scripts/mcq_eval.py --model $HOME/adtc_models/gguf/qwen3-1.7b-q4km.gguf --task arc_easy --limit 100
   # repeat for --task medmcqa, and for 4B once downloaded
   ```
2. **Pick model size with data + `src/score.py`.** Rule: ship the largest Qwen3 that clears ~15 tps on the **scalar** build; if 1.7B is well under 15 tps scalar, 0.6B likely out-scores it (perf cap + low RAM > ~10-pt accuracy gap). Feed measured (acc, tps, RAM) into `estimate_total()` under both fixed and relative TPS interpretations.
3. **Confirm scalar tps on x86.** This Mac can't produce x86-scalar numbers. Run `bash scripts/build_llamacpp_scalar.sh` then `make bench-audit` on an x86 box (or accept the audit measures it at Gate 2). This resolves the size decision definitively.
4. **Run the accuracy SFT (the 50% lever) on GPU** (free Colab T4 ~1–3 h, or Udutech):
   ```
   pip install -r requirements-dev.txt
   python scripts/build_accuracy_sft.py --max-per-dataset 20000
   python scripts/prepare_dataset.py
   python scripts/train_lora.py --base_model Qwen/Qwen3-1.7B-Base   # (or 0.6B-Base if size flips)
   bash scripts/export_gguf.sh
   ```
   Then **re-measure** base vs base+SFT with `mcq_eval.py` to confirm the SFT lifts the scored metric (it should — it trains the exact task).
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
