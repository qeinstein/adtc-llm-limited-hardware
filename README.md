# Jamii Afya — Offline Clinical Advisor (ADTC 2026)

**Domain:** Healthcare & Medical · **Languages:** English + Kiswahili · **Runtime:** llama.cpp / GGUF, CPU-only, 100% offline

*Jamii Afya* ("community health") is an offline clinical **decision-support** assistant for community health workers and nurses in rural African clinics. It runs on a commodity 8 GB-RAM laptop with no GPU and no internet, answers in the language of the question, grounds answers in a curated WHO/IMCI knowledge base, and always surfaces **danger signs and when to refer**. It is decision support — not a diagnosis, and not a replacement for a clinician.

> Built for the **Africa Deep Tech Challenge 2026 — The Laptop LLM Challenge.**

---

## Why this design wins the rubric

The score is `0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal`. The official `adtc-profiler` measures perf/memory by running **`llama-bench` on the raw GGUF** and accuracy via **lm-eval** — it never runs our app — and the **audit build has all SIMD disabled**. So:

- **Active model:** a fine-tuned **Falcon-H1-1.5B-Deep-Instruct** at the pinned revision, exported as `Falcon-H1-1.5B-Deep-JamiiAfya-Q4_K_M.gguf`. Training uses fp16 LoRA SFT on a T4-class sm75+ worker with Falcon-H1's optimized Mamba/causal-conv path.
- **Accuracy is protected by the fixed trajectory:** assistant-only vectorized SFT, clinical safety rendered both with and without a system prompt, normal clinical and conversational data, public train-split-only MCQA SFT, and additive Kiswahili oversampling.
- **Our numbers survive the audit.** We benchmark against a **scalar (no-SIMD) llama.cpp build** that mirrors the grading VM, so Gate-1 self-reports match the Gate-2 audit within tolerance — a variance-fail trap most teams miss.

Full reasoning and the model A/B (0.6B vs 4B) are in **[REPORT.md](REPORT.md)**.

---

## Architecture

```
Query (EN/SW)
   │
   ▼
BM25 retriever ──▶ extractive compressor ──▶ prompt: [system+few-shot] → [context] → [query]
(src/retriever.py)   (src/compressor.py)              │
   over data/medical_guidelines.json                  ▼
                                          llama.cpp engine (Falcon-H1 1.5B Q4_K_M)
                                          (src/engine.py: q8_0 KV cache, KV prefix
                                           cache; speculative decode disabled) 
                                                       │
                                                       ▼
                                     Safety-framed bilingual advisory + danger signs
```

The RAG stack (`retriever`, `compressor`, `evaluator`, `config`, `manifest`, `score`) is **pure standard library** — it runs and is unit-tested with **no model weights and no heavy deps**.

For clinical safety, a question with no relevant match in the reviewed local corpus receives a fixed bilingual referral-to-clinician response; it is never sent to the model for ungrounded clinical generation.

The repository also contains experimental constrained-answer and cache-augmented prototypes under `src/fact_answer.py` and `scripts/build_cag_cache.py`. They are deliberately not part of the shipped CLI/web path until their artifacts, latency, and clinical behavior are independently validated.

---

## Quickstart

```bash
make setup            # venv + runtime deps (llama-cpp-python, psutil)
make test             # offline test suite — passes WITHOUT model weights

# Try the pipeline before downloading anything (RAG-preview mode):
PYTHONPATH=. python -m src.main --query "Mtoto ana homa kali na kikohozi. Nifanye nini?"

# Full offline advisor (downloads the final Falcon GGUF once):
make model            # ./download_model.sh
make run              # interactive     |  make demo  (runs the metadata test prompts)

# Or the full web UI in ONE command (installs deps, downloads the model,
# launches the server, opens your browser automatically):
make webui             # -> http://localhost:8420
```

`--no-rag` is a safety-path diagnostic: it disables retrieval and returns the
fixed referral response instead of asking the model for ungrounded clinical advice.

### Measure it the way the judges will
```bash
make scalar           # build a no-SIMD llama.cpp matching the audit environment
make bench-audit      # llama-bench -p 512 -n 128 + RSS sampling == profiler parity
make accuracy         # predict S_acc via lm-eval (arc_easy + medical MCQA)
make profiler         # run the official adtc-profiler (Gate-1 self-check)
```

### Reproduce the model (Kaggle T4 x2)
```bash
pip install -r requirements-falcon-production.txt
TORCH_CUDA_ARCH_LIST=7.5 python scripts/verify_falcon_fast_path.py
python scripts/build_accuracy_sft.py --max-per-dataset 250 --fail-on-source-error
python scripts/build_falcon_submission_sft.py
torchrun --standalone --nproc_per_node=2 scripts/train_falcon_submission_v1.py
# then run deterministic selection, merge/export, exact GGUF validation, and profiling
```

---

## Repository layout

```
├── metadata.json            # profiler manifest (strict schema; validated by src/manifest.py)
├── download_model.sh        # fetch and SHA256-verify the active Falcon GGUF
├── MODEL_CARD.md            # active Falcon artifact card
├── REPORT.md                # technical report (official template)
├── requirements*.txt · Makefile · LICENSE (MIT)
├── data/
│   ├── medical_guidelines.json   # bilingual WHO/IMCI knowledge base (BM25 corpus)
│   ├── swahili_eval_set.json     # EN/SW clinical concept-recall eval
│   └── medical_lora_dataset.json # bilingual instruction data for the LoRA
├── src/
│   ├── config.py · retriever.py · compressor.py · rag.py   # RAG (stdlib)
│   ├── engine.py                 # llama-cpp-python CPU serving (the product)
│   ├── evaluator.py · accuracy.py · score.py               # eval + score estimation
│   ├── benchmark.py              # honest bench + profiler-parity mode
│   ├── manifest.py · main.py     # schema self-check + CLI app
├── scripts/                 # fixed Falcon builder/trainer · gate · merge/export · validation/profiler
├── tests/                   # offline tests (no weights needed)
└── model/                   # weights land here (git-ignored)
```

---

## Status & honesty

The active artifact is the exact Q4_K_M Falcon GGUF described in [MODEL_CARD.md](MODEL_CARD.md). Its SHA256 and byte size are recorded by the export, hosting, validation, and download paths. The frozen quality gates use held-out data only; the public MCQA material used for training is restricted to train splits. Everything except the weights is testable offline (`make test`).

*Medical content is derived from public WHO/IMCI/national-guideline material and is for clinical decision support only — not a substitute for a qualified clinician.*
