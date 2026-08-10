# Jamii Afya — Offline Clinical Advisor (ADTC 2026)

**Domain:** Healthcare & Medical · **Languages:** English + Kiswahili · **Runtime:** llama.cpp / GGUF, CPU-only, 100% offline

*Jamii Afya* ("community health") is an offline clinical **decision-support** assistant for community health workers and nurses in rural African clinics. It runs on a commodity 8 GB-RAM laptop with no GPU and no internet, answers in the language of the question, grounds answers in a curated WHO/IMCI knowledge base, and always surfaces **danger signs and when to refer**. It is decision support — not a diagnosis, and not a replacement for a clinician.

> Built for the **Africa Deep Tech Challenge 2026 — The Laptop LLM Challenge.**

---

## Why this design wins the rubric

The score is `0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal`. The official `adtc-profiler` measures perf/memory by running **`llama-bench` on the raw GGUF** and accuracy via **lm-eval** — it never runs our app — and the **audit build has all SIMD disabled**. So:

- **Small beats big here.** A 14B model (the plan we inherited) scores ~12/100 on efficiency and near-zero on throughput on a scalar build — it throws away half the score. We ship a fine-tuned **Qwen3-0.6B-Base** (Apache-2.0, Kiswahili-capable) at **Q4_0, 364 MB on disk → 527 MB peak RAM**, measured by the official profiler on a scalar x86 build: **20.33 tok/s (S_perf 100/100), S_eff 92.65/100, no thermal throttling**.
- **Accuracy is recovered, not sacrificed:** WHO/IMCI **RAG grounding** + few-shot + a domain **LoRA** (Kiswahili + answer format), while the base model's general reasoning is preserved for lm-eval.
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
                                          llama.cpp engine (Qwen3-0.6B Q4_0)  
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

# Full offline advisor (downloads the ~460 MB GGUF once):
make model            # ./download_model.sh
make run              # interactive     |  make demo  (runs the metadata test prompts)

# Or the full web UI in ONE command (installs deps, downloads the model,
# launches the server, opens your browser automatically):
make webui             # -> http://localhost:8420
```

### Measure it the way the judges will
```bash
make scalar           # build a no-SIMD llama.cpp matching the audit environment
make bench-audit      # llama-bench -p 512 -n 128 + RSS sampling == profiler parity
make accuracy         # predict S_acc via lm-eval (arc_easy + medical MCQA)
make profiler         # run the official adtc-profiler (Gate-1 self-check)
```

### Reproduce the model (GPU, e.g. Udutech credits)
```bash
make setup-dev                         # torch/transformers/peft/trl/lm-eval/...
python scripts/prepare_dataset.py      # splits + EN/SW imatrix calibration corpus
python scripts/train_lora.py           # QLoRA on Qwen3-0.6B (Kiswahili + format)
bash scripts/export_gguf.sh            # merge → convert → domain imatrix → Q4_0
```

---

## Repository layout

```
├── metadata.json            # profiler manifest (strict schema; validated by src/manifest.py)
├── download_model.sh        # fetch GGUF → model/  (baseline: Qwen3-0.6B; swap to our final model)
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
├── scripts/                 # prepare_dataset · train_lora · export_gguf · build_llamacpp_scalar · run_profiler
├── tests/                   # offline tests (no weights needed)
└── model/                   # weights land here (git-ignored)
```

---

## Status & honesty

Benchmark tables in REPORT.md are **measured**, not projected: throughput and memory come from a real `adtc-profiler` participant run on a scalar (no-SIMD) build, accuracy from held-out test splits. `download_model.sh` fetches our **fine-tuned** GGUF from Hugging Face. REPORT.md §6 documents known limitations and the bugs we found by testing — including one where our own accuracy checker was wrong before the model was. Everything except the weights is testable offline (`make test`).

*Medical content is derived from public WHO/IMCI/national-guideline material and is for clinical decision support only — not a substitute for a qualified clinician.*
