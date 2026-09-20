# Jamii Afya — Offline Clinical Advisor (ADTC 2026)

**Domain:** Healthcare & Medical · **Languages:** English + Kiswahili · **Runtime:** llama.cpp / GGUF, CPU-only, 100% offline

*Jamii Afya* ("community health") is an offline clinical **decision-support** assistant for community health workers and nurses in rural African clinics. It runs a 35B-parameter sparse mixture-of-experts in under 3 GB of working RAM on a commodity 8 GB laptop — no GPU, no internet — answers in the language of the question, and can add relevant local reference material through offline RAG. It is decision support — not a diagnosis, and not a replacement for a clinician.

> Built for the **Africa Deep Tech Challenge 2026 — The Laptop LLM Challenge.**

---

## Run it

```bash
git clone https://github.com/qeinstein/adtc-llm-limited-hardware.git
cd adtc-llm-limited-hardware
make model            # download + SHA256-verify the 12.3 GB GGUF (once)
make webui            # build runtime, launch UI -> http://localhost:8420
```

`make webui` is the only normal command: it creates a venv, installs deps,
verifies the model, builds the pinned sparse runtime if needed, starts the
backend + frontend, waits for readiness, and prints the URL. Ctrl+C stops
the frontend (a lingering backend `llama-server` may need a manual kill).

The runtime cache and launcher support Linux, macOS, and native Windows. On
Windows, use GNU Make from Git Bash for the same command, or run the
platform-neutral setup directly from PowerShell:

```powershell
py -3 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python scripts\download_model.py
venv\Scripts\python scripts\build_runtime.py
venv\Scripts\python -m uvicorn src.webapp:app --host 0.0.0.0 --port 8420
```

The Windows build requires Git, CMake, and a C/C++ toolchain visible to CMake
(Visual Studio Build Tools or clang-cl). The default bounded arm is used on
all three platforms; set `ADTC_SPARSE_ARM=resident` only when the machine has
enough memory for the full model.

The shipped sparse runtime limits only internal thinking to 1024 tokens by
default, leaving the rest of the generation budget for the answer. Set
`ADTC_REASONING_BUDGET=-1` for unrestricted thinking, or choose another
non-negative token budget; `ADTC_MAX_TOKENS` still controls the total completion.

Other entry points:

```bash
make test             # offline test suite — passes WITHOUT model weights
make run              # interactive CLI advisor   |  make demo (metadata test prompts)
make profiler         # official adtc-profiler self-check (needs model + build)
```

---

## How it works

A sparse MoE only runs a few experts per token — so a 35B model can answer on
a laptop if the runtime stages exactly the experts each token needs:

```
Question (EN/SW)
   │
   ▼
system prompt ──▶ optional offline RAG context ──▶ model output
                                                       │
                                                       ▼
                                           streamed answer + model thinking
```

- **Model:** `Qwen3.6-35B-A3B-UD-Q2K-experts.gguf` (12.26 GB on disk,
  <3 GB working RSS) — routed experts requantized to Q2_K, everything else
  keeps base types. **Not fine-tuned.** See [MODEL_CARD.md](MODEL_CARD.md).
- **Response path:** the application supplies one detailed system prompt,
  attaches relevant offline RAG context when available, and returns the model
  response without classification, labels, rewriting, regeneration, or a
  fixed fallback.
- **Medication behavior:** the system prompt tells the model not to volunteer
  medication names, doses, or prescriptions unless the user explicitly asks
  about medication or treatment.
- **Official result:** ADTC profiler PASS — **2502 MB peak RSS**,
  **11.0 tok/s** (CI hardware), arc_easy 0.72, CPU-only bounded_3gb arm.
  Run 35514252643, main @ `c454f1a`. See [REPORT.md](REPORT.md).

Full architecture: [ARCHITECTURE.md](ARCHITECTURE.md) · report: [REPORT.md](REPORT.md) ·
evaluation: [EVALUATION.md](EVALUATION.md) · training preflight (NO-GO record): [TRAINING.md](TRAINING.md) ·
novelty & prior art: [NOVELTY.md](NOVELTY.md)

---

## Repository layout

```
├── metadata.json            # profiler manifest (strict schema; `make validate`)
├── download_model.sh        # fetch + SHA256-verify the Q2K GGUF (static URL)
├── model/manifest.json      # machine-readable artifact manifest
├── MODEL_CARD.md            # artifact card (provenance, honesty)
├── REPORT.md                # technical report (Gate-2)
├── provenance/              # truthful provenance package (no fake training logs)
├── src/                     # RAG (stdlib) · sparse backend · web UI · CLI
│   ├── sparse.py            # managed llama-server (frozen K4/16 runtime)
│   ├── webapp.py            # FastAPI backend + SSE streaming
│   └── static/              # offline single-page UI (no CDN)
├── data/medical_guidelines.json # optional offline RAG corpus
├── evals/                   # model/evaluation fixtures
├── training/                # prepared (unrun) post-training pipeline + NO-GO preflight
├── probes/edge0_port/       # frozen llama.cpp patch set (K4/16 + bounded executor)
├── tests/                   # offline tests (no weights needed)
└── model/                   # weights land here (git-ignored)
```

Historical tracks (superseded Falcon line, early experiments) remain under
`scripts/*falcon*`, `tests/test_falcon*`, `configs/falcon-*`,
`kaggle/phase04-falcon-*`, and `docs/research/` — retained with green tests as
research records, not production paths.

---

*Medical content is derived from public WHO/IMCI/national-guideline material and is for clinical decision support only — not a substitute for a qualified clinician.*
