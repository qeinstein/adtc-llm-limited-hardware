# Jamii Afya — Offline Clinical Advisor (ADTC 2026)

**Domain:** Healthcare & Medical · **Languages:** English + Kiswahili · **Runtime:** llama.cpp / GGUF, CPU-only, 100% offline

*Jamii Afya* ("community health") is an offline clinical **decision-support** assistant for community health workers and nurses in rural African clinics. It runs a 35B-parameter sparse mixture-of-experts in under 3 GB of working RAM on a commodity 8 GB laptop — no GPU, no internet — answers in the language of the question, grounds answers in curated WHO/IMCI guidance, and always surfaces **danger signs and when to refer**. It is decision support — not a diagnosis, and not a replacement for a clinician.

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
backend + frontend, waits for readiness, prints the URL, and cleans up on exit.

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
Query (EN/SW)
   │
   ▼
fact/risk extraction ──▶ deterministic safety rules ──▶ structured guidance
(runtime/safety)         (urgency, overrides)            (18 clinical domains)
                                                           │
   ┌───────────────────────────────────────────────────────┘
   ▼
Qwen3.6-35B-A3B + real router + K4/16 + Q2_K routed experts + bounded staging
(pinned llama.cpp + edge0 patch set; Fast/Medium/High reasoning modes)
   │
   ▼
output safety lint ──▶ streamed answer (thinking shown separately, collapsed)
```

- **Model:** `Qwen3.6-35B-A3B-UD-Q2K-experts.gguf` (12.26 GB on disk,
  <3 GB working RSS) — routed experts requantized to Q2_K, everything else
  keeps base types. **Not fine-tuned.** See [MODEL_CARD.md](MODEL_CARD.md).
- **Safety:** deterministic rules fire before generation (emergency banners
  render instantly, never after reasoning); authority claims (WHO/IMCI/NCDC)
  must come from retrieved guidance; outputs are linted with one regen, else a
  safe fallback. See [SAFETY.md](SAFETY.md).
- **Guidance:** 18 clinical domains as structured cards (required/prohibited/
  ask-if-missing). See [GUIDANCE.md](GUIDANCE.md).
- **Reasoning modes:** Fast / Medium (default) / High change thinking effort
  only — safety behavior is identical. Reasoning gets its own allowance with a
  protected, guaranteed final answer.
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
├── MODEL_CARD.md            # artifact card (provenance, honesty, limits)
├── REPORT.md                # technical report (Gate-2)
├── provenance/              # truthful provenance package (no fake training logs)
├── src/                     # RAG (stdlib) · sparse backend · web UI · CLI
│   ├── modes.py             # Fast/Medium/High canonical definitions
│   ├── sparse.py            # managed llama-server (frozen K4/16 runtime)
│   ├── webapp.py            # FastAPI backend + SSE streaming
│   └── static/              # offline single-page UI (no CDN)
├── runtime/safety/          # deterministic safety layer (facts/rules/lint)
├── guidance/                # 18-domain structured guidance cards + engine
├── evals/                   # regression suites (judge/domains/kiswahili/safety)
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
