# Jamii Afya — Bringing Frontier Intelligence to African Doorsteps

**This is not a project about making small models available to Africa. That is
already possible.**

The world is already moving toward frontier intelligence. If Africa is still
stuck celebrating 0.5B, 1.5B, or 4B models because they fit on a laptop, we are
not solving the real problem. We are designing around our infrastructure gap
and quietly accepting that the gap should remain. That is painful, and it is
not a serious long-term strategy for the continent.

Small models can be useful for narrow, stable tasks. They are not frontier
intelligence. They do not offer the same breadth of reasoning, language,
planning, coding, and context handling as a much stronger model. [Scaling
studies](https://arxiv.org/abs/2001.08361) and [compute-optimal training
research](https://arxiv.org/abs/2203.15556) make the relationship between model
capacity, data, and capability difficult to ignore. Fine-tuning can specialise a
model; it cannot turn a small base checkpoint into a frontier model.
[LoRA](https://arxiv.org/abs/2106.09685) and related methods adapt a fixed
pretrained parameter budget, and recent research on reasoning
[distillation](https://arxiv.org/abs/2502.12143) has documented a learnability
gap in very small models. The point is not that fine-tuning is useless. The
point is that an interesting fine-tune does not erase the capability ceiling of
the base model.

**The next thing Africa needs to do is not keep fine-tuning smaller models. It
is to bring frontier intelligence a step closer to our hands and doorsteps.**

Jamii Afya is a proof of that direction. We took a considerably stronger,
35-billion-parameter sparse Mixture-of-Experts model and made it run on
commodity CPU-only hardware with no cloud and no internet during inference. The
official ADTC profiler measured **2,502.49 MB peak RSS** and **16.0 tok/s** while
the full model artifact remained **12.26 GB on disk**.

This is a genuine systems breakthrough for Africa, but it is not magic and it is
not finished. **Even this 35B system is still a failure against the real target:**
it is storage-backed, slower than a server, and nowhere near a phone-sized
deployment. It is nevertheless a considerably more meaningful failure in the
right direction than treating a tiny model as Africa's permanent ceiling simply
because it is convenient to run.

The runtime is domain-portable. Healthcare is the current ADTC submission use
case; the underlying system can be adapted to agriculture, education, coding,
enterprise work, local-language tools, and other African ecosystems with a new
system prompt, offline reference corpus, and—when justified—future domain
adaptation.

## What is novel here

We are not claiming to have invented sparse Mixture-of-Experts execution,
expert offloading, SSD-backed inference, or quantization independently. The
novelty is the **measured systems composition and operating point**:

**Qwen3.6-35B-A3B + the real router + K4/16 execution + routed-expert-only
Q2_K quantization + explicit bounded expert staging + CPU-only llama.cpp,
measured below 3 GB RSS at 16.0 tok/s on the official ADTC profiler.**

The runtime uses a fixed **755-slot** staging pool with **80 pinned experts**,
keeps the 12.26 GB model artifact on disk, and proves through preflight traffic
that the bounded executor—not a resident fallback—is the path being measured.
That distinction is the project's systems contribution. The full prior-art
boundary and claim discipline are documented in [NOVELTY.md](NOVELTY.md).

> Built for the **Africa Deep Tech Challenge 2026 — The Laptop LLM Challenge.**

> **Read [ARCHITECTURE.md](ARCHITECTURE.md) before running or evaluating this
> repository.** The README is the quickstart; the architecture document is the
> detailed source of truth for K4/16 routing, selective quantization, bounded
> expert staging, benchmark methodology, platform assumptions, limitations, and
> the actual response path.

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

The shipped sparse runtime uses Qwen3.6's native private reasoning channel. The
server explicitly enables the model's thinking template and the `deepseek`
reasoning parser, so reasoning is returned as `reasoning_content` and remains
separate from the final answer. The UI places it in a collapsed, click-to-view
panel; copied answers and follow-up history contain only final content.
`ADTC_REASONING_BUDGET` controls the reasoning sub-budget;
`ADTC_MAX_TOKENS` controls the total completion.

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
Question
   │
   ▼
system prompt ──▶ optional offline RAG context ──▶ model output
                                                       │
                                                       ▼
                                      quiet thinking state + final answer
```

- **Model:** `Qwen3.6-35B-A3B-UD-Q2K-experts.gguf` (12.26 GB on disk,
  <3 GB working RSS) — routed experts requantized to Q2_K, everything else
  keeps base types. **No weight-level fine-tuning was performed; the shipped
  weights are the base model with runtime quantization only.** See
  [MODEL_CARD.md](MODEL_CARD.md).
- **Response path:** the application supplies one concise behavioral system prompt,
  attaches relevant offline RAG context when available, and returns the model
  response without classification, labels, content rewriting, or fixed answers.
  Reasoning is available only in a collapsed panel and stays separate from
  answer text and conversation history. The pinned server uses Qwen3.6's native
  chat-template controls rather than asking the model to simulate a private
  reasoning protocol in prose. If llama-server ends a turn with hidden
  reasoning but no content, the adapter continues that same assistant turn in
  its content channel instead of discarding the answer.
- **Prompt scope:** the production prompt is 31 words: identity, general scope,
  health-information specialization, audience, desired tone, and
  response-language matching. It contains no emergency, death, medication,
  output-format, or hidden-reasoning checklist.
- **ADTC profiler measurement:** **2502 MB peak RSS**, **16.0 tok/s headline**
  (16.46 and 15.5 tok/s observed, rounded), arc_easy 0.72, CPU-only
  bounded_3gb arm. The checked-in profiler evidence snapshot was run
  35514252643 against historical commit `c454f1a`; it must be rerun on the
  final release commit before final Gate 2 submission. See [REPORT.md](REPORT.md).

Full architecture: [ARCHITECTURE.md](ARCHITECTURE.md) · report: [REPORT.md](REPORT.md) ·
evaluation: [EVALUATION.md](EVALUATION.md) · training preflight (NO-GO record): [TRAINING.md](TRAINING.md) ·
novelty & prior art: [NOVELTY.md](NOVELTY.md)

### Alignment limitation and next step

No weight-level SFT or DPO was performed. A short positive prompt is more stable
for this base model than the former rule-heavy rubric, but prompting is not a
substitute for alignment. With more compute, time, and clinician-reviewed data,
the next step is targeted multi-turn SFT followed by preference optimization to
internalize natural response style, multilingual consistency, clinical tone, and
medicine behavior. Those weights would then require fresh safety, quality, and
clinician evaluation before deployment.

---

## Repository layout

```
├── metadata.json            # profiler manifest (strict schema; `make validate`)
├── download_model.sh        # fetch + SHA256-verify the Q2K GGUF (static URL)
├── scripts/download_model.py # cross-platform fetch + SHA256 verification
├── model/manifest.json      # machine-readable artifact manifest
├── MODEL_CARD.md            # artifact card (model provenance and details)
├── REPORT.md                # technical report (Gate-2)
├── provenance/              # model provenance package
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

## Final Note
Most of our limitations regarding post training and finetuning is limited by compute, given enough compute and support, SFT/DPO will be performed on the abse model for alignment, and fine-tuning for domain specific use cases, so we can ship to real suers at scale.

---

*Medical content is derived from public WHO/IMCI/national-guideline material and is for clinical decision support only — not a substitute for a qualified clinician.*
