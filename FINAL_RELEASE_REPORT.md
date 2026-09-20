# FINAL RELEASE REPORT — Jamii Afya Qwen3.6-35B Q2K (ADTC Gate 2)

## Final model / runtime

- Weights: `Qwen3.6-35B-A3B-UD-Q2K-experts.gguf` — 12,262,341,600 bytes,
  SHA256 `0f3698ae92f91db2eb10a3650bdb6693ff8cfaf7060cbc5f256845c700c7603b`
- HF: `Fluxx08/jamii-afya-qwen36-35b-q2k` @ `e938cd2af04dd5f30731922bc4780ef2f264f032`
- Base: `unsloth/Qwen3.6-35B-A3B-GGUF` @ `a483e9e6` (verified SHA before transcode)
- Transform: pinned `llama-quantize --allow-requantize`, 120 routed-expert
  tensors → Q2_K, rest keep base types; byte-identity asserted in-kernel
- Runtime: llama.cpp @ `3057bb6` + edge0 patch set (incl. bench
  LLAMA_ARG_LAZY_MODE compat); K1=4/K2=16, real router, bounded_3gb
  (755 slots + 80 pins); serving ctx 4096; CPU-only, offline
- Fine-tuning: NONE (prepared pipeline NO-GO — see TRAINING.md)

## African use case and adaptability

Jamii Afya targets community health workers at rural and peri-urban African
clinics in Kenya, Tanzania, Uganda, Nigeria, and comparable settings where
connectivity is unreliable and patient information should remain on the local
device. English/Kiswahili offline decision support covers childhood danger
signs, pregnancy red flags, injuries, and referral decisions while deferring
clinical authority to local protocols and qualified clinicians.

No weight-level fine-tuning was performed. The base model can therefore be
adapted to other domains through the system prompt, an optional local RAG
corpus, and runtime configuration. The central contribution here is systems
engineering rather than domain-specific weight training.

## Git

- Repo: `qeinstein/adtc-llm-limited-hardware`
- Current release submission: `e2c17325917f1c471e0f2ecae6ca4a8c5f61dc87`
- Last profiler evidence source: `c454f1a6342b5426c943c2096029c26f993e694d`
  (the benchmark snapshot must be rerun after this release commit before
  final Gate 2 submission).

## System tests

- `pytest tests/`: 174 passed (RAG, webapp wiring, runtime, historical, and
  JOIN4 anchors — all model-independent)
- CI `offline-gates`: PASS (tests, metadata/manifest validation, download
  script checks, stale/secret/size audits, UI static check, lint)
- `make validate`: metadata.json VALID (strict schema incl.
  `base_model_commit_sha`)

## Application response path

- short system prompt with clear medical explanations and explicit medication
  request behavior
- optional offline RAG context when the corpus matches the question
- direct model output; internal reasoning is not exposed, and there are no
  application-side labels, linting, regeneration, or fixed response path

## `make model` / `make webui`

- `make model`: static-URL `download_model.sh`, resume-capable, size + SHA256
  verified (hosted sidecar), skip-if-valid
- `make webui`: venv → deps → model verify → runtime build → backend+frontend
  → readiness wait → URL → cleanup on exit; single command

## ADTC profiler evidence

- Workflow: `.github/workflows/official-profiler.yml` (manual dispatch),
  profiler pin `12be4f384c18d554d99cef380979132273578c59` (latest main)
- Run: 35514252643 on main @ `c454f1a` (artifacts preserved 90 days; this is
  the evidence source for the checked-in snapshot, not a fresh run of the
  current release commit)
- Result: peak_rss 2502.49 MB, steady 2436.26 MB; 16.0 tok/s headline
  generation (16.46 and 15.5 tok/s observed, rounded), TTFT 26394.46 ms;
  arc_easy 0.72 (stock-K8 accuracy path); EPYC 7763
  4-core / 15.6 GB CPU-only;
  model SHA verified in-run; no throttling — GREEN, gate (<7 GB) passed

## Development RSS (separate from official)

- Bounded arm: 2301.2 MiB demonstrated (Kaggle 4-vCPU, exact-IQP executor)
- Frozen b3 (Qwen3.6/K4/16/Q2K): 2.37 GB same methodology
- These are dev numbers, never presented as audit results

## Known limitations

- Weights untuned and unvalidated; no clinician review (cards/data/weights)
- Kiswahili heuristic-tested only; linting heuristic
- Before/after base captures outstanding (scaffolding ready, needs ~25 GB box)
- Profiler accuracy stage uses stock backend (K8/full-mmap) — organizer
  confirmation needed for 8 GB audit boxes (see ARCHITECTURE.md §13)

## Training NO-GO (one paragraph)

A production 4-bit QLoRA/FSDP2 run was fully prepared (pinned env, audited
24,066-row data, frozen config, 7-stage gate script) but NO-GO at the load
gate: 22.0 GB/rank required vs 14.56 GB usable on 2×T4 (exact arithmetic +
loader source + empirical v7 OOM). Nothing was launched; the pipeline is
retained for provisioned hardware. The submission never claims fine-tuning.
