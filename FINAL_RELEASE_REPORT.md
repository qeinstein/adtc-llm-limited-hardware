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

## Git

- Repo: `qeinstein/adtc-llm-limited-hardware`
- PR: #19 (`release/jamii-afya-q2k` → `main`)
- Merge of #19: `dc201741c8f960216c7a3398e243e16d503e72c5`; profiled main:
  `c454f1a6342b5426c943c2096029c26f993e694d`

## System tests

- `pytest tests/`: 157 passed (RAG, webapp wiring, runtime, historical, and
  JOIN4 anchors — all model-independent)
- CI `offline-gates`: PASS (tests, metadata/manifest validation, download
  script checks, stale/secret/size audits, UI static check, lint)
- `make validate`: metadata.json VALID (strict schema incl.
  `base_model_commit_sha`)

## Application response path

- system prompt with detailed medical explanations and explicit medication
  request behavior
- optional offline RAG context when the corpus matches the question
- direct model output with model-emitted thinking; no application-side labels,
  linting, regeneration, or fixed response path

## `make model` / `make webui`

- `make model`: static-URL `download_model.sh`, resume-capable, size + SHA256
  verified (hosted sidecar), skip-if-valid
- `make webui`: venv → deps → model verify → runtime build → backend+frontend
  → readiness wait → URL → cleanup on exit; single command

## Official profiler

- Workflow: `.github/workflows/official-profiler.yml` (manual dispatch),
  profiler pin `12be4f384c18d554d99cef380979132273578c59` (latest main)
- Run: 35514252643 on main @ `c454f1a` (artifacts preserved 90 days)
- Result: peak_rss 2502.49 MB, steady 2436.26 MB; 11.0 tok/s generation,
  TTFT 26394.46 ms; arc_easy 0.72 (stock-K8 accuracy path); EPYC 7763
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
