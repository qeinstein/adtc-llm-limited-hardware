# Edge0 Phase 3 — Prerouter Port to N100/CPU Engine (GATE-FREE INFRASTRUCTURE)

Status: LAUNCHED with BASE=3.6 (Phase 2 verdict: quality TIE, 3.6 wins on
efficiency). Pre-launch review fixed 3 deltas: 3.6 pin filled, parity
threshold 1.0→0.999 (fp16 boundary ties), cmake = v5's exact proven recipe.
Hook anchors verified exactly-once in pin 3057bb6; `--poll` verified present.
Branch: research/edge0-port. No quality-gate assumption made; nothing irreversible started.

## What Edge0's mechanism needs (from Phase 2 fidelity analysis, PHASE2_STATIC.md)

1. Gate-input router inputs — the prerouter must run on hidden states BEFORE the
   gate matmuls (else it predicts K+1 with K's firing pattern and recall collapses);
   our hook captures gate0/gate1 inputs (up-proj inputs), never post-MoE outputs.
2. Trainset > testset records — the prerouter trains on this artifact alone, so every
   record carries the gate-input hidden state (fp16), K8 routes, and router logits.
3. Retriever/generator boundaries — the trace filter must be consistent with the
   measured curve; checked post-hoc by `verify_coverage_local` (empirical cumulative
   mass on a held-out 20%).

## Artifact: kaggle/native-sparse-edge0trace-v1/ (v1, unlaunched)

- `edge0trace_v1.py` — downloads pinned Qwen3-30B-A3B-Instruct-2507 BF16 (sha-verified),
  builds llama.cpp pin b7847 (GGML_BACKEND_DL=OFF, AVX2, K-quants ON — v2's proven recipe
  with the two v2 build defects fixed: static-backend link + single -D list),
  injects a gate-input hook into ggml-cpu.c (CALL line mirrors edge0phase2's proven
  route hook at FOREACH_OP; capture guarded by ith==0 + op/type checks; fp16 conversion
  via exported `ggml_fp32_to_fp16_row` — no hand-rolled converter),
  runs FreeText-2507 decode (n=257, T=512, K=8, full 48 layers), emits:
  - `route_traces.npz` (K8 routes/scores/q4w, all tokens),
  - `prerouter_trainset.bin` (fp16 gate-input records + K8/K4 teacher + AGENTS.txt label),
  - `trace.json` (curve fit + provenance).
- `kernel-metadata.json` — GPU, internet-enabled, dataset output.
- Verification so far (no launch): hook syntax-checked (`gcc -fsyntax-only -Werror`
  against stub decls: OK); script `py_compile`: OK; build flag/API provenance grepped
  from the local pin (build-cmake failure in v2 reproduced by construction, then fixed).

## Fidelity resolutions from the static-implementation round (kept)

- Speculative expert prefetch: KILLED for the N100 path (fidelity #1 violated: K4-vs-K8
  route equality only 40%, far below the 90% needed; overlap pipe covers the same
  latency at 96% hidden instead).
- Recovery training with silent labels: KILLED (fidelity #3: it would un-teach the
  sparse attention without the KL guardrail).
- K4 override in any quality gate: KILLED (fidelity #2 violated: K4 trims the K8 map
  instead of re-routing).

## Launch plan (after v3 adjudication)

1. v3 picks base (3.5 vs 3.6) -> set MODEL_URL/MODEL_SHA256 in edge0trace_v1.py.
2. Push kernel, run (~40 min), download outputs.
3. Run held-out coverage check; fit curve; write prerouter spec + PHASE3 results.
4. Only then: prerouter prototype training (local, on the trainset artifact).

## Cost of this phase so far

Zero GPU minutes (nothing launched). Disk/RSS impact: none (kernel dir ~15 KB).
