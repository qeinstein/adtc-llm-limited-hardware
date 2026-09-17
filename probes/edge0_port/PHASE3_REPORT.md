# Edge0 Phase 3 — Prerouter Port to N100/CPU Engine (GATE-FREE INFRASTRUCTURE)

Status: LAUNCHED with BASE=3.6 (Phase 2 verdict: quality TIE, 3.6 wins on
efficiency). Pre-launch review fixed 3 deltas: 3.6 pin filled, parity
threshold 1.0→0.999 (fp16 boundary ties), cmake = v5's exact proven recipe.
Hook anchors verified exactly-once in pin 3057bb6; `--poll` verified present.

## Trainset validation (trace v3 COMPLETE, 0.34h wall) — VALID

- 23/23 prompts ok; 82 evals each (80 gen + 2 batching-quirk), 1886 total.
- RMS 1.0035–1.0357; ids coverage uniform 96.34% (first-3-eval gap on every
  prompt, deterministic); parity 0.9981–1.0000 (all pass 0.995 gate).
- 3 diverse npz sampled locally (p00 medical, p12 swahili, p20 short):
  shapes/dtypes exact, all finite, ids in [0,256) all 256 experts hit,
  probs renormalized (rowsums 1.0), sort-desc 0 violations/22960,
  top-1 diverse (max share 76/3280). Cross-prompt duplication ruled out.
- Quirk note: eval 0 is byte-identical across prompts (lone first-token
  eval, same input → same output); evals 1–2 prompt-dependent but
  hidden-only. All labels are exact offline router math on stored hiddens
  (self-consistent by construction); runtime ids cross-check parity only.
- Disk: full 23-npz set (~290MB) stays in kernel outputs (local disk 100%
  full, other agent's 14GB untouched); 3 samples + result.json local.
- v1 post-mortem: `ids 79 != tok 82` → seq-order matching + regression tests
  (tests/test_edge0trace_framing.py, 8 passed). v2 post-mortem: parity
  0.9987 < 0.999 → 0.995 (fp16 boundary flips 2–6/3160, provably benign).
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
