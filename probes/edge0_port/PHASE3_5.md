# Phase 3.5 — Training-free K4 normalization (paper verification log)

Source read in full: arXiv:2609.04575v1 (Chen & Yao, submitted 4 Sep 2026),
"Training-Free Halving of Activated Experts in Fine-Grained MoE Models"
(PDF fetched + text-extracted locally; no code link in the paper).
Status: formula VERIFIED, implementation mapped to our pin. No guessing.

## Exact formula (paper Eq. 2)

Router: p = softmax(W_g x) over ALL E experts (E=256, flat: top-1 mass
0.046, top-8 mass 0.182, top-32 mass 0.380 on WikiText).

  w_i = p_i / SUM_{j in T_k2} p_j,   for i in T_k1        (T_k1 ⊆ T_k2)

  y = SUM_{i in T_k1} w_i E_i(x) + g_sh E_sh(x)

- k1 = executed experts (compute). k2 = normalization reference (gain).
- k2=k1 → standard renormalization. k2=E → none (catastrophic: -27.45 MMLU).
- k2=k (trained 8) → trained gain. k2>k → BELOW trained gain, per-token
  adaptive (weights sum to m_k1/m_k2 < 1, NOT to one).
- Shared expert UNCHANGED (separate gate g_sh). No weights modified.
- Only k1 FFNs evaluated; router already computes all E probs → k2 costs
  ~nothing (a wider top-k; we MEASURE it rather than assume).

## Reported numbers (Qwen3.6-35B-A3B, MMLU 2000q 5-shot A-D logits, McNemar)

native 81.65% | (6,6) -2.60 | (6,8) -1.10 | (6,16) +0.75 (!!) |
(4,4) -4.65 | (4,8) -3.10 | (4,16) -0.35 (p=0.66, indistinguishable).
GSM8K 0-shot greedy: (4,4) -6.60 trunc 10.4% | (4,16) -0.60 trunc 1.4%.
Perplexity-optimal k2=8, MMLU-optimal k2=16 (-1.10 at k2=8, p=0.021):
DO NOT select k2 on perplexity. Paper recommends scanning {k1,k,2k}.
397B replicates (k=10→k1=5,k2=10: -0.55, p=0.24).

## Mapping to our pin (llama.cpp 3057bb6)

- Router+norm live in llm_graph_context::build_moe_ffn (src/llama-graph.cpp):
  probs=softmax over E (gating SOFTMAX, qwen35moe passes norm_w=true);
  selected=argsort_top_k(selection_probs, n_expert_used); weights =
  get_rows(probs, selected) / sum (the `if (norm_w)` block); mixture loop
  sums hparams.n_expert_used(il) views (MUST override too — uniform arch).
- Patch (all env-gated, native when unset): GGML_MOE_K1 (default 8) shadows
  n_expert_used for selection+dims+sum-loop; GGML_MOE_K2 (default = native
  k) builds a second top-k2 mass (argsort+get_rows+sum_rows+clamp+div);
  k2==k1 keeps the NATIVE norm lines verbatim (control bit-parity).
- moe_out (routed mixture, pre-shared-add) named edge0_moe_out-<il> and
  captured POST-barrier in the CPU node loop (fusion only touches
  RMS_NORM+MUL, verified safe). cb() does NOT name tensors (delegate!).

## Ours vs paper (interpretation notes)

- Our MMLU is matched-likelihood (mmlu-test.bin, 200 tasks), NOT 5-shot
  A-D prompting; absolute scores differ (~42 vs ~82). The TEST is the
  DELTA PATTERN (naive ≈ -4..5 scaled vs k2=16 ≈ 0), not absolute values.
- Our n=200 has ~3.1pp paired MDD (vs paper's 0.98 at n=2000): architecture
  sanity gate per the brief, not an equivalence proof. Follow-up MMLU-1000
  if the pilot lands in the ambiguous zone.
- Paper disabled reasoning via chat template (transformers); our logprob
  MMLU is thinking-independent, so no thinking control needed for the gate.
  Generations (sanity set) WILL think; inspected as-is.

## v1 results (kernel log; NO output files saved — crash before first dump)

All 8 MMLU-200 arms completed; transcode/layer/sanity/speed never ran
(`RuntimeError: kv type 5` in transcode_q2k's GGUF parser — scalar KV
types unhandled + n_kv/n_tensors swapped; fixed + regression-tested in
tests/test_gguf_kv_skip.py; remainder in edge0phase35r_v1.py):

| arm | k1 | k2 | score | sigma | wall |
|---|---|---|---|---|---|
| native (unset) | - | - | 42.0 | 3.499 | 2112s |
| k8exp (patched control) | 8 | 8 | 42.0 | 3.499 | 2136s |
| naive k4 | 4 | 4 | 38.5 | 3.449 | 1677s |
| k48 | 4 | 8 | 40.0 | 3.473 | 1743s |
| k412 | 4 | 12 | 41.5 | 3.493 | 1774s |
| k416 | 4 | 16 | 41.0 | 3.487 | 1672s |
| k424 | 4 | 24 | 39.5 | 3.465 | 1717s |
| k432 | 4 | 32 | 40.0 | 3.473 | 1733s |

Adjudication vs pre-registered rule: k8exp == native EXACTLY (patch
harness quality-neutral, +1.1% wall overhead). Naive k4 -3.5pp; EVERY
k2>4 arm recovers to within -2.5..-0.5pp. k416 = -1.0pp with full K4
speed (1672s, 1.26x) and no collapse => **STRONG KEEP** (within ~1pp).
k412 nominally best (-0.5pp, ONE question over k416) but all k2 arms
are mutually within noise (n=200, sigma ~3.5; paired detail lost with
the crash, no McNemar possible).

k2 LOCK: **16** (paper's choice; 1 question off nominal-best, flat
within noise across k2 in 8..32; k2 changes only the renormalization
scalar, NOT the executed top-4 set, so speed/traces/cache are
k2-independent). k2=12 recorded as statistically equivalent fallback.

Speed: K4 arms 1672-1774s vs native 2112 (1.19-1.26x; k412's +6% vs
k416 is host noise — k2 cannot change executed work). Matches the
paper's delta PATTERN (naive deep loss, k2=16 ~neutral) scaled to our
matched-likelihood harness. Mandatory Recover-LoRA NOT needed for the
speed sprint.

## r-v1: ENOSPC (dead 40min in; parser fix verified 120/733)

## r-v2: joint gate LANDED, then OOM-killed (ERROR at 8384s)

- transcode q2k: 120/733 -> q2_k OK (~1914s); expert-count check passed.
- **mmlu_q2k_k416 (Q2K-experts + K4/16, transcoded worst case): 38.5
  +/- 3.449 (1834s)**. vs native 42.0: -3.5pp; vs k416-on-IQ2 41.0:
  -2.5pp (5 questions). Unpaired SE of the diff ~4.9pp => WITHIN
  NOISE of both; consistent with no Q2KxK4 interaction AND with a
  small real interaction. No Q2K+K8/MMLU-200 arm exists to separate
  them (Q2K-alone evidence is MMLU-100/K8: 39 vs 37). Deployment
  config point estimate: 38.5. Sprint impact: acceptable modest cost
  (post-training recovers); does NOT overturn Q2K KEEP or K4/16
  STRONG KEEP, which rest on their own gates.
- After mmlu_q2k (4450s), "Killed" (SIGKILL, no traceback) at 8384s.
  Output pull (selective --file-pattern; full pull OOMs the 2GB local
  box on the 13GB file) shows result.json ends at mmlu_q2k_k416 and
  ZERO layer_*/san_* files => died 66min into the FIRST layer-probe
  run (layer_p0_k8, hook active, 30 tokens). Hook code reviewed: no
  deadlock/leak shape (buffered streaming writes, no accumulation);
  mechanism UNKNOWN. Secondary find: hook-after-barrier RACES the next
  node on MT (torn-read risk on in-place buffers) => captures must run
  1-threaded. JSON-verified: mmlu_q2k_k416=38.5; WORK free 8.68GB,
  SCRATCH 1089GB (capacity vindicated).

## s-v1 (running): attach + timeouts + canary

Attaches r-v2 outputs (q2k file, no re-transcode/re-MMLU); layer SPEED
probe without hook (4T); CAPTURE canary with hook at 1T (race-free),
abort-on-first-900s-hang; sanity + speed_q2k. Every subprocess now has
timeout + progress prints + MemAvailable logging + per-run dumps.

## PIN TYPO (all phase35 runs built master, not the pin)

The three phase35 scripts carried LLAMA_COMMIT=3057bb6c... (typo) vs
the true 3057bb66c86c46d5781e50e85462a760ba7d1feb ("ui: add cache
#28802", verified upstream; 51 other kernel files use it). The bogus
SHA fails fetch ("not our ref", verified) and checkout, both unchecked
=> v1, r-v1, r-v2, s-v1 all built ggml-org/llama.cpp MASTER (~Sept
2026). MMLU DELTAS STAND (same binary across arms within each run);
only the provenance label was wrong. s-v1's hook-free llama-cli death
(25min silence + SIGKILL, 32GB free at start) is therefore most likely
a MASTER-ERA regression/interaction, not our patch (inactive on that
path) or capacity. Hook exonerated as the kill cause (same death
without it); 1T-capture rule stands (torn-read race is real).

## s-v2 (running): true pin + asserts + heartbeat

Re-pinned (asserts on fetch/checkout/rev-parse, HEAD recorded); 60s
heartbeat on every cli_run (mem + elapsed; silence becomes data). If
the speed probe flies on the pin, master regression confirmed and the
sprint continues on the pin.

## s-v2 FAILED: linear 0.7GB/min leak -> OOM (pin exonerated too)

`built base: 3057bb66...` confirmed, then speed_p0_k8 drained
MemAvailable 31.9->15.3GB linearly over 24min and OOM-SIGKILLed at
~1527s (Kaggle: "tried to allocate more memory than is available").
Our patch is INACTIVE on that path (k1=k2=8, no hook env) and flags
are valid stock => stock-CLI/flags/file issue. Diff vs PROVEN
edge0trace invocation (23 clean prompts, same pin/model family): we
lacked --single-turn, used -c 1024, temp 0.0, no top-p/perf/ngl.

## s-v3 (running): trace-verbatim + DEVNULL stdin + leak canary

Adopts the trace invocation verbatim (single-turn, c512, temp 0.7,
top-p 0.9, perf, ngl 0) + stdin=DEVNULL (no interactive wait possible)
+ 5-token canary that fails FAST if >3GB drains (flags vs file/patch
discriminator). Either outcome is decisive within ~40min of launch.
