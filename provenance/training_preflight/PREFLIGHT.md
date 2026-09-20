# Production preflight record (PREFLIGHT.md)

One-shot QLoRA run: Qwen3.6-35B-A3B, Axolotl FSDP2, 4-bit, 2xT4 (Kaggle).
Brief: full preflight A–Z, then production ONLY if every §Y gate is GREEN.

## VERDICT: NO-GO — DO NOT LAUNCH (load gate RED, structural)

Production was NOT launched, per §Y (all-green required). No GPU was burned
re-proving this: the load failure is deterministic — same Axolotl SHA, same
config path, same silicon as the failed v7 canary — and is now additionally
proven by exact arithmetic from the weight index + the loader source.

## Why the load gate is RED (three independent proofs)

**1. Exact arithmetic (this preflight, `/tmp/vram_exact.py`):**
per-tensor shapes read from all 26 safetensor headers via Range requests
(weights never downloaded; BF16 total re-sums to exactly 71.90 GB):

- Linear4bit-quantized params: 33.6213B → NF4 data+scales: **17.336 GB**
- Non-quantized residents (fp16): 655 tensors (embed+lm_head untied,
  333 vision, 19 MTP, norms, routers, GDN vectors): **4.661 GB**
- **Per-rank load total: 21.997 GB vs T4 usable ≈14.56 GB → +51% over.**

**2. Loader source (Axolotl @82109ef, verified in clone):**
`loaders/model.py::_set_device_map_config` forces
`device_map={"": LOCAL_RANK}` for QLoRA+FSDP — the FULL quantized model
materializes on EACH rank's GPU before FSDP wrapping. Offload, grad
checkpointing, CCE, seq len, rank, batch affect STEP memory only and
cannot change this number. `utils/schemas/validation.py` explicitly
REJECTS the only low-memory flag (`cpu_ram_efficient_loading`) with
FSDP2+4-bit ("deadlocks the state dict scatter").

**3. Empirical (canary v7, 2026-09-20, same SHA/path/silicon):**
CUDA OOM on BOTH ranks at ~46% of tensors, 14.21/14.56 GB. See TRAINING.md.

## Alternatives examined (all closed on Kaggle-class hardware)

- FSDP1 + `qlora_sharded_model_loading`: violates the §G FSDP2 mandate AND
  still dies — the sharded loader bypasses transformers' load path, so the
  MoE on-load quant patch (`param_value.is_cuda` gate) never sees the 3D
  expert params; rank 0 accumulates experts in BF16 on CPU (~65 GB) plus
  ~17 GB quantized linears ≈ 82 GB host RAM vs ~30 GB available.
- Excluding vision/MTP: saves ≤ ~3 GB; gap is 7.4 GB. No such load flag
  exists for this arch in Axolotl's FSDP path anyway.
- DeepSpeed ZeRO-3 / 8-bit / offload-first orderings: do not change the
  22 GB per-rank materialization (bnb cannot quantize meta-init params;
  offload engages only after wrapping).
- Newer Axolotl: 82109ef IS main HEAD (2026-09-19) and already contains
  all Qwen3.5 fixes (block_type, MRoPE get_cu_seqlens, reshape). No
  upstream init-path fix exists to pin.

## Next: the one shot is preserved (nothing was launched)

1. **Provisioned GPU (recommended):** 2×40GB+ (or 4×A100-40GB per
   TRAINING.md). Everything is ready: `training/kaggle_train.yaml`
   (frozen), `training/preflight.py` (full gate script), data audited.
2. **Stay untuned (A):** deployment remains Q2_K+K4/16 + system prompt +
   safety guard + guidance layer; closeout continues to final acceptance.
3. FSDP1/custom-loader experiments: NOT recommended (mandate-violating,
   unproven, still RAM-blocked on Kaggle).

## Gate detail

- SOFTWARE (§C): GREEN — Axolotl 82109ef = HEAD, contains block_type +
  MRoPE/reshape fixes (last qwen3_5 touch c381957, 2026-09-04); dep pins
  recorded in `training/environment.lock.txt`; install path proven by v7.
- HARDWARE (§A/...): GREEN for preflight, RED for load — 2xT4-16GB
  recognized; 22.0 GB required per rank.
- MODEL (§B): GREEN — all 8 arch assertions PASS against the pinned
  revision (Qwen3_5MoeForConditionalGeneration/qwen3_5_moe/40L/256E/
  8topk/H2048/MI512).
- DATA (§K/§L/§M): see `training/data_audit.json` — filled below.
- TRAINABLES (§E): GREEN by construction — q/k/v/o only, exactly 80
  tensors / 1,392,640 params (0.004%), router/experts/vision/embed/head
  excluded; asserted in preflight.py stage5 + `train.py`.
- CANARY (§N): NOT RUN — load gate failed first; re-running the
  identical load would burn ~1h GPU quota for zero new information.
- RECOVERY (§O/§Q)/EXPORT (§P): scripted in preflight.py, unexecuted
  (gated on load).
- PRODUCTION ESTIMATE (§A/§T): cannot be derived — sec/step is
  unmeasurable without a load; no max_steps invented.

## Data (§K/§L/§M summary)

Full numbers: `training/data_audit.json` (measured 2026-09-20, real
tokenizer @995ad96e + Axolotl qwen3_5 template, seed 7).

- §K validation: 0 fails / 24,931 (roles, non-empty assistant, control
  strings, system placement all clean). Contamination re-screen vs
  32,551 quarantine prompts: 0 exact, 0 near — quarantine HOLDS.
- §K dedup: 865 exact-dup groups, ALL afrimed ×2 upsampling pairs
  (1,730 rows = 865×2, intentional mixture weight, not corruption).
  Production subset MUST drop second copies → 24,066 unique rows
  (one-pass preference, §A). Pilot (1,465) has 0 internal dups.
- §K PHI/placeholders: 38 + 6 warns, ALL false positives on manual
  exhibit review (binary jokes, lab ranges, citations, karyotype
  "XXX"). No SSN/phone/PHI. PASS.
- §L lengths (templated total): p50=755 p75=825 p90=910 p95=981
  p99=1214 max=2205; assistant-only p50=38 p95=313 max=1542.
  Overlong: >512: ALL 24,931 (system prompt alone exceeds 512);
  >1024: 882; >2048: 5. DECISION: sequence_len=2048, drop the 5 rows
  >2048 at subset build. 512/1024 are REFUTED by the data (1024 would
  silently amputate 882 answers, forbidden by §L). Total corpus:
  19.37M tokens (2.10M assistant).
- §M mask audit: 100/100 clean (prefix exact token-prefix, trained
  span >0, turn ends [im_end, newline]); 0/24,931 rows missing
  im_end termination. (First checker revision wrongly required
  last-token==EOS and scored 0/100 — checker bug, fixed, documented
  here so nobody re-"discovers" it.)
- Mix: medqa 9,992 / medmcqa 11,967 / oasst1 1,242 / afrimed 1,730;
  Kiswahili-heuristic rows: 516. §U subset recipe (not materialized —
  sizing needs measured sec/step): dedup → stratify by §U priorities
  → cap MCQ-conv share → max_steps from canary throughput.

## Production-subset build (recipe, runs when hardware exists)

1. `convert_chat.py` output minus 865 afrimed second copies minus 5
   rows >2048 tokens = 24,061 candidate rows.
2. Stratified pick per §U (afrimed → domains → triage → meds →
   pregnancy/peds → Kiswahili → uncertainty → concise → replay).
3. Freeze SHA, upload as Kaggle dataset, run `preflight.py`, set
   max_steps from measured sec/step (§A), launch.
