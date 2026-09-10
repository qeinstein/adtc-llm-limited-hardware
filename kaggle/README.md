# Kaggle execution

Heavy Phase 0/1 work runs on Kaggle. Local execution is limited to repository inspection, lightweight unit tests, and report/schema validation.

Push a notebook with:

\`\`\`bash
kaggle kernels push -p kaggle/phase01-data-audit
\`\`\`

Check status and download outputs with:

\`\`\`bash
kaggle kernels status toheebogunade/jamii-afya-phase-01-data-audit
kaggle kernels output toheebogunade/jamii-afya-phase-01-data-audit \
  -p output/kaggle-phase01-data-audit
\`\`\`

The notebook is evaluation-only. It does not start fine-tuning.

## Phase 02 — stock-model frontier screen (CPU, internet, <1 hr)

Push with:

```bash
kaggle kernels push -p kaggle/phase02-stock-bench
```

Two-stage screen-then-confirm on 5 Q4_0 candidates (exact manifest in the
notebook — official Qwen repos ship only Q8_0, so Base comes from
jelawless/fernandoruiz and post-trained from unsloth):

- Jamii (`Fluxx08/jamii-afya-qwen3-0.6b`), 0.6B-Base, 0.6B-Instruct,
  1.7B-Base, 1.7B-Instruct (~3.3 GB total).
- One scalar llama.cpp build (llama-bench only), profiler-parity bench +
  RSS, 8 generation probes (clinical/safety/Kiswahili/repetition), frozen
  MCQ screen (arc_easy+medmcqa x30, 2 parallel workers), conditional
  confirmation (+70 disjoint via `--offset`, only if CIs overlap and the
  screen stayed under 35 min). Every stage timed into `timings.json`.

Results: `phase02-results/{screen_perf,screen_probes,screen_mcq,
confirmation,frontier}.json/md`. No training; Phase 3 stays held until the
frontier picks the target.

## Phase 03 — Qwen3-0.6B-Base QLoRA training (GPU, internet)

Push with:

```bash
kaggle kernels push -p kaggle/phase03-train-06b
```

One epoch of the locked `train_lora.py` recipe + Q4_0 export. Step
checkpoints (`--save_steps 500`) allow `--resume_adapter` continuation if the
9h session limit hits. Artifacts: `phase03-results/` (GGUF + adapter).
