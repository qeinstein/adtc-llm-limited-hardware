# Gate-2 before/after capture

This directory contains the identical prompts used for the requested comparison
between the raw model and Jamii Afya's application path.

Run the capture on a machine with the base weights, the pinned runtime, and the
web UI already serving the submitted model:

```bash
python3 scripts/capture_before_after.py \
  --prompts evals/gate2_before_after/prompts.json \
  --base-model /path/to/base-model.gguf \
  --out evals/gate2_before_after/
```

For every prompt, the generated `tp_*.json` stores:

- `base.text`: the answer before the Jamii Afya system prompt and optional RAG;
- `system.reply`: the answer after the current system prompt and optional RAG;
- `system.sources` and `system.telemetry`: the evidence attached to the
  submitted response.

The current checkout contains the capture method and prompts, but not fabricated
outputs. A measured before/after answer should only be added after the vanilla
base-model run has completed.
