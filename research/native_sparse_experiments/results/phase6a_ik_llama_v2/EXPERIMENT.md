# Phase 6A ik_llama.cpp v2 — CLI option-surface failure

Version 2 correctly accepted ik_llama’s nonzero `--help` status and completed
the upstream build and exact model download. It then stopped before inference
because the ik_llama CLI help has no `-ngl`/`--gpu-layers` option; CPU-only is
the default in this build. The v2 smoke-command generator treated that option
as mandatory.

No model performance, RSS, route, or output number is valid from this run.
The full help, build, model identity, result JSON, and Kaggle log are preserved
in `raw/`. Version 3 adapts to the actual CPU-only option surface and proceeds
to smoke and 64-token/3-repeat benchmark measurement.
