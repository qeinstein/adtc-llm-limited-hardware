# Phase 6C dense Q5_K repack v2 — script import failure

Version 2 corrected the v1 choice of `llama-bench` (which has no repack
flags) and switched the A/B measurement to `llama-cli`. The Kaggle job failed
at script startup because the new parser used `re` without importing it.

No model was downloaded or executed, and no performance or correctness number
is valid. The one-line failure is preserved in `raw/`. Version 3 adds the
missing import and is the runnable measurement arm; it retains the same exact
model, 4-thread CPU configuration, 64-token repetitions, and repack flags.
