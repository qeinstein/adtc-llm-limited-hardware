# Native-sparse dense Q5_K repack v2

Version 1 established two harness issues: `llama-bench` in this pinned
mainline has no `--repack` option, and its invalid-argument output was not
parsed as a failed benchmark. Version 2 uses `llama-cli` for three measured
64-token repetitions per arm, where `--repack` and `--no-repack` are accepted,
and uses the same conservative output hash parser as the exact control.

This remains a standard CPU Q5_K repack donor experiment on the real Qwen
decode path, not a bespoke final dense kernel.
