# ik_llama.cpp compatibility probe v2

Version 1 built ik_llama successfully but stopped because its CLI returns
status 1 while printing a valid `--help` usage page. Version 2 records that
conventional behavior and continues to the exact-model smoke and benchmark.

The benchmark uses the current 64-token/3-repeat configuration. It remains a
compatibility and donor measurement: no native-routing or model weights are
changed, and route equality is reported as unmeasured if the upstream runtime
does not expose an observation hook.
