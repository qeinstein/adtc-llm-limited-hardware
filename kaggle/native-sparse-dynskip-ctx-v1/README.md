# DynSkip contextual validation v1

One kernel: REAL decode-time contextual states for L00/L20/L39 (MoE input `x`,
block input `h0`, post-mixer residual `h1`) over the same 32 greedy-decoded
prompts as `native-sparse-l20hdump-v1`, plus a control-vs-policy A/B of the
frozen L20 rule "skip routed experts iff runtime top1 <= 0.02".

- Control (`GGML_DYNSKIP=2`, log-only, never mutates): dumps + routes + text.
- Policy (`GGML_DYNSKIP=1`, armed): L20 routed weights exact-zeroed per token
  when the runtime pre-norm top-8 slot-0 weight (== full top1 prob) is <= 0.02.
  Records skip log + routes + text for exact flip measurement.
- Routing semantics preserved by construction (ids/topk untouched; only the
  combine weights of skipped tokens are zeroed). Policy-vs-control route
  prefix equality is verified locally with flip-position awareness.

Fail-loud gates in-kernel: stream count agreement (9 streams x 32 prompts),
route 40-cycle, skiplog/decode-count agreement, log-only non-mutation,
runtime `top1<=0.02` rate within [5%, 40%], determinism re-run byte-equality,
>=2000 decode tokens total.

Push with:

```bash
kaggle kernels push -p kaggle/native-sparse-dynskip-ctx-v1
kaggle kernels status toheebogunade/jamii-native-sparse-dynskip-ctx-v1
kaggle kernels output toheebogunade/jamii-native-sparse-dynskip-ctx-v1 -p output/dynskip-ctx-v1
```

Local analysis lands in `probes/dynskip_contextual/REPORT.md`.
