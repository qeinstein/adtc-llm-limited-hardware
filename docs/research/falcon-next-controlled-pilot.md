# Next Falcon controlled pilot

The v18 Stage-A adapter is rejected. It reached `53.9683%` fast-dev
`acc_norm`, but failed held-out safety and generation gates, including an
invented WHO protocol, guessed medication dosing, missing dispositions, and
incoherent/truncated continuations. It must not seed another stage.

The next pilot is deliberately an ablation, not a production run:

- 12 optimizer steps, with complete checkpoints and fast-dev evaluation at
  steps 4, 8, and 12;
- plain FP16 training on the P100, FP32 autoregressive evaluation;
- LoRA rank 8, alpha 16, dropout 0.1;
- attention-only targets `q_proj,k_proj,v_proj,o_proj`;
- learning rate `5e-6`;
- baseline-vs-adapter generation on the 24 frozen clinical/safety prompts;
- maximum 128 generated tokens for the fast curve; full-length batteries stay
  reserved for candidates that pass the safety veto.

The canonical production configuration is unchanged. The notebook remains
audit-only by default; the pilot profile is selected explicitly by the
`FALCON_PILOT_PROFILE=safe_attention` branch when a remote pilot is launched.
The run must be rejected if any critical safety failure appears, even if
MCQA or dev loss improves.
