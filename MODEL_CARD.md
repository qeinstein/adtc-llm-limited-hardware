# Jamii Afya Falcon-H1

Jamii Afya is an offline English/Kiswahili health and general assistant for
community health workers. It is clinical decision support, not a diagnosis or a
replacement for a qualified clinician.

## Active artifact

`Falcon-H1-1.5B-Deep-JamiiAfya-Q4_K_M.gguf`

- Base: `tiiuae/Falcon-H1-1.5B-Deep-Instruct`
- Base revision: `b6648636ddc906688974282de6e7a243395f5423`
- Runtime: llama.cpp / GGUF, CPU-only at deployment
- Quantization: Q4_K_M
- Training: fp16 full-model LoRA SFT with assistant-only loss
- Safety: clinical safety examples are rendered both with and without the Jamii Afya system prompt

The training trajectory is fixed at 96 domain-lock-in steps, 24 safety-polish
steps, and 16 capability-replay steps on a sm75+ CUDA worker using the
optimized Mamba/causal-conv path. No DPO, ORPO, KTO, PPO, or GRPO is used.

The model can be wrong. For urgent symptoms, poisoning, severe breathing
problems, pregnancy danger signs, heavy bleeding, altered consciousness, or
other emergencies, seek immediate qualified medical care.
