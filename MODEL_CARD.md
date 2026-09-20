# Jamii Afya — Qwen3.6-35B-A3B Q2K-experts GGUF

Jamii Afya is an offline English/Kiswahili clinical decision-support
assistant for community health workers. It is decision support, not a
diagnosis or a replacement for a qualified clinician.

## Active artifact

`Qwen3.6-35B-A3B-UD-Q2K-experts.gguf` — 12,262,341,600 bytes
SHA256: `0f3698ae92f91db2eb10a3650bdb6693ff8cfaf7060cbc5f256845c700c7603b`
HF: `Fluxx08/jamii-afya-qwen36-35b-q2k`

- Base: `unsloth/Qwen3.6-35B-A3B-GGUF` (`Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`
  @ `a483e9e6`, SHA256 `2e8f5f70…7bef`), from `Qwen/Qwen3.6-35B-A3B`.
- Transform: `llama-quantize --allow-requantize` @ llama.cpp `3057bb6`:
  the 120 routed-expert tensors requantized to Q2_K, all other tensors
  keep base types. Byte-verified, layout-verified (120/120 Q2_K experts).
- Runtime: llama.cpp / GGUF, CPU-only, K4/16 sparse execution + bounded
  expert staging (<3 GB working set). Fully offline after download.
- **Fine-tuning: NONE.** No LoRA/QLoRA/full weight updates. Adaptation is
  system prompt + guidance retrieval + safety rules + output lint + runtime.

## Safety architecture (not in the weights)

Deterministic fact/risk extraction, urgency routing with instant emergency
banners, 18-domain structured guidance, authority grounding (WHO/IMCI/NCDC
claims must come from retrieved cards), output lint with one regen else a
safe fallback, Fast/Medium/High reasoning modes with a guaranteed answer.

## Limitations

- Not fine-tuned, not clinically validated; no clinician has reviewed the
  weights, cards, or data rows.
- Kiswahili coverage is heuristic-tested, not native-speaker reviewed.
- Linting is heuristic; the model can be wrong. For urgent symptoms,
  poisoning, severe breathing problems, pregnancy danger signs, heavy
  bleeding, altered consciousness, or other emergencies, seek immediate
  qualified medical care.

## Reproduce

```bash
git clone https://github.com/qeinstein/adtc-llm-limited-hardware.git
cd adtc-llm-limited-hardware
make model && make webui
```
