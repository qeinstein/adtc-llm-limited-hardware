---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
tags:
  - gguf
  - moe
  - medical
  - swahili
  - africa
  - cpu
  - edge-ai
  - llama.cpp
  - quantized
---

# Jamii Afya — Qwen3.6-35B-A3B Sparse CPU GGUF

**Jamii Afya** is a CPU-first deployment of **Qwen3.6-35B-A3B** built for the
Africa Deep Tech Challenge 2026.

The goal is unusual:

> Run a capable 35B-class sparse MoE model on an ordinary CPU laptop with
> very limited RAM, while preserving enough quality for useful frontline
> health information and triage assistance.

Instead of shrinking the system to a small dense model, Jamii Afya keeps the
large sparse model and changes **how its routed experts are represented,
loaded, and executed**.

The deployment stack combines:

- **Qwen3.6-35B-A3B**
- routed-expert-only **Q2_K** quantization
- **K4/16** sparse expert execution
- bounded expert staging from storage
- CPU-only `llama.cpp` inference
- an offline health-assistance layer for English and Kiswahili

No weight-level fine-tuning was performed on this release.

---

## Model artifact

| Field | Value |
|---|---|
| File | `Qwen3.6-35B-A3B-UD-Q2K-experts.gguf` |
| Size | `12,262,341,600` bytes |
| SHA256 | `0f3698ae92f91db2eb10a3650bdb6693ff8cfaf7060cbc5f256845c700c7603b` |
| Base architecture | Qwen3.6-35B-A3B |
| Runtime | custom `llama.cpp` sparse execution path |
| Routed expert execution | K4 |
| Router normalization mass | K16 |
| Routed expert quantization | Q2_K |
| Intended hardware | CPU-only commodity laptop |

---

## What is different about this GGUF?

A normal deployment would attempt to keep far more of the model resident in
memory.

Jamii Afya instead treats the routed expert bank as a **storage-backed sparse
resource**.

At inference time:

```text
token
  ↓
native router
  ↓
top-16 router scores
  ↓
execute top-4 routed experts
  ↓
required expert already staged?
  ├── yes → execute
  └── no  → load into bounded expert slot → execute
  ↓
combine using K16 normalization mass
  ↓
next layer

## Reproduce

```bash
git clone https://github.com/qeinstein/adtc-llm-limited-hardware.git
cd adtc-llm-limited-hardware
make model && make webui
```
