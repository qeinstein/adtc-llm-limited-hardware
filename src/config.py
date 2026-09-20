"""Central configuration for the ADTC 2026 offline medical advisor.

Single source of truth for paths, the profiler manifest (``metadata.json``), and
CPU-runtime knobs. Nothing here imports heavy/optional dependencies (``llama_cpp``,
``numpy``), so it is safe to import in tests and when model weights are absent.

Runtime knobs can be overridden with environment variables (useful on the
constrained target machine) — e.g. ``ADTC_N_THREADS=4 ADTC_N_CTX=1024``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Repository layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
METADATA_PATH = ROOT / "metadata.json"

GUIDELINES_PATH = DATA_DIR / "medical_guidelines.json"
EVAL_SET_PATH = DATA_DIR / "swahili_eval_set.json"
LORA_DATASET_PATH = DATA_DIR / "medical_lora_dataset.json"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_metadata(path: Path | str = METADATA_PATH) -> dict[str, Any]:
    """Load and return the profiler manifest (``metadata.json``)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_path(meta: dict[str, Any] | None = None) -> Path:
    """Resolve the GGUF path exactly the way the adtc-profiler does.

    The profiler reads ``_runtime.model_path`` relative to the submission root
    and falls back to ``model.gguf``.

    ADTC_MODEL_PATH overrides it for local A/B testing only. Candidate models are
    deliberately exported to their own filenames (``-v2``, ``-v3``) so nothing
    ships until an A/B says it should — but that also meant the web UI always
    loaded the shipped model, and a whole manual audit got run against the wrong
    one. The env var makes trying a candidate in the browser a one-liner instead
    of an edit to metadata.json (which the profiler reads, so editing it for a
    local test risks shipping a half-tested path).
    """
    override = os.environ.get("ADTC_MODEL_PATH", "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else (ROOT / p)
    if meta is None:
        meta = load_metadata()
    rel = meta.get("_runtime", {}).get("model_path", "model.gguf")
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


@dataclass(frozen=True)
class RuntimeConfig:
    """CPU inference knobs for the interactive product (llama-cpp-python).

    NOTE: these settings drive our *application/demo* only. The competition's
    automated throughput/memory numbers are produced by the adtc-profiler
    running ``llama-bench`` on the raw GGUF — not by this engine. See REPORT.md.
    """

    # The serving context is independent of the model's response style. KV cost
    # is small (GQA Q8_0: ~40KB/token, ~164MB at 4096) and GDN state is
    # context-constant, so this stays inside the bounded-RSS arms.
    n_ctx: int = field(default_factory=lambda: _env_int("ADTC_N_CTX", 4096))
    # Target eval machine is 4 vCPU; default to a safe value and clamp at runtime.
    n_threads: int = field(default_factory=lambda: _env_int("ADTC_N_THREADS", 4))
    n_batch: int = field(default_factory=lambda: _env_int("ADTC_N_BATCH", 256))
    # CPU-only, always. No GPU offload on the target hardware.
    n_gpu_layers: int = 0
    # mmap keeps peak RSS down (pages are demand-loaded / reclaimable); mlock off
    # so the kernel can evict under memory pressure and we never risk OOM.
    use_mmap: bool = True
    use_mlock: bool = False
    # Quantized KV cache shrinks the cache footprint substantially.
    type_k: str = field(default_factory=lambda: _env_str("ADTC_TYPE_K", "q8_0"))
    type_v: str = field(default_factory=lambda: _env_str("ADTC_TYPE_V", "q8_0"))
    flash_attn: bool = True
    # Cap only the model's internal <think> block. The pinned runtime closes
    # that block when the budget is reached and continues with the final answer,
    # so max_tokens remains available for visible output. -1 is unrestricted.
    reasoning_budget: int = field(
        default_factory=lambda: max(-1, _env_int("ADTC_REASONING_BUDGET", 1024))
    )
    # Optional speculative decoding draft model (path); empty disables it.
    draft_model_path: str = field(default_factory=lambda: _env_str("ADTC_DRAFT_MODEL", ""))


@dataclass(frozen=True)
class GenerationConfig:
    """One unconstrained generation configuration for the model."""

    max_tokens: int = field(default_factory=lambda: _env_int("ADTC_MAX_TOKENS", 2048))
    temperature: float = field(default_factory=lambda: _env_float("ADTC_TEMPERATURE", 0.7))
    top_p: float = field(default_factory=lambda: _env_float("ADTC_TOP_P", 0.95))
    top_k: int = field(default_factory=lambda: _env_int("ADTC_TOP_K", 40))
    repeat_penalty: float = field(default_factory=lambda: _env_float("ADTC_REPEAT_PENALTY", 1.0))
    stop: tuple[str, ...] = ()


# Versioned production prompt (single source: prompts/system.json). The RAG
# layer may add retrieved context to the user message. Falls back to a compact
# string only if the prompts tree is missing.
def _load_system_prompt() -> str:
    try:
        import json as _json
        from pathlib import Path as _Path

        doc = _json.loads(
            (_Path(__file__).resolve().parent.parent / "prompts" / "system.json")
            .read_text(encoding="utf-8"))
        text = doc.get("text", "")
        if text.strip():
            return text
    except (OSError, ValueError):
        pass
    return (
        "You are Jamii Afya, an offline general-purpose assistant with strong "
        "health-information expertise. Answer non-health questions directly too, "
        "and never invent protocols, diagnoses, medicine doses, or thresholds.")


SYSTEM_PROMPT = _load_system_prompt()
SYSTEM_PROMPT_VERSION = "prompts/system.json v2.2.0"


def get_runtime_config() -> RuntimeConfig:
    return RuntimeConfig()


def get_generation_config() -> GenerationConfig:
    return GenerationConfig()
