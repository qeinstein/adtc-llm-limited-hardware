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

    # Context window: kept small on purpose — the KV cache is O(n_ctx) RAM, and
    # our RAG pipeline compresses context so long windows are unnecessary.
    n_ctx: int = field(default_factory=lambda: _env_int("ADTC_N_CTX", 2048))
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
    # Optional speculative decoding draft model (path); empty disables it.
    draft_model_path: str = field(default_factory=lambda: _env_str("ADTC_DRAFT_MODEL", ""))


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling defaults tuned for factual, low-variance clinical answers."""

    max_tokens: int = 512
    # ADTC_TEMPERATURE override: real testing showed the same clinical prompt
    # sometimes returns the correct answer and sometimes hallucinates an unindicated
    # drug (e.g. an antihistamine for pre-eclampsia) purely from sampling variance
    # at temperature 0.3. Lower values (e.g. 0.1, or 0.0 for greedy) reduce that
    # variance -- they don't remove the wrong association from the model, but they
    # make the model's most-likely (and here, more often correct) output win
    # consistently instead of a roll of the dice each time.
    temperature: float = field(default_factory=lambda: _env_float("ADTC_TEMPERATURE", 0.3))
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    # We fine-tuned from Qwen3-0.6B-**Base**, which has no instruction post-training
    # of its own, so it does not reliably emit EOS at the end of an answer. Real
    # testing showed it produce a correct clinical answer and then keep going —
    # starting a fresh "Q:"/"S:" pair, opening a markdown section, or drifting into
    # Chinese (base-model bleed-through). These stops cut generation at the point
    # the answer is actually finished. Without them the answer is right but the
    # output looks broken.
    stop: tuple[str, ...] = (
        "\nQ:", "\nA:", "\nS:", "\nJ:",
        "\nQuestion:", "\nAnswer:", "\nSwali:", "\nJibu:",
        "\nExample", "\nMfano",
        "\n##", "\n---",
        "<|im_end|>", "<|endoftext|>",
    )


# System prompt for the advisor. Deliberately safety-first: this is clinical
# DECISION SUPPORT, not a diagnosis, and it must escalate danger signs.
SYSTEM_PROMPT = (
    "You are Jamii Afya, an offline medical decision-support assistant for community "
    "health workers and nurses in rural African clinics. You answer in the SAME "
    "language as the question (English or Kiswahili). Be clear, concise, and "
    "practical for a low-resource setting.\n"
    "Rules:\n"
    "1. Use only the retrieved reference context for clinical management. If it is "
    "missing or irrelevant, do not give clinical instructions: advise consultation "
    "with a clinician or the national treatment guideline.\n"
    "2. Always surface DANGER SIGNS and say clearly when to REFER urgently.\n"
    "3. Give only widely-standardized doses (e.g. ORS, zinc 20 mg, paracetamol "
    "10-15 mg/kg, ACT by weight band); if unsure of a dose, say to follow the "
    "national treatment guideline rather than guessing.\n"
    "4. You are decision support, NOT a substitute for a clinician's examination. "
    "State this when giving management advice."
)


def get_runtime_config() -> RuntimeConfig:
    return RuntimeConfig()


def get_generation_config() -> GenerationConfig:
    return GenerationConfig()
