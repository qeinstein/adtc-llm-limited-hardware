"""CPU inference engine for the interactive Jamii Afya product (llama-cpp-python).

Scope note: this engine powers our *application and demo* — the interactive
clinical advisor judges will try. It is NOT what produces the competition's
automated throughput/memory numbers; those come from the adtc-profiler running
``llama-bench`` on the raw GGUF. Here we optimise the real user experience with
techniques that help on CPU:

- Quantized KV cache (+ flash-attention) to keep RAM low.
- Prompt/KV prefix caching so the fixed system+few-shot prefix is not re-processed
  on every query (prefill/TTFT dominates CPU latency).
- Optional prompt-lookup (n-gram) speculative decoding — OFF by default. Found by
  testing (not documented anywhere) that this llama-cpp-python version crashes with
  "could not broadcast input array from shape (N,) into shape (0,)" specifically on
  longer, real chat-formatted prompts (system + few-shot + RAG context) — it only
  appears to work on short single-line completions, which is exactly the case a
  smoke test would use and miss. Real end-to-end testing with the actual RAG prompt
  is what caught this. Safe to try enabling for short, non-chat completions only.

``llama_cpp`` is imported lazily so this module (and the RAG stack) import cleanly
in tests and when no model weights are present.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from src.config import (
    GenerationConfig,
    RuntimeConfig,
    get_generation_config,
    get_runtime_config,
    resolve_model_path,
)


def _ggml_kv_type(name: str, llama_cpp: Any) -> int:
    mapping = {
        "f32": llama_cpp.GGML_TYPE_F32,
        "f16": llama_cpp.GGML_TYPE_F16,
        "q8_0": llama_cpp.GGML_TYPE_Q8_0,
        "q5_1": llama_cpp.GGML_TYPE_Q5_1,
        "q5_0": llama_cpp.GGML_TYPE_Q5_0,
        "q4_1": llama_cpp.GGML_TYPE_Q4_1,
        "q4_0": llama_cpp.GGML_TYPE_Q4_0,
    }
    return mapping.get(name.lower(), llama_cpp.GGML_TYPE_Q8_0)


@dataclass
class Telemetry:
    elapsed_sec: float
    prompt_tokens: int
    completion_tokens: int
    throughput_tps: float
    peak_rss_mb: float

    def as_dict(self) -> dict[str, float]:
        return {
            "elapsed_sec": round(self.elapsed_sec, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "throughput_tps": round(self.throughput_tps, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
        }


def _rss_mb() -> float:
    try:
        import psutil  # local import: optional dependency

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


SAFETY_FALLBACK = (
    "I'm not confident I can give a complete, reliable answer to this one — "
    "please consult a clinician or refer to the nearest health facility, "
    "especially if there are any danger signs. / Sina uhakika wa kutosha kujibu "
    "swali hili kikamilifu — tafadhali wasiliana na daktari au kituo cha afya "
    "kilicho karibu, hasa ikiwa kuna ishara za hatari."
)


def _trim_foreign_script(text: str) -> str:
    """Cut the answer at the first CJK character.

    Jamii Afya answers only in English or Kiswahili — both Latin script. Qwen3 is
    Chinese-pretrained, and because we fine-tuned from the Base checkpoint (no
    instruction post-training), real testing showed it finish a correct Kiswahili
    clinical answer and then drift into Chinese. Any CJK output is therefore
    base-model bleed-through, never a legitimate answer, so truncating there is
    safe and unambiguous — no language detection or heuristics needed.
    """
    for i, ch in enumerate(text):
        o = ord(ch)
        if (
            0x4E00 <= o <= 0x9FFF      # CJK Unified Ideographs
            or 0x3040 <= o <= 0x30FF   # Hiragana + Katakana
            or 0xAC00 <= o <= 0xD7AF   # Hangul
            or 0x3000 <= o <= 0x303F   # CJK punctuation
            or 0xFF00 <= o <= 0xFFEF   # Fullwidth forms
        ):
            return text[:i].rstrip()
    return text


def _guard_repetition(text: str, min_chunk_words: int = 6, min_repeats: int = 3) -> str:
    """Detect degenerate generation (a word-chunk repeating back-to-back) and
    truncate to the last clean sentence before it, falling back to a short
    honest safety message if nothing usable is left.

    Found by testing (not a hypothetical): the fine-tuned model reliably loops
    on longer, complex Kiswahili prompts even with a raised repeat_penalty —
    this is a real generation-time failure mode, not a sampling knob to tune.
    Doesn't touch the automated scoring path (that reads raw logits on fixed
    MCQ continuations, never free generation) — this only protects the
    interactive chat product from shipping garbled/looping output.
    """
    # Sentence-level pass FIRST. The word-chunk scan below only looks at blocks up
    # to ~26 words, which real testing showed is too narrow: the model looped a
    # ~50-word hallucinated citation four times and slipped straight through. Whole
    # repeated sentences are both the common shape of this failure and much cheaper
    # to spot than widening the word scan.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen: dict[str, int] = {}
    for idx, sent in enumerate(sentences):
        key = sent.strip().lower()
        if len(key.split()) < 5:
            continue  # short fragments legitimately recur ("Refer urgently.")
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 2:
            clean = " ".join(sentences[:idx]).strip()
            return clean if len(clean) >= 20 else SAFETY_FALLBACK

    words = text.split()
    for chunk_len in range(min_chunk_words, min_chunk_words + 20):
        for start in range(0, max(len(words) - chunk_len * min_repeats, 0)):
            chunk = words[start:start + chunk_len]
            if not chunk:
                continue
            repeats = 1
            pos = start + chunk_len
            while words[pos:pos + chunk_len] == chunk:
                repeats += 1
                pos += chunk_len
                if repeats >= min_repeats:
                    break
            if repeats >= min_repeats:
                clean = " ".join(words[:start]).strip()
                # Keep only up to the last full sentence to avoid a mid-clause cutoff,
                # unless it already ends cleanly on its own (no trailing partial clause).
                if clean and clean[-1] not in ".!?":
                    cut = max(clean.rfind(sep) for sep in (".", "!", "?"))
                    clean = clean[:cut + 1] if cut != -1 else ""
                return clean if len(clean) >= 20 else SAFETY_FALLBACK
    return text


class MedicalLLMEngine:
    """Thin, model-agnostic wrapper over ``llama_cpp.Llama`` for CPU serving."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        runtime: Optional[RuntimeConfig] = None,
        prompt_lookup_decoding: bool = False,
        prompt_cache: bool = True,
        verbose: bool = False,
    ):
        self.runtime = runtime or get_runtime_config()
        self.model_path = str(model_path) if model_path else str(resolve_model_path())
        self.verbose = verbose

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_path}\n"
                "Run ./download_model.sh first to fetch the GGUF."
            )

        import llama_cpp  # lazy: only needed when we actually serve
        from llama_cpp import Llama

        n_threads = min(self.runtime.n_threads, os.cpu_count() or self.runtime.n_threads)

        draft_model = None
        if prompt_lookup_decoding:
            try:
                from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

                draft_model = LlamaPromptLookupDecoding(num_pred_tokens=10)
            except Exception:
                draft_model = None  # older llama-cpp-python: silently skip

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.runtime.n_ctx,
            n_threads=n_threads,
            n_batch=self.runtime.n_batch,
            n_gpu_layers=self.runtime.n_gpu_layers,
            use_mmap=self.runtime.use_mmap,
            use_mlock=self.runtime.use_mlock,
            flash_attn=self.runtime.flash_attn,
            type_k=_ggml_kv_type(self.runtime.type_k, llama_cpp),
            type_v=_ggml_kv_type(self.runtime.type_v, llama_cpp),
            draft_model=draft_model,
            verbose=verbose,
        )

        # Real root cause of the crash (see module docstring): the speculative
        # draft_model breaks specifically on longer chat-formatted prompts, not on
        # any interaction with this cache. prompt_lookup_decoding now defaults to
        # False, so this branch runs normally; the `draft_model is None` guard is
        # kept as defense-in-depth in case someone re-enables it manually anyway.
        if prompt_cache and draft_model is None:
            try:
                # Reuses KV for a shared prompt prefix across calls (big CPU win).
                self.llm.set_cache(llama_cpp.LlamaRAMCache(capacity_bytes=256 * 1024 * 1024))
            except Exception:
                pass

    # -- generation -----------------------------------------------------------
    def _messages(self, prompt: str, system_prompt: Optional[str]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        generation: Optional[GenerationConfig] = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        gen = generation or get_generation_config()
        start = time.time()
        peak = _rss_mb()
        resp = self.llm.create_chat_completion(
            messages=self._messages(prompt, system_prompt),
            max_tokens=overrides.get("max_tokens", gen.max_tokens),
            temperature=overrides.get("temperature", gen.temperature),
            top_p=overrides.get("top_p", gen.top_p),
            top_k=overrides.get("top_k", gen.top_k),
            repeat_penalty=overrides.get("repeat_penalty", gen.repeat_penalty),
            stop=list(overrides.get("stop", gen.stop)),
        )
        elapsed = max(time.time() - start, 1e-3)
        peak = max(peak, _rss_mb())

        choices = resp.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage = resp.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", 0))
        telemetry = Telemetry(
            elapsed_sec=elapsed,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=completion_tokens,
            throughput_tps=completion_tokens / elapsed,
            peak_rss_mb=peak,
        )
        return {"text": _guard_repetition(_trim_foreign_script(text.strip())), "telemetry": telemetry.as_dict()}

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        generation: Optional[GenerationConfig] = None,
        **overrides: Any,
    ) -> Iterator[str]:
        gen = generation or get_generation_config()
        for chunk in self.llm.create_chat_completion(
            messages=self._messages(prompt, system_prompt),
            max_tokens=overrides.get("max_tokens", gen.max_tokens),
            temperature=overrides.get("temperature", gen.temperature),
            top_p=overrides.get("top_p", gen.top_p),
            top_k=overrides.get("top_k", gen.top_k),
            repeat_penalty=overrides.get("repeat_penalty", gen.repeat_penalty),
            stop=list(overrides.get("stop", gen.stop)),
            stream=True,
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield piece

    # -- multi-turn chat (full message history, for the web UI) ---------------
    def chat(
        self,
        messages: list[dict[str, str]],
        generation: Optional[GenerationConfig] = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Like `generate`, but takes a full [{"role", "content"}, ...] history
        (system + alternating user/assistant turns + the new user message) so the
        model has real conversational memory across turns, not just the latest
        question."""
        gen = generation or get_generation_config()
        start = time.time()
        peak = _rss_mb()
        resp = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=overrides.get("max_tokens", gen.max_tokens),
            temperature=overrides.get("temperature", gen.temperature),
            top_p=overrides.get("top_p", gen.top_p),
            top_k=overrides.get("top_k", gen.top_k),
            repeat_penalty=overrides.get("repeat_penalty", gen.repeat_penalty),
            stop=list(overrides.get("stop", gen.stop)),
        )
        elapsed = max(time.time() - start, 1e-3)
        peak = max(peak, _rss_mb())

        choices = resp.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        usage = resp.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", 0))
        telemetry = Telemetry(
            elapsed_sec=elapsed,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=completion_tokens,
            throughput_tps=completion_tokens / elapsed,
            peak_rss_mb=peak,
        )
        return {"text": _guard_repetition(_trim_foreign_script(text.strip())), "telemetry": telemetry.as_dict()}

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        generation: Optional[GenerationConfig] = None,
        **overrides: Any,
    ) -> Iterator[str]:
        gen = generation or get_generation_config()
        for chunk in self.llm.create_chat_completion(
            messages=messages,
            max_tokens=overrides.get("max_tokens", gen.max_tokens),
            temperature=overrides.get("temperature", gen.temperature),
            top_p=overrides.get("top_p", gen.top_p),
            top_k=overrides.get("top_k", gen.top_k),
            repeat_penalty=overrides.get("repeat_penalty", gen.repeat_penalty),
            stop=list(overrides.get("stop", gen.stop)),
            stream=True,
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield piece
