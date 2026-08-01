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
        return {"text": text.strip(), "telemetry": telemetry.as_dict()}

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
        return {"text": text.strip(), "telemetry": telemetry.as_dict()}

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
            stream=True,
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield piece
