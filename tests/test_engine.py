"""Fallback llama-cpp chat handling without loading model weights."""

from src.engine import MedicalLLMEngine


class FakeLLM:
    def create_chat_completion(self, *, stream=False, **kwargs):
        if stream:
            def chunks():
                yield {"choices": [{"delta": {"content": "<thi"}}]}
                yield {"choices": [{"delta": {"content": "nk>careful"}}]}
                yield {"choices": [{"delta": {"content": "</think>answer"}}]}

            return chunks()
        return {
            "choices": [{"message": {
                "reasoning_content": "careful",
                "content": "answer",
            }}],
            "usage": {"completion_tokens": 2},
        }


def _engine():
    engine = object.__new__(MedicalLLMEngine)
    engine.llm = FakeLLM()
    return engine


def test_fallback_chat_preserves_structured_reasoning():
    out = _engine().chat([{"role": "user", "content": "hello"}])
    assert out["thinking"] == "careful"
    assert out["text"] == "answer"


def test_fallback_stream_exposes_legacy_thinking_as_events():
    events = list(_engine().stream_chat_events([
        {"role": "user", "content": "hello"},
    ]))
    assert "".join(piece for kind, piece in events if kind == "thinking") == "careful"
    assert "".join(piece for kind, piece in events if kind == "text") == "answer"
