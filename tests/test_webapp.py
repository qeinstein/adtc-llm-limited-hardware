"""Tests for the system-prompt -> optional-RAG -> model path."""

import asyncio
import json

import pytest

from src import webapp
from src.rag import RAGResult


QUESTION = "What should I do about this headache?"


class FakeRAG:
    def __init__(self, grounded=True):
        self.grounded = grounded
        self.retriever = []

    def build(self, query, top_n=3):
        return RAGResult(
            query=query,
            retrieved=[{"id": "headache-1"}] if self.grounded else [],
            context="headache reference" if self.grounded else "",
            user_content=("Reference guidance:\nheadache reference\n\nQuestion:\n" + query)
            if self.grounded else query,
            is_grounded=self.grounded,
        )

    def system_prompt_for(self, result):
        return "SYSTEM PROMPT"


class FakeSparse:
    def __init__(self):
        self.chat_calls = []

    def chat(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return {"thinking": "model reasoning", "text": "model answer", "usage": {}}

    def stream_chat(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        yield "thinking", "model reasoning"
        yield "text", "model answer"

    def rss_mb(self):
        return 100.0


@pytest.fixture
def faked(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    monkeypatch.setattr("src.config.resolve_model_path", lambda *a, **k: model)
    monkeypatch.setattr(webapp, "_backend", lambda: "sparse")
    return model


def _drain(resp):
    async def collect():
        out = []
        async for chunk in resp.body_iterator:
            out.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return out

    frames = []
    for blob in asyncio.run(collect()):
        for frame in blob.split("\n\n"):
            for line in frame.splitlines():
                if line.startswith("data:"):
                    frames.append(json.loads(line[5:].strip()))
    return frames


def test_chat_sends_only_system_rag_and_history_to_model(faked, monkeypatch):
    monkeypatch.setattr(webapp, "_rag", FakeRAG())
    server = FakeSparse()
    monkeypatch.setattr(webapp, "_get_sparse", lambda: server)

    response = webapp.chat(webapp.ChatRequest(message=QUESTION))

    assert response.reply == "model answer"
    assert response.thinking == "model reasoning"
    messages, options = server.chat_calls[0]
    assert messages[0] == {"role": "system", "content": "SYSTEM PROMPT"}
    assert messages[-1]["role"] == "user"
    assert "Reference guidance" in messages[-1]["content"]
    assert "mode" not in options
    assert not hasattr(response, "urgency")


def test_ungrounded_questions_still_reach_the_model(faked, monkeypatch):
    monkeypatch.setattr(webapp, "_rag", FakeRAG(grounded=False))
    server = FakeSparse()
    monkeypatch.setattr(webapp, "_get_sparse", lambda: server)

    response = webapp.chat(webapp.ChatRequest(message="hello"))

    assert response.reply == "model answer"
    assert len(server.chat_calls) == 1
    assert server.chat_calls[0][0][-1] == {"role": "user", "content": "hello"}


def test_stream_has_no_deterministic_meta_event(faked, monkeypatch):
    monkeypatch.setattr(webapp, "_rag", FakeRAG())
    server = FakeSparse()
    monkeypatch.setattr(webapp, "_get_sparse", lambda: server)

    frames = _drain(webapp.chat_stream(webapp.ChatRequest(message=QUESTION)))

    assert [frame["kind"] for frame in frames] == ["thinking", "text", "done"]
    assert frames[-1]["reply"] == "model answer"
    assert "urgency" not in frames[-1]
    assert "guard_ok" not in frames[-1]


def test_missing_model_returns_only_a_runtime_message(monkeypatch, tmp_path):
    monkeypatch.setattr("src.config.resolve_model_path", lambda: tmp_path / "missing.gguf")
    monkeypatch.setattr(webapp, "_rag", FakeRAG(grounded=False))

    response = webapp.chat(webapp.ChatRequest(message="hello"))

    assert response.model_ready is False
    assert "model is not downloaded" in response.reply.lower()
    assert "verified guidance" not in response.reply.lower()
