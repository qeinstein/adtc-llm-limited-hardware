"""Backend wiring tests for reasoning modes (no weights, faked backends).

Covers: UI->backend mode propagation, medium default, invalid-mode 400,
per-mode budgets, the phase-2 answer guarantee (stream + non-stream),
identical safety behavior across modes, and the first-event meta contract.
"""
import asyncio
import json

import pytest
from fastapi import HTTPException

from src import webapp
from src.modes import phase1_max_tokens
from src.rag import RAGResult

ROUTINE_Q = "How do I prepare oral rehydration solution at home?"
BENIGN_A = ("Mix one litre of clean water with six teaspoons of sugar and "
            "half a teaspoon of salt. Give fluids and see a clinician if "
            "danger signs appear.")


class FakeRAG:
    def __init__(self, grounded=True):
        self._grounded = grounded

    def build(self, query, top_n=3):
        return RAGResult(query=query,
                         retrieved=[{"id": "ors-1"}] if self._grounded else [],
                         context="ORS context" if self._grounded else "",
                         user_content=f"Q: {query}",
                         is_grounded=self._grounded)

    def system_prompt_for(self, result):
        return "SYS"


class FakeSparse:
    """Scripted sparse backend. chats: list of {thinking, text} replies."""

    def __init__(self, chats, stream_events=()):
        self.chats = list(chats)
        self.stream_events = list(stream_events)
        self.chat_calls = []
        self.stream_calls = []

    def chat(self, messages, max_tokens=512, **kw):
        self.chat_calls.append({"messages": messages,
                                "max_tokens": max_tokens})
        if not self.chats:
            raise AssertionError("unexpected extra chat() call")
        return dict(self.chats.pop(0))

    def stream_chat(self, messages, max_tokens=512, **kw):
        self.stream_calls.append({"messages": messages,
                                  "max_tokens": max_tokens})
        yield from self.stream_events

    def rss_mb(self):
        return 100.0


@pytest.fixture
def faked(tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    monkeypatch.setattr("src.config.resolve_model_path",
                        lambda *a, **k: model)
    monkeypatch.setattr(webapp, "_backend", lambda: "sparse")
    return model


def _drain(resp):
    async def _collect():
        out = []
        async for chunk in resp.body_iterator:
            out.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return out
    frames = []
    for blob in asyncio.run(_collect()):
        for frame in blob.split("\n\n"):
            for line in frame.splitlines():
                if line.startswith("data:"):
                    frames.append(json.loads(line[5:].strip()))
    return frames


def test_default_mode_medium(faked, monkeypatch):
    monkeypatch.setattr(webapp, "_rag", FakeRAG())
    srv = FakeSparse([{"thinking": "t", "text": BENIGN_A}])
    monkeypatch.setattr(webapp, "_get_sparse", lambda: srv)
    req = webapp.ChatRequest(message=ROUTINE_Q)
    assert req.mode == "medium"
    resp = webapp.chat(req)
    assert resp.mode == "medium"
    assert resp.regenerated is False
    assert resp.reply == BENIGN_A
    assert srv.chat_calls[0]["max_tokens"] == phase1_max_tokens("medium")


def test_invalid_mode_400(faked):
    with pytest.raises(HTTPException) as exc:
        webapp.chat(webapp.ChatRequest(message=ROUTINE_Q, mode="turbo"))
    assert exc.value.status_code == 400


def test_invalid_mode_400_stream(faked):
    with pytest.raises(HTTPException) as exc:
        webapp.chat_stream(webapp.ChatRequest(message=ROUTINE_Q, mode="ultra"))
    assert exc.value.status_code == 400


@pytest.mark.parametrize("mode,total", [("fast", 384), ("medium", 1024),
                                        ("high", 2304)])
def test_per_mode_budgets_propagate(faked, monkeypatch, mode, total):
    monkeypatch.setattr(webapp, "_rag", FakeRAG())
    srv = FakeSparse([{"thinking": "t", "text": BENIGN_A}])
    monkeypatch.setattr(webapp, "_get_sparse", lambda: srv)
    resp = webapp.chat(webapp.ChatRequest(message=ROUTINE_Q, mode=mode))
    assert resp.mode == mode
    assert srv.chat_calls[0]["max_tokens"] == total


def test_phase2_answer_guarantee(faked, monkeypatch):
    monkeypatch.setattr(webapp, "_rag", FakeRAG())
    srv = FakeSparse([{"thinking": "long deliberation", "text": ""},
                      {"thinking": "", "text": BENIGN_A}])
    monkeypatch.setattr(webapp, "_get_sparse", lambda: srv)
    resp = webapp.chat(webapp.ChatRequest(message=ROUTINE_Q, mode="high"))
    assert resp.reply == BENIGN_A
    assert resp.regenerated is True
    assert resp.thinking == "long deliberation"
    assert len(srv.chat_calls) == 2
    # phase 2 is bounded by the answer allowance, not the full budget:
    assert srv.chat_calls[1]["max_tokens"] == 768
    ms2 = srv.chat_calls[1]["messages"]
    assert ms2[-2] == {"role": "assistant",
                       "content": "long deliberation"}
    assert ms2[-1]["role"] == "user"


def test_no_phase2_when_answer_present(faked, monkeypatch):
    monkeypatch.setattr(webapp, "_rag", FakeRAG())
    srv = FakeSparse([{"thinking": "", "text": BENIGN_A}])
    monkeypatch.setattr(webapp, "_get_sparse", lambda: srv)
    resp = webapp.chat(webapp.ChatRequest(message=ROUTINE_Q, mode="fast"))
    assert resp.regenerated is False
    assert len(srv.chat_calls) == 1


def test_safety_identical_across_modes(faked, monkeypatch):
    seen = {}
    for mode in ("fast", "medium", "high"):
        monkeypatch.setattr(webapp, "_rag", FakeRAG())
        srv = FakeSparse([{"thinking": "t", "text": BENIGN_A}])
        monkeypatch.setattr(webapp, "_get_sparse", lambda srv=srv: srv)
        resp = webapp.chat(webapp.ChatRequest(message=ROUTINE_Q, mode=mode))
        seen[mode] = (resp.urgency, resp.safety_override,
                      tuple(resp.guidance_cards), srv.chat_calls[0]["messages"])
    base = seen["medium"]
    for mode in ("fast", "high"):
        assert seen[mode][:3] == base[:3]
    # system prompts differ ONLY by the steering line:
    for mode in ("fast", "high"):
        a = seen[mode][3][0]["content"].replace(
            "\n\nREASONING EFFORT: ", "\n\nX: ", 1)
        b = base[3][0]["content"]
        assert a.split("\n\nX: ")[0] == b


def test_stream_meta_first_then_done(faked, monkeypatch):
    monkeypatch.setattr(webapp, "_rag", FakeRAG())
    srv = FakeSparse([], stream_events=[("thinking", "hmm"),
                                        ("text", "part1"),
                                        ("text", "part2")])
    # stream path needs no chat() calls when the answer streams:
    monkeypatch.setattr(webapp, "_get_sparse", lambda: srv)
    frames = _drain(webapp.chat_stream(
        webapp.ChatRequest(message=ROUTINE_Q, mode="fast")))
    assert frames[0]["kind"] == "meta"
    assert frames[0]["mode"] == "fast"
    assert "urgency" in frames[0] and "safety_override" in frames[0]
    assert "guidance_cards" in frames[0]
    kinds = [f["kind"] for f in frames]
    assert kinds[-1] == "done"
    done = frames[-1]
    assert done["reply"] == "part1part2"
    assert done["thinking"] == "hmm"
    assert done["mode"] == "fast"
    assert done["regenerated"] is False
    assert srv.stream_calls[0]["max_tokens"] == phase1_max_tokens("fast")


def test_stream_phase2_regen(faked, monkeypatch):
    monkeypatch.setattr(webapp, "_rag", FakeRAG())
    srv = FakeSparse([{"thinking": "", "text": BENIGN_A}],
                     stream_events=[("thinking", "only-thinking")])
    monkeypatch.setattr(webapp, "_get_sparse", lambda: srv)
    frames = _drain(webapp.chat_stream(
        webapp.ChatRequest(message=ROUTINE_Q, mode="medium")))
    done = frames[-1]
    assert done["kind"] == "done"
    assert done["reply"] == BENIGN_A
    assert done["regenerated"] is True
    texts = [f["piece"] for f in frames if f["kind"] == "text"]
    assert "".join(texts) == BENIGN_A
    assert srv.chat_calls[0]["max_tokens"] == 512  # answer allowance


def test_early_exit_echoes_mode(faked, monkeypatch):
    monkeypatch.setattr(webapp, "_rag", FakeRAG(grounded=False))
    resp = webapp.chat(webapp.ChatRequest(message="xqzt blorpt",
                                          mode="high"))
    assert resp.mode == "high"
    assert resp.reply  # safe ungrounded response, no backend touch
