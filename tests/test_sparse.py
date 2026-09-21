"""Sparse backend (frozen Qwen3.6 system) unit tests. All offline: no model
weights, no server process, no network. The managed-server path is covered
by construction tests (argv/env) plus a live repro on real hardware."""
import json

import pytest

from src import sparse


def test_split_thinking_marked():
    t, a = sparse.split_thinking("Hi <think>reasoning here</think> final answer")
    assert t == "reasoning here"
    assert a == "Hi  final answer"


def test_split_thinking_unclosed_is_all_thinking():
    t, a = sparse.split_thinking("<think>still going...")
    assert t == "still going..."
    assert a == ""


def test_split_thinking_absent():
    t, a = sparse.split_thinking("plain answer")
    assert (t, a) == ("", "plain answer")


def test_split_thinking_orphan_close_marker_non_streaming():
    t, a = sparse.split_thinking("</think>answer")
    assert (t, a) == ("", "answer")


def test_streaming_legacy_thinking_markers_are_split_across_chunks():
    parser = sparse._StreamingThinkingParser()
    events = []
    for chunk in ("before <thi", "nk>reason", "ing</think", "> after"):
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    merged = {}
    for kind, piece in events:
        merged[kind] = merged.get(kind, "") + piece
    assert merged == {"text": "before  after", "thinking": "reasoning"}


def test_streaming_orphan_close_marker_is_not_visible():
    parser = sparse._StreamingThinkingParser()
    events = []
    for chunk in ("</thi", "nk>answer"):
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    assert events == [("text", "answer")]


def test_stream_payload_requests_usage():
    payload = sparse.SparseServer._payload(
        object(), [], 32, 0.7, 0.95, True
    )
    assert payload["stream_options"] == {"include_usage": True}


def test_sparse_stream_exposes_usage_and_timings(monkeypatch):
    monkeypatch.setattr(
        sparse,
        "_post_sse",
        lambda *args, **kwargs: iter([
            {"choices": [{"delta": {"content": "answer"}}]},
            {
                "choices": [],
                "usage": {"completion_tokens": 3, "prompt_tokens": 8},
                "timings": {"predicted_per_second": 12.5},
            },
        ]),
    )
    server = object.__new__(sparse.SparseServer)
    server.base_url = "http://127.0.0.1:1"
    server.timeout_s = 1.0
    events = list(server.stream_chat_events([{"role": "user", "content": "hi"}]))
    assert ("text", "answer") in events
    assert ("usage", {
        "completion_tokens": 3,
        "prompt_tokens": 8,
        "timings": {"predicted_per_second": 12.5},
    }) in events


def test_sparse_stream_discards_structured_reasoning(monkeypatch):
    monkeypatch.setattr(
        sparse,
        "_post_sse",
        lambda *args, **kwargs: iter([
            {"choices": [{"delta": {"reasoning_content": "careful"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
        ]),
    )
    server = object.__new__(sparse.SparseServer)
    server.base_url = "http://127.0.0.1:1"
    server.timeout_s = 1.0
    assert list(server.stream_chat_events([{"role": "user", "content": "hi"}])) == [
        ("text", "answer"),
    ]


def test_sparse_stream_discards_prompt_like_reasoning(monkeypatch):
    """A leaked-looking reasoning trace must not become the visible answer."""
    monkeypatch.setattr(
        sparse,
        "_post_sse",
        lambda *args, **kwargs: iter([
            {"choices": [{"delta": {
                "reasoning_content": (
                    "URGENT SITUATIONS MEDICATIONS hidden internal text"
                ),
            }}]},
            {"choices": [{"delta": {
                "content": "Jibu la mwisho kwa Kiswahili.",
            }}]},
        ]),
    )
    server = object.__new__(sparse.SparseServer)
    server.base_url = "http://127.0.0.1:1"
    server.timeout_s = 1.0
    assert list(server.stream_chat_events([{"role": "user", "content": "hi"}])) == [
        ("text", "Jibu la mwisho kwa Kiswahili."),
    ]


def test_sparse_stream_reports_finish_reason(monkeypatch):
    monkeypatch.setattr(
        sparse,
        "_post_sse",
        lambda *args, **kwargs: iter([
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
        ]),
    )
    server = object.__new__(sparse.SparseServer)
    server.base_url = "http://127.0.0.1:1"
    server.timeout_s = 1.0
    assert list(server.stream_chat_events([{"role": "user", "content": "hi"}])) == [
        ("finish", "length"),
    ]


def test_sparse_stream_keeps_terminal_content_with_finish_reason(monkeypatch):
    monkeypatch.setattr(
        sparse,
        "_post_sse",
        lambda *args, **kwargs: iter([
            {"choices": [{
                "delta": {"content": "final answer"},
                "finish_reason": "stop",
            }]},
        ]),
    )
    server = object.__new__(sparse.SparseServer)
    server.base_url = "http://127.0.0.1:1"
    server.timeout_s = 1.0
    assert list(server.stream_chat_events([{"role": "user", "content": "hi"}])) == [
        ("text", "final answer"),
        ("finish", "stop"),
    ]


def test_server_cmd_frozen_flags():
    argv = sparse.server_cmd("/m/model.gguf", port=8421, n_ctx=2048,
                            threads=4, poll=0)
    assert argv[0].endswith("llama-server")
    for flag, val in (("-m", "/m/model.gguf"), ("--port", "8421"),
                      ("-t", "4"), ("--poll", "0"), ("-c", "2048"),
                      ("-ngl", "0"), ("--reasoning", "on"),
                      ("--reasoning-format", "deepseek"),
                      ("--chat-template-kwargs", '{"enable_thinking":true}'),
                      ("--reasoning-budget", "1024")):
        assert flag in argv and argv[argv.index(flag) + 1] == val


def test_server_cmd_reasoning_budget_is_configurable():
    argv = sparse.server_cmd("/m/model.gguf", reasoning_budget=0)
    assert argv[argv.index("--reasoning-budget") + 1] == "0"
    with pytest.raises(ValueError):
        sparse.server_cmd("/m/model.gguf", reasoning_budget=-2)


def test_server_cmd_can_prompt_the_model_to_finish_after_budget():
    argv = sparse.server_cmd(
        "/m/model.gguf", reasoning_budget=1024,
        reasoning_budget_message="Answer now.",
    )
    assert argv[argv.index("--reasoning-budget-message") + 1] == \
        "Answer now.\n</think>\n\n"


def test_server_cmd_preserves_an_explicit_native_close_marker():
    argv = sparse.server_cmd(
        "/m/model.gguf", reasoning_budget=1024,
        reasoning_budget_message="Answer now.\n</think>\n\n",
    )
    assert argv[argv.index("--reasoning-budget-message") + 1] == \
        "Answer now.\n</think>\n\n"


def test_default_reasoning_budget_message_closes_qwen_thinking_block():
    from src.config import RuntimeConfig

    assert "</think>" in RuntimeConfig().reasoning_budget_message


def test_build_env_resident():
    env = sparse.build_env("resident")
    assert env["GGML_MOE_K1"] == "4" and env["GGML_MOE_K2"] == "16"
    assert "GGML_PHASE6_BOUNDED_CACHE" not in env


def test_build_env_bounded_needs_pins():
    with pytest.raises(ValueError):
        sparse.build_env("bounded_3gb")
    env = sparse.build_env("bounded_3gb", "/p/pins_3.0.txt", profile=True)
    assert env["GGML_PHASE6_BOUNDED_CACHE"] == "1"
    assert env["GGML_PHASE6_SLOTS"] == "755"
    assert env["GGML_PHASE6_ASYNC"] == "1"
    assert env["GGML_PHASE6_PINS"] == "/p/pins_3.0.txt"
    assert env["GGML_PHASE6_PROFILE"] == "1"
    assert env["LLAMA_ARG_LAZY_MODE"] == "on"


def test_build_env_unknown_arm():
    with pytest.raises(KeyError):
        sparse.build_env("bounded_9gb", "/p/x")


def test_export_pins(tmp_path):
    dest = sparse.export_pins("3.0", tmp_path)
    keys = [x for x in dest.read_text().split() if x.strip()]
    assert len(keys) == 80
    assert all(0 <= int(k) < 10240 for k in keys)
    with pytest.raises(KeyError):
        sparse.export_pins("9.9", tmp_path)


def test_freeze_file_contract():
    fz = sparse.load_freeze()
    assert fz["frozen"] is True
    assert fz["llama_cpp_commit"] == "3057bb66c86c46d5781e50e85462a760ba7d1feb"
    assert (fz["decode"]["k1"], fz["decode"]["k2"]) == (4, 16)
    assert (fz["decode"]["threads"], fz["decode"]["poll"]) == (4, 0)
    assert fz["model"]["final"]["size_bytes"] == 12262341600
    assert fz["model"]["final"]["sha256"] == \
        "0f3698ae92f91db2eb10a3650bdb6693ff8cfaf7060cbc5f256845c700c7603b"
    assert fz["arms"]["bounded_3gb"]["slots"] == 755


def test_webapp_backend_selector(monkeypatch):
    import src.webapp as w

    monkeypatch.setenv("ADTC_BACKEND", "sparse")
    assert w._backend() == "sparse"
    monkeypatch.setenv("ADTC_BACKEND", "llamacpp")
    assert w._backend() == "llamacpp"
    monkeypatch.setenv("ADTC_BACKEND", "auto")
    monkeypatch.setenv("ADTC_MODEL_PATH", "model/Qwen3.6-35B-A3B-UD-Q2K-experts.gguf")
    assert w._backend() == "sparse"
    monkeypatch.setenv("ADTC_MODEL_PATH", "model/Falcon-H1-1.5B-Deep-JamiiAfya-Q4_K_M.gguf")
    assert w._backend() == "llamacpp"


def test_webapp_health_and_early_stream():
    # No model on disk here: health reports not-downloaded and the stream
    # endpoint terminates with the RAG-preview done event (no backend call).
    import os

    from fastapi.testclient import TestClient

    import src.webapp as w

    os.environ.pop("ADTC_MODEL_PATH", None)
    c = TestClient(w.app)
    h = c.get("/api/health").json()
    assert h["model_downloaded"] is False
    assert "backend" in h
    r = c.post("/api/chat/stream", json={"message": "Hello.", "history": []})
    assert r.status_code == 200
    body = r.text
    assert '"kind": "done"' in body
def test_webapp_shutdown_stops_sparse(monkeypatch):
    import src.webapp as w
    from fastapi.testclient import TestClient

    stopped = []

    class FakeSrv:
        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(w, "_sparse", FakeSrv())
    with TestClient(w.app):
        pass
    assert stopped == [True]
    assert w._sparse is None
