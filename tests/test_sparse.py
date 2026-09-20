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


def test_server_cmd_frozen_flags():
    argv = sparse.server_cmd("/m/model.gguf", port=8421, n_ctx=2048,
                            threads=4, poll=0)
    assert argv[0].endswith("llama-server")
    for flag, val in (("-m", "/m/model.gguf"), ("--port", "8421"),
                      ("-t", "4"), ("--poll", "0"), ("-c", "2048"),
                      ("-ngl", "0")):
        assert flag in argv and argv[argv.index(flag) + 1] == val


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
    assert "verified guidance" in body  # ungrounded early-exit reply


def test_webapp_chat_model_shape():
    import src.webapp as w

    fields = getattr(w.ChatResponse, "model_fields", None) or w.ChatResponse.__fields__
    assert "thinking" in fields


def test_guarded_once_regen_then_ok():
    import src.webapp as w
    from runtime.safety import SAFE_FALLBACK

    calls = []

    def fake_call(corrective):
        calls.append(corrective)
        if corrective is None:
            return {"text": "Chest pain has many causes. Rest and drink water.", "thinking": ""}
        return {"text": "Call emergency services immediately and go to the hospital.", "thinking": ""}

    risk = {"level": "EMERGENCY", "hits": ["chest-pain"]}
    text, thinking, tele, guard = w._guarded_once(fake_call, risk, "chest pain")
    assert calls[0] is None and "emergency-no-escalation" in calls[1]
    assert "hospital" in text
    assert guard["lint"] == [] and guard["regened"] is True


def test_guarded_once_fallback():
    import src.webapp as w
    from runtime.safety import SAFE_FALLBACK

    def fake_call(corrective):
        return {"text": "Just monitor at home.", "thinking": ""}

    risk = {"level": "EMERGENCY", "hits": ["stroke"]}
    text, _, _, guard = w._guarded_once(fake_call, risk, "face droop")
    assert text == SAFE_FALLBACK
    assert guard["fallback"] is True
    assert "wait-at-home-despite-acuity" in guard["lint"]


def test_chat_end_to_end_guard(monkeypatch, tmp_path):
    import src.webapp as w
    from fastapi.testclient import TestClient

    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    monkeypatch.setenv("ADTC_BACKEND", "sparse")
    monkeypatch.setattr("src.config.resolve_model_path", lambda: model)

    calls = []

    class FakeSrv:
        def chat(self, messages, max_tokens=512):
            calls.append(messages)
            if len(calls) == 1:
                return {"text": "Chest pain has many causes. Rest.", "thinking": "", "usage": {}}
            return {"text": "Call emergency services immediately.", "thinking": "", "usage": {}}

        def rss_mb(self):
            return 123.0

    monkeypatch.setattr(w, "_sparse", FakeSrv())
    c = TestClient(w.app)
    r = c.post("/api/chat", json={
        "message": "My husband has crushing chest pain",
        "history": []})
    assert r.status_code == 200
    body = r.json()
    assert "emergency services" in body["reply"]
    assert body["telemetry"]["guard"]["regened"] is True
    assert len(calls) == 2  # initial + one corrective regen
    w._sparse = None
