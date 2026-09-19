"""Jamii Afya — minimal offline chat web UI (for demo/judges, not the scored path).

Scope note: the ADTC profiler scores the raw GGUF directly (llama-bench / lm-eval);
this app is never in that automated loop. It exists for the qualitative/judge
experience and the demo video — a clean, real product on top of the same engine
and RAG stack, with genuine multi-turn conversation memory.

Runs 100% locally/offline: no CDN assets, no external calls, single static page.

    pip install fastapi uvicorn
    PYTHONPATH=. uvicorn src.webapp:app --host 0.0.0.0 --port 8420
    # then open http://localhost:8420
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.rag import RAGPipeline, ungrounded_response

STATIC_DIR = Path(__file__).resolve().parent / "static"

CAREFUL_MODE_SUFFIX = (
    "\n\nFor this question, reason through the clinical assessment step by step "
    "first (danger signs, likely causes, what to check), THEN give your final "
    "clear recommendation."
)

app = FastAPI(title="Jamii Afya")

_engine = None  # lazy-loaded on first request so the server starts even pre-download
_sparse = None  # managed llama-server for the frozen Qwen sparse system
_rag = RAGPipeline()


def _get_engine():
    global _engine
    if _engine is None:
        from src.engine import MedicalLLMEngine

        _engine = MedicalLLMEngine()
    return _engine


def _backend() -> str:
    """Backend selector. ADTC_BACKEND=auto (default): the frozen Q2K sparse
    artifact is served by the pinned llama-server; anything else keeps the
    legacy in-process llama-cpp-python engine. Explicit "sparse"/"llamacpp"
    override for testing."""
    import os

    from src.config import resolve_model_path

    want = os.environ.get("ADTC_BACKEND", "auto").strip().lower()
    if want in ("sparse", "llamacpp"):
        return want
    return "sparse" if "Q2K-experts" in resolve_model_path().name else "llamacpp"


def _get_sparse():
    global _sparse
    if _sparse is None:
        import os

        from src.config import get_runtime_config, resolve_model_path
        from src.sparse import SparseServer, free_port

        rt = get_runtime_config()
        port = int(os.environ.get("ADTC_SPARSE_PORT", "0")) or free_port()
        _sparse = SparseServer(
            resolve_model_path(),
            arm=os.environ.get("ADTC_SPARSE_ARM", "bounded_3gb"),
            port=port, n_ctx=rt.n_ctx,
            threads=min(rt.n_threads, os.cpu_count() or rt.n_threads),
            poll=int(os.environ.get("ADTC_POLL", "0")))
        _sparse.start()
    return _sparse


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []
    careful_mode: bool = False


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    telemetry: dict
    model_ready: bool
    thinking: str = ""  # model-emitted reasoning text (Qwen sparse backend)


@app.get("/")
def index() -> FileResponse:
    """The landing page — project story (problem, approach, what we learned)."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/chat")
def chat_page() -> FileResponse:
    """The actual interactive advisor."""
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/api/health")
def health() -> dict:
    from src.config import resolve_model_path

    return {"model_downloaded": resolve_model_path().exists(),
            "model": resolve_model_path().name,
            "backend": _backend(),
            "guidelines_loaded": len(_rag.retriever)}


def _ground(req: ChatRequest):
    """RAG grounding shared by streaming and non-streaming chat."""
    from src.config import resolve_model_path
    from runtime.safety import classify_input, reminder_for

    risk = classify_input(req.message)
    reminder = reminder_for(risk)
    result = _rag.build(req.message, top_n=3)
    sources = [d.get("id", d.get("title", "?")) for d in result.retrieved] if result.is_grounded else []
    if not result.is_grounded:
        return None, ChatResponse(
            reply=ungrounded_response(req.message), sources=[],
            telemetry={"elapsed_sec": 0, "throughput_tps": 0, "peak_rss_mb": 0},
            model_ready=resolve_model_path().exists())
    if not resolve_model_path().exists():
        preview = (
            "[Model not downloaded yet — RAG preview only]\n\n"
            f"Retrieved context that would ground the answer:\n{result.context or '(no match found)'}"
        )
        return None, ChatResponse(
            reply=preview, sources=sources,
            telemetry={"elapsed_sec": 0, "throughput_tps": 0, "peak_rss_mb": 0},
            model_ready=False)
    system_prompt = _rag.system_prompt_for(result) + (
        CAREFUL_MODE_SUFFIX if req.careful_mode else ""
    )
    if reminder:
        system_prompt += "\n\nSAFETY NOTE FOR THIS TURN: " + reminder
    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": t.role, "content": t.content} for t in req.history]
    messages.append({"role": "user", "content": result.user_content})
    return (messages, sources, risk), None


def _guarded_once(call, risk, prompt):
    """Run one generation, lint it, regen once with a corrective
    instruction on failure, else return the safe fallback. The call takes
    an optional corrective instruction and returns {text, thinking?, tele?}.
    Returns (text, thinking, tele, guard_event)."""
    from runtime.safety import (SAFE_FALLBACK, corrective_instruction,
                                lint_output, log_event)

    out = call(None)
    lint = lint_output(risk, prompt, out["text"])
    regened = False
    if not lint["ok"]:
        out = call(corrective_instruction(lint["failures"]))
        lint = lint_output(risk, prompt, out["text"])
        regened = True
    if not lint["ok"]:
        return SAFE_FALLBACK, "", {}, log_event(
            risk, {"lint": lint["failures"], "fallback": True})
    return (out["text"], out.get("thinking", ""), out.get("tele", {}),
            log_event(risk, {"lint": [], "regened": regened}))


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    import time

    grounded, early = _ground(req)
    if early is not None:
        return early
    messages, sources, risk = grounded
    if _backend() == "sparse":
        start = time.time()
        srv = _get_sparse()

        def _call(corrective):
            ms = messages + ([{"role": "user", "content": corrective}]
                             if corrective else [])
            return dict(srv.chat(ms, max_tokens=512), tele={})

        text, thinking, _, guard = _guarded_once(_call, risk, req.message)
        elapsed = max(time.time() - start, 1e-3)
        tele = {"elapsed_sec": round(elapsed, 3),
                "peak_rss_mb": round(srv.rss_mb(), 1), "guard": guard}
        return ChatResponse(reply=text, sources=sources, thinking=thinking,
                            telemetry=tele, model_ready=True)
    engine = _get_engine()

    def _call2(corrective):
        ms = messages + ([{"role": "user", "content": corrective}]
                         if corrective else [])
        r = engine.chat(ms, max_tokens=512)
        return {"text": r["text"], "thinking": "", "tele": r["telemetry"]}

    text, _, tele, guard = _guarded_once(_call2, risk, req.message)
    tele = dict(tele)
    tele["guard"] = guard
    return ChatResponse(reply=text, sources=sources, telemetry=tele,
                        model_ready=True)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE stream: thinking pieces and answer pieces as separate events.

    Event payloads (JSON in `data:` lines):
      {"kind": "thinking", "piece": "..."}   model-emitted reasoning text
      {"kind": "text", "piece": "..."}       final-answer text
      {"kind": "done", "reply": ..., "thinking": ..., "sources": [...],
       "telemetry": {...}}                    terminal event (always sent)
      {"kind": "error", "error": "..."}       terminal on failure
    """
    import json
    import time

    grounded, early = _ground(req)
    if early is not None:
        final = dict(early.model_dump() if hasattr(early, "model_dump") else early.dict())
        final["kind"] = "done"

        def _early():
            yield f"data: {json.dumps(final)}\n\n"
        return StreamingResponse(_early(), media_type="text/event-stream")

    messages, sources, risk = grounded

    def _gen():
        from runtime.safety import lint_output, log_event

        start = time.time()
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        try:
            if _backend() == "sparse":
                srv = _get_sparse()
                for kind, piece in srv.stream_chat(messages, max_tokens=512):
                    (thinking_parts if kind == "thinking" else text_parts).append(piece)
                    yield f"data: {json.dumps({'kind': kind, 'piece': piece})}\n\n"
                peak = srv.rss_mb()
            else:
                engine = _get_engine()
                for piece in engine.stream_chat(messages, max_tokens=512):
                    text_parts.append(piece)
                    yield f"data: {json.dumps({'kind': 'text', 'piece': piece})}\n\n"
                peak = 0.0
        except Exception as exc:  # interrupted generation / backend down
            yield f"data: {json.dumps({'kind': 'error', 'error': str(exc)[:500]})}\n\n"
            return
        elapsed = max(time.time() - start, 1e-3)
        reply = "".join(text_parts)
        thinking = "".join(thinking_parts)
        n_tok = len(reply.split())  # approx; exact usage comes from headers
        # End-of-stream lint (advisory: the client shows a caution banner on
        # failure; regen inside a live stream would double latency, so the
        # non-stream endpoint is the regen path).
        lint = lint_output(risk, req.message, reply)
        log_event(risk, {"lint": lint["failures"], "stream": True})
        yield f"data: {json.dumps({'kind': 'done', 'reply': reply, 'thinking': thinking, 'sources': sources, 'guard_ok': lint['ok'], 'guard_failures': lint['failures'], 'telemetry': {'elapsed_sec': round(elapsed, 3), 'throughput_tps': 0, 'peak_rss_mb': round(peak, 1), 'approx_tokens': n_tok}})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.exception_handler(Exception)
def _on_error(request, exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=500, content={"error": str(exc)})
