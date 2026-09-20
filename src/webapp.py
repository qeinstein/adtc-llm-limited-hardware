"""Jamii Afya's local chat application.

Each request follows one path:

    system prompt -> relevant offline RAG context (when available) -> model

The application does not classify the question, attach urgency labels, rewrite
the model's response, or run a second corrective generation. The model owns
the answer; the application only assembles the conversation and streams it.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.rag import RAGPipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    global _sparse
    if _sparse is not None:
        _sparse.stop()
        _sparse = None


app = FastAPI(title="Jamii Afya", lifespan=_lifespan)

_engine = None
_sparse = None
_rag = RAGPipeline()


def _get_engine():
    global _engine
    if _engine is None:
        from src.engine import MedicalLLMEngine

        _engine = MedicalLLMEngine()
    return _engine


def _backend() -> str:
    """Select the sparse server for the shipped Q2K model."""
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
        server = SparseServer(
            resolve_model_path(),
            arm=os.environ.get("ADTC_SPARSE_ARM", "bounded_3gb"),
            port=port,
            n_ctx=rt.n_ctx,
            threads=min(rt.n_threads, os.cpu_count() or rt.n_threads),
            poll=int(os.environ.get("ADTC_POLL", "0")),
            reasoning_budget=rt.reasoning_budget,
        )
        server.start()
        _sparse = server
    return _sparse


class ChatTurn(BaseModel):
    # Only the two conversation roles are client-controlled.  In particular,
    # never allow a browser/API caller to insert a second system message after
    # the trusted system prompt.
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = Field(default_factory=list)
    telemetry: dict = Field(default_factory=dict)
    model_ready: bool
    thinking: str = ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/chat")
def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/api/health")
def health() -> dict:
    from src.config import resolve_model_path

    return {
        "model_downloaded": resolve_model_path().exists(),
        "model": resolve_model_path().name,
        "backend": _backend(),
        "guidelines_loaded": len(_rag.retriever),
    }


def _prepare(req: ChatRequest):
    """Build the system/user messages and nothing else."""
    from src.config import resolve_model_path

    result = _rag.build(req.message, top_n=3)
    sources = (
        [d.get("id", d.get("title", "?")) for d in result.retrieved]
        if result.is_grounded
        else []
    )
    model_path = resolve_model_path()
    if not model_path.exists():
        context = result.context or "No matching offline reference was found."
        return None, ChatResponse(
            reply=(
                "The model is not downloaded yet. Run `make model` and try again.\n\n"
                f"RAG context available for this question:\n{context}"
            ),
            sources=sources,
            telemetry={"elapsed_sec": 0},
            model_ready=False,
        )

    system_prompt = _rag.system_prompt_for(result)
    from src.config import get_runtime_config

    gen = _generation_config()
    messages = [{"role": "system", "content": system_prompt}]
    messages += _bounded_history(
        req.history,
        n_ctx=get_runtime_config().n_ctx,
        reserved_chars=len(system_prompt) + len(result.user_content),
        max_completion_tokens=gen.max_tokens,
    )
    messages.append({"role": "user", "content": result.user_content})
    return (messages, sources), None


def _telemetry(elapsed: float, peak: float, usage: dict | None = None) -> dict:
    usage = usage or {}
    completion = int(usage.get("completion_tokens", 0) or 0)
    timings = usage.get("timings") or {}
    runtime_tps = float(timings.get("predicted_per_second", 0) or 0)
    throughput = runtime_tps if runtime_tps > 0 else completion / max(elapsed, 1e-3)
    return {
        "elapsed_sec": round(max(elapsed, 1e-3), 3),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": completion,
        "throughput_tps": round(throughput, 2),
        "peak_rss_mb": round(peak, 1),
    }


def _generation_config():
    from src.config import get_generation_config

    return get_generation_config()


def _bounded_history(
    history: list[ChatTurn],
    *,
    n_ctx: int = 4096,
    reserved_chars: int = 0,
    max_completion_tokens: int = 2048,
) -> list[dict[str, str]]:
    """Keep the newest complete conversation within the serving context.

    The model context includes the system prompt, optional RAG block, current
    question, and generated answer. Sending unbounded browser history can make
    the runtime truncate the beginning of the prompt—the least visible but most
    important part. We bound history by characters as a portable approximation
    that works for the sparse server and the llama-cpp fallback without a
    tokenizer dependency.
    """
    import os

    def configured(name: str, default: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except ValueError:
            value = default
        return max(1, value)

    configured_limit = configured("ADTC_HISTORY_CHARS", 10000)
    # There is no tokenizer dependency in the web layer. Four characters per
    # token is a conservative cross-language approximation; the reserved
    # allowance includes the trusted prompt, current user/RAG block, and the
    # requested completion budget. Explicit ADTC_HISTORY_CHARS can only lower
    # this safe limit, never raise it past the context window.
    reserved_tokens = (max(0, reserved_chars) + 3) // 4
    history_tokens = max(
        0,
        int(n_ctx) - max(0, int(max_completion_tokens)) - reserved_tokens,
    )
    limit = min(configured_limit, history_tokens * 4)
    if limit <= 32:
        return []
    turn_limit = max(1000, min(limit, configured("ADTC_HISTORY_TURN_CHARS", 5000)))
    selected: list[dict[str, str]] = []
    used = 0
    for turn in reversed(history):
        content = turn.content.strip()
        if not content:
            continue
        if len(content) > turn_limit:
            content = content[:turn_limit] + "\n[Earlier part of this turn omitted.]"
        cost = len(content) + 32
        if selected and used + cost > limit:
            break
        if not selected and cost > limit:
            content = content[:max(1, limit - 32)]
            cost = len(content) + 32
        selected.append({"role": turn.role, "content": content})
        used += cost

    selected.reverse()
    # A clipped window must not begin with an assistant answer detached from
    # the user turn that caused it.
    while selected and selected[0]["role"] == "assistant":
        selected.pop(0)
    return selected


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    prepared, early = _prepare(req)
    if early is not None:
        return early

    messages, sources = prepared
    gen = _generation_config()
    start = time.time()
    if _backend() == "sparse":
        server = _get_sparse()
        out = server.chat(
            messages,
            max_tokens=gen.max_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p,
        )
        elapsed = time.time() - start
        return ChatResponse(
            reply=out.get("text", ""),
            thinking=out.get("thinking", ""),
            sources=sources,
            telemetry=_telemetry(elapsed, server.rss_mb(), out.get("usage")),
            model_ready=True,
        )

    engine = _get_engine()
    out = engine.chat(messages, generation=gen)
    return ChatResponse(
        reply=out.get("text", ""),
        sources=sources,
        telemetry=out.get("telemetry", {}),
        model_ready=True,
    )


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """Stream model-emitted thinking/text and finish with the raw response."""
    import json

    prepared, early = _prepare(req)
    if early is not None:
        final = dict(early.model_dump() if hasattr(early, "model_dump") else early.dict())
        final["kind"] = "done"

        def _early():
            yield f"data: {json.dumps(final)}\n\n"

        return StreamingResponse(_early(), media_type="text/event-stream")

    messages, sources = prepared
    gen = _generation_config()

    def _gen():
        start = time.time()
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        peak = 0.0
        usage: dict = {}
        try:
            if _backend() == "sparse":
                server = _get_sparse()
                events = getattr(server, "stream_chat_events", None)
                if events is None:
                    events = server.stream_chat
                for kind, piece in events(
                    messages, max_tokens=gen.max_tokens,
                    temperature=gen.temperature, top_p=gen.top_p,
                ):
                    if kind == "usage":
                        if isinstance(piece, dict):
                            usage.update(piece)
                        continue
                    (thinking_parts if kind == "thinking" else text_parts).append(piece)
                    yield f"data: {json.dumps({'kind': kind, 'piece': piece})}\n\n"
                peak = server.rss_mb()
            else:
                engine = _get_engine()
                events = getattr(engine, "stream_chat_events", None)
                if events is None:
                    for piece in engine.stream_chat(messages, generation=gen):
                        text_parts.append(piece)
                        yield f"data: {json.dumps({'kind': 'text', 'piece': piece})}\n\n"
                else:
                    for kind, piece in events(messages, generation=gen):
                        if kind == "usage":
                            if isinstance(piece, dict):
                                usage.update(piece)
                            continue
                        (thinking_parts if kind == "thinking" else text_parts).append(piece)
                        yield f"data: {json.dumps({'kind': kind, 'piece': piece})}\n\n"
        except Exception as exc:  # noqa: BLE001 - stream must report failures
            yield f"data: {json.dumps({'kind': 'error', 'error': str(exc)[:500]})}\n\n"
            return

        reply = "".join(text_parts)
        thinking = "".join(thinking_parts)
        telemetry = _telemetry(time.time() - start, peak, usage)
        yield f"data: {json.dumps({'kind': 'done', 'reply': reply, 'thinking': thinking, 'sources': sources, 'telemetry': telemetry})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.exception_handler(Exception)
def _on_error(request, exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=500, content={"error": str(exc)})
