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
from fastapi.responses import FileResponse, JSONResponse
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
_rag = RAGPipeline()


def _get_engine():
    global _engine
    if _engine is None:
        from src.engine import MedicalLLMEngine

        _engine = MedicalLLMEngine()
    return _engine


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

    return {"model_downloaded": resolve_model_path().exists(), "guidelines_loaded": len(_rag.retriever)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    from src.config import resolve_model_path

    result = _rag.build(req.message, top_n=3)
    sources = [d.get("id", d.get("title", "?")) for d in result.retrieved] if result.is_grounded else []

    # Never ask the model for clinical management from parametric memory when
    # the reviewed local corpus does not support the question.
    if not result.is_grounded:
        return ChatResponse(
            reply=ungrounded_response(req.message), sources=[],
            telemetry={"elapsed_sec": 0, "throughput_tps": 0, "peak_rss_mb": 0},
            model_ready=resolve_model_path().exists(),
        )

    if not resolve_model_path().exists():
        preview = (
            "[Model not downloaded yet — RAG preview only]\n\n"
            f"Retrieved context that would ground the answer:\n{result.context or '(no match found)'}"
        )
        return ChatResponse(
            reply=preview, sources=sources,
            telemetry={"elapsed_sec": 0, "throughput_tps": 0, "peak_rss_mb": 0},
            model_ready=False,
        )

    # Few-shot exemplars only when we actually retrieved grounding — otherwise a
    # small model copies them verbatim for any low-signal input (see
    # RAGPipeline.system_prompt_for).
    system_prompt = _rag.system_prompt_for(result) + (
        CAREFUL_MODE_SUFFIX if req.careful_mode else ""
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": t.role, "content": t.content} for t in req.history]
    messages.append({"role": "user", "content": result.user_content})

    engine = _get_engine()
    out = engine.chat(messages, max_tokens=512)
    return ChatResponse(reply=out["text"], sources=sources, telemetry=out["telemetry"], model_ready=True)


@app.exception_handler(Exception)
def _on_error(request, exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=500, content={"error": str(exc)})
