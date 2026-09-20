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

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.modes import (
    DEFAULT_MODE,
    needs_answer_regen,
    normalize_mode,
    phase2_messages,
    steering,
)
from src.rag import RAGPipeline, ungrounded_response

STATIC_DIR = Path(__file__).resolve().parent / "static"

CAREFUL_MODE_SUFFIX = (
    "\n\nFor this question, reason through the clinical assessment step by step "
    "first (danger signs, likely causes, what to check), THEN give your final "
    "clear recommendation."
)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    global _sparse
    if _sparse is not None:
        _sparse.stop()
        _sparse = None


app = FastAPI(title="Jamii Afya", lifespan=_lifespan)

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
        server = SparseServer(
            resolve_model_path(),
            arm=os.environ.get("ADTC_SPARSE_ARM", "bounded_3gb"),
            port=port, n_ctx=rt.n_ctx,
            threads=min(rt.n_threads, os.cpu_count() or rt.n_threads),
            poll=int(os.environ.get("ADTC_POLL", "0")))
        # Publish only a ready server.  If startup aborts, the next request
        # must be able to retry instead of reusing a dead SparseServer object.
        server.start()
        _sparse = server
    return _sparse


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []
    careful_mode: bool = False
    mode: str = DEFAULT_MODE  # fast | medium | high (reasoning effort only)


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    telemetry: dict
    model_ready: bool
    thinking: str = ""  # model-emitted reasoning text (Qwen sparse backend)
    urgency: str = "ROUTINE"  # SELF-CARE | ROUTINE | URGENT | EMERGENCY
    safety_override: bool = False  # deterministic rule fired: show prominently
    guidance_cards: list[str] = []  # matched structured-guidance card ids
    mode: str = DEFAULT_MODE  # echo of the effective reasoning mode
    regenerated: bool = False  # phase-2 answer recovery ran


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


_URGENCY_OF = {"LOW": "SELF-CARE", "ROUTINE": "ROUTINE",
               "URGENT": "URGENT", "EMERGENCY": "EMERGENCY",
               "MEDICATION_HIGH_RISK": "URGENT"}
_URGENCY_RANK = {"SELF-CARE": 0, "ROUTINE": 1, "URGENT": 2, "EMERGENCY": 3}

_RETRIEVE_FACTS = {"dosing_request", "high_risk_med", "pregnant",
                   "postpartum", "child_under5", "infant_under2mo",
                   "neonate"}


def _guidance_block(cards: list[dict]) -> str:
    parts = [("STRUCTURED GUIDANCE (follow these; cite only the listed "
              "attributions, never invent WHO/IMCI/NCDC claims):")]
    for c in cards:
        parts.append(f"[CARD {c['id']} | risk={c['risk']} | {c['attribution']}]")
        parts.append("REQUIRED: " + " / ".join(c["required"]))
        parts.append("PROHIBITED: " + " / ".join(c["prohibited"]))
        parts.append("ASK IF MISSING: " + " / ".join(c["missing_info"]))
    return "\n".join(parts)


def _mode_or_400(req: ChatRequest) -> str:
    try:
        return normalize_mode(req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _ground(req: ChatRequest, mode: str):
    """RAG grounding shared by streaming and non-streaming chat."""
    import re

    from runtime.safety import (
        classify_input,
        evaluate_rules,
        extract_facts,
        reminder_for,
        retrieval_payload,
    )
    from src.config import resolve_model_path

    risk = classify_input(req.message)
    reminder = reminder_for(risk)
    facts = extract_facts(req.message)["facts"]
    rules = evaluate_rules(facts)
    risk["facts"] = sorted(facts)
    risk["cards"] = rules["cards"]
    urgency = _URGENCY_OF.get(risk["level"], "ROUTINE")
    rules_urg = {"low": "SELF-CARE", "routine": "ROUTINE",
                 "urgent": "URGENT", "emergency": "EMERGENCY"}[rules["risk"]]
    if _URGENCY_RANK[rules_urg] > _URGENCY_RANK[urgency]:
        urgency = rules_urg
    want_cards = (rules["risk"] in ("urgent", "emergency")
                  or bool(facts & _RETRIEVE_FACTS)
                  or re.search(r"what does (WHO|NCDC|IMCI|UNICEF)\b",
                               req.message, re.IGNORECASE))
    payload = retrieval_payload(rules["cards"]) if want_cards else []
    attributions = [c["attribution"] for c in payload]
    result = _rag.build(req.message, top_n=3)
    sources = [d.get("id", d.get("title", "?")) for d in result.retrieved] if result.is_grounded else []
    if not result.is_grounded:
        return None, ChatResponse(
            reply=ungrounded_response(req.message), sources=[],
            telemetry={"elapsed_sec": 0, "throughput_tps": 0, "peak_rss_mb": 0},
            model_ready=resolve_model_path().exists(), urgency=urgency,
            safety_override=rules["override"],
            guidance_cards=rules["cards"], mode=mode)
    if not resolve_model_path().exists():
        preview = (
            "[Model not downloaded yet — RAG preview only]\n\n"
            f"Retrieved context that would ground the answer:\n{result.context or '(no match found)'}"
        )
        return None, ChatResponse(
            reply=preview, sources=sources,
            telemetry={"elapsed_sec": 0, "throughput_tps": 0, "peak_rss_mb": 0},
            model_ready=False, urgency=urgency,
            safety_override=rules["override"],
            guidance_cards=rules["cards"], mode=mode)
    system_prompt = _rag.system_prompt_for(result) + (
        CAREFUL_MODE_SUFFIX if req.careful_mode else ""
    )
    steer = steering(mode)
    if steer:
        system_prompt += "\n\nREASONING EFFORT: " + steer
    if reminder:
        system_prompt += "\n\nSAFETY NOTE FOR THIS TURN: " + reminder
    if payload:
        system_prompt += "\n\n" + _guidance_block(payload)
    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": t.role, "content": t.content} for t in req.history]
    messages.append({"role": "user", "content": result.user_content})
    meta = {"urgency": urgency, "override": rules["override"],
            "cards": rules["cards"], "attributions": attributions}
    return (messages, sources, risk, meta), None


def _guarded_once(call, risk, prompt, attributions=None):
    """Run one generation, lint it, regen once with a corrective
    instruction on failure, else return the safe fallback. The call takes
    an optional corrective instruction and returns {text, thinking?, tele?}.
    Returns (text, thinking, tele, guard_event)."""
    from runtime.safety import (
        SAFE_FALLBACK,
        corrective_instruction,
        lint_output,
        log_event,
    )

    out = call(None)
    lint = lint_output(risk, prompt, out["text"], attributions,
                       out.get("thinking", ""))
    regened = False
    if not lint["ok"]:
        out = call(corrective_instruction(lint["failures"]))
        lint = lint_output(risk, prompt, out["text"], attributions,
                           out.get("thinking", ""))
        regened = True
    if not lint["ok"]:
        return SAFE_FALLBACK, "", {}, log_event(
            risk, {"lint": lint["failures"], "fallback": True})
    return (out["text"], out.get("thinking", ""), out.get("tele", {}),
            log_event(risk, {"lint": [], "regened": regened}))


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    import time

    from src.modes import budgets as _mode_budgets

    mode = _mode_or_400(req)
    grounded, early = _ground(req, mode)
    if early is not None:
        return early
    messages, sources, risk, meta = grounded
    think_allow, answer_allow = _mode_budgets(mode)
    max1 = think_allow + answer_allow
    regened_answer = False

    def _phase2(primary, ask):
        """Phase-2 answer guarantee: reasoning without an answer triggers
        one bounded final-answer request instead of an empty reply."""
        nonlocal regened_answer
        if needs_answer_regen(primary.get("thinking", ""),
                               primary.get("text", "")):
            regened_answer = True
            ms2 = phase2_messages(messages, primary["thinking"])
            second = ask(ms2, answer_allow)
            return {"text": second.get("text", ""),
                    "thinking": primary.get("thinking", ""),
                    "tele": second.get("tele", primary.get("tele", {}))}
        return primary

    if _backend() == "sparse":
        start = time.time()
        srv = _get_sparse()

        def _call(corrective):
            ms = messages + ([{"role": "user", "content": corrective}]
                             if corrective else [])
            out = dict(srv.chat(ms, max_tokens=max1), tele={})
            return _phase2(
                out, lambda m2, lim: dict(srv.chat(m2, max_tokens=lim),
                                          tele={}))

        text, thinking, _, guard = _guarded_once(
            _call, risk, req.message, meta["attributions"])
        elapsed = max(time.time() - start, 1e-3)
        tele = {"elapsed_sec": round(elapsed, 3),
                "peak_rss_mb": round(srv.rss_mb(), 1), "guard": guard}
        return ChatResponse(reply=text, sources=sources, thinking=thinking,
                            telemetry=tele, model_ready=True,
                            urgency=meta["urgency"],
                            safety_override=meta["override"],
                            guidance_cards=meta["cards"], mode=mode,
                            regenerated=regened_answer)
    engine = _get_engine()

    def _call2(corrective):
        ms = messages + ([{"role": "user", "content": corrective}]
                         if corrective else [])
        r = engine.chat(ms, max_tokens=max1)
        return {"text": r["text"], "thinking": "", "tele": r["telemetry"]}

    text, _, tele, guard = _guarded_once(
        _call2, risk, req.message, meta["attributions"])
    tele = dict(tele)
    tele["guard"] = guard
    return ChatResponse(reply=text, sources=sources, telemetry=tele,
                        model_ready=True, urgency=meta["urgency"],
                        safety_override=meta["override"],
                        guidance_cards=meta["cards"], mode=mode,
                        regenerated=False)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE stream: thinking pieces and answer pieces as separate events.

    Event payloads (JSON in `data:` lines):
      {"kind": "meta", "urgency": ..., "safety_override": ...,
       "guidance_cards": [...], "mode": ...}  FIRST event, before generation
      {"kind": "thinking", "piece": "..."}   model-emitted reasoning text
      {"kind": "text", "piece": "..."}       final-answer text
      {"kind": "done", "reply": ..., "thinking": ..., "sources": [...],
       "telemetry": {...}}                    terminal event (always sent)
      {"kind": "error", "error": "..."}       terminal on failure
    """
    import json
    import time

    from src.modes import budgets as _mode_budgets

    mode = _mode_or_400(req)
    grounded, early = _ground(req, mode)
    if early is not None:
        final = dict(early.model_dump() if hasattr(early, "model_dump") else early.dict())
        final["kind"] = "done"

        def _early():
            yield f"data: {json.dumps(final)}\n\n"
        return StreamingResponse(_early(), media_type="text/event-stream")

    messages, sources, risk, meta = grounded
    think_allow, answer_allow = _mode_budgets(mode)
    max1 = think_allow + answer_allow

    def _gen():
        from runtime.safety import lint_output, log_event

        # Deterministic safety verdict FIRST: emergencies render immediately,
        # never after High-mode reasoning completes.
        yield f"data: {json.dumps({'kind': 'meta', 'urgency': meta['urgency'], 'safety_override': meta['override'], 'guidance_cards': meta['cards'], 'mode': mode})}\n\n"
        start = time.time()
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        regened_answer = False
        try:
            if _backend() == "sparse":
                srv = _get_sparse()
                for kind, piece in srv.stream_chat(messages, max_tokens=max1):
                    (thinking_parts if kind == "thinking" else text_parts).append(piece)
                    yield f"data: {json.dumps({'kind': kind, 'piece': piece})}\n\n"
                peak = srv.rss_mb()
            else:
                engine = _get_engine()
                for piece in engine.stream_chat(messages, max_tokens=max1):
                    text_parts.append(piece)
                    yield f"data: {json.dumps({'kind': 'text', 'piece': piece})}\n\n"
                peak = 0.0
        except Exception as exc:  # noqa: BLE001 — any stream fault -> error event
            yield f"data: {json.dumps({'kind': 'error', 'error': str(exc)[:500]})}\n\n"
            return
        elapsed = max(time.time() - start, 1e-3)
        reply = "".join(text_parts)
        thinking = "".join(thinking_parts)
        if _backend() == "sparse" and needs_answer_regen(thinking, reply):
            # Streamed phase 2: bounded final-answer request, re-emitted as
            # text events so the UI path stays identical.
            try:
                ms2 = phase2_messages(messages, thinking)
                second = srv.chat(ms2, max_tokens=answer_allow)
                reply = second.get("text", "")
                for i in range(0, len(reply), 200):
                    yield f"data: {json.dumps({'kind': 'text', 'piece': reply[i:i+200]})}\n\n"
                regened_answer = True
            except Exception as exc:  # noqa: BLE001 — phase-2 fault -> error event
                yield f"data: {json.dumps({'kind': 'error', 'error': str(exc)[:500]})}\n\n"
                return
        n_tok = len(reply.split())  # approx; exact usage comes from headers
        # End-of-stream lint (advisory: the client shows a caution banner on
        # failure; regen inside a live stream would double latency, so the
        # non-stream endpoint is the regen path).
        lint = lint_output(risk, req.message, reply, meta["attributions"],
                           thinking)
        log_event(risk, {"lint": lint["failures"], "stream": True})
        yield f"data: {json.dumps({'kind': 'done', 'reply': reply, 'thinking': thinking, 'sources': sources, 'urgency': meta['urgency'], 'safety_override': meta['override'], 'guidance_cards': meta['cards'], 'mode': mode, 'regenerated': regened_answer, 'guard_ok': lint['ok'], 'guard_failures': lint['failures'], 'telemetry': {'elapsed_sec': round(elapsed, 3), 'throughput_tps': 0, 'peak_rss_mb': round(peak, 1), 'approx_tokens': n_tok}})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.exception_handler(Exception)
def _on_error(request, exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=500, content={"error": str(exc)})
