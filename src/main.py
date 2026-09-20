"""Jamii Afya — offline bilingual (English/Kiswahili) clinical advisor CLI.

Usage:
    PYTHONPATH=. python -m src.main                 # interactive chat
    PYTHONPATH=. python -m src.main --query "..."   # one-shot
    PYTHONPATH=. python -m src.main --demo          # run the metadata test prompts
    PYTHONPATH=. python -m src.main --no-rag        # disable retrieval; return safe referral

Runs end-to-end WITH model weights. Without weights it degrades to a
"RAG preview" that shows the retrieved+compressed clinical context, so the
pipeline is demonstrable and testable before ./download_model.sh is run.
"""

from __future__ import annotations

import argparse
import sys

from src.config import load_metadata, resolve_model_path
from src.rag import RAGPipeline, ungrounded_response


def _print_header(domain: str, model_ready: bool) -> None:
    print("=" * 70)
    print("  Jamii Afya — Offline Clinical Advisor (ADTC 2026, healthcare_medical)")
    print("=" * 70)
    print(f"  Domain: {domain} | Languages: English + Kiswahili")
    print(f"  Model:  {'ready' if model_ready else 'NOT downloaded (RAG-preview mode)'}")
    print("  Note:   Clinical decision support — not a substitute for a clinician.")
    print("=" * 70)


def _answer(engine, rag: RAGPipeline, query: str, args) -> None:
    result = rag.build(query, top_n=args.top_n) if not args.no_rag else rag.build(query, top_n=0)
    if result.is_grounded:
        srcs = ", ".join(d.get("id", d.get("title", "?")) for d in result.retrieved)
        print(f"\n[RAG] Retrieved: {srcs}  ({len(result.context.split())} context words)")

    if engine is None:
        if not result.is_grounded:
            print("\n--- Advisory ---")
            print(ungrounded_response(query))
        else:
            print("\n[RAG preview — model not downloaded]")
            print("Retrieved clinical context that would ground the answer:\n")
            print(result.context)
        return

    if not result.is_grounded:
        print("\n--- Advisory ---")
        print(ungrounded_response(query))
        return

    print("\n--- Advisory ---")
    if args.no_stream:
        out = engine.generate(
            result.user_content, system_prompt=rag.system_prompt, max_tokens=args.max_tokens
        )
        print(out["text"])
        t = out["telemetry"]
        print(
            f"\n[telemetry] {t['throughput_tps']} tok/s | "
            f"{t['completion_tokens']} tok | {t['elapsed_sec']}s | "
            f"peak RSS {t['peak_rss_mb']} MB"
        )
    else:
        for piece in engine.stream(
            result.user_content, system_prompt=rag.system_prompt, max_tokens=args.max_tokens
        ):
            print(piece, end="", flush=True)
        print()


class _SparseCLI:
    """Minimal generate/stream adapter over the managed sparse server."""

    def __init__(self):
        import os

        from src.config import get_runtime_config
        from src.sparse import SparseServer, free_port

        rt = get_runtime_config()
        port = int(os.environ.get("ADTC_SPARSE_PORT", "0")) or free_port()
        self._srv = SparseServer(
            resolve_model_path(),
            arm=os.environ.get("ADTC_SPARSE_ARM", "bounded_3gb"),
            port=port, n_ctx=rt.n_ctx,
            threads=min(rt.n_threads, os.cpu_count() or rt.n_threads),
            poll=int(os.environ.get("ADTC_POLL", "0")))
        print("[cli] starting sparse backend (first load takes a minute)...")
        self._srv.start()

    def generate(self, prompt, system_prompt=None, max_tokens=512):
        import time

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        start = time.time()
        out = self._srv.chat(messages, max_tokens=max_tokens)
        el = max(time.time() - start, 1e-3)
        return {"text": out["text"], "telemetry": {
            "elapsed_sec": round(el, 3), "throughput_tps": 0,
            "completion_tokens": 0, "peak_rss_mb": self._srv.rss_mb()}}

    def stream(self, prompt, system_prompt=None, max_tokens=512):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        for kind, piece in self._srv.stream_chat(messages,
                                                 max_tokens=max_tokens):
            if kind == "text":
                yield piece


def _load_engine():
    """Build the engine if weights exist, else return None (RAG-preview mode)."""
    if not resolve_model_path().exists():
        return None
    try:
        if "Q2K-experts" in resolve_model_path().name:
            return _SparseCLI()
        from src.engine import MedicalLLMEngine

        return MedicalLLMEngine()
    except Exception as e:  # noqa: BLE001 — any backend failure degrades to preview
        print(f"[warn] Could not initialise engine ({e}); using RAG-preview mode.")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jamii Afya offline clinical advisor")
    parser.add_argument("--query", type=str, help="Single question, then exit")
    parser.add_argument("--demo", action="store_true", help="Run the metadata test prompts")
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable retrieval (returns safe referral; never generates ungrounded advice)",
    )
    parser.add_argument("--no-stream", action="store_true", help="Print full answer at once")
    parser.add_argument("--top-n", type=int, default=3, help="Docs to retrieve (default 3)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--mode", type=str, default="medium",
                        help="Reasoning mode: fast | medium | high (default medium)")
    args = parser.parse_args(argv)
    from src.modes import normalize_mode, phase1_max_tokens
    try:
        args.mode = normalize_mode(args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_tokens == 512:
        args.max_tokens = phase1_max_tokens(args.mode)

    meta = load_metadata()
    rag = RAGPipeline()
    engine = _load_engine()
    _print_header(meta.get("domain", "healthcare_medical"), engine is not None)
    print(f"Loaded {len(rag.retriever)} clinical guidelines into the offline BM25 index.")

    if args.query:
        _answer(engine, rag, args.query, args)
        return 0

    if args.demo:
        for tp in meta.get("test_prompts", []):
            print("\n" + "-" * 70)
            print(f"[{tp.get('prompt_id', '?')}] {tp['prompt']}")
            _answer(engine, rag, tp["prompt"], args)
        return 0

    # interactive
    print("\nType a clinical question (English or Kiswahili). Ctrl-C or 'exit' to quit.")
    try:
        while True:
            query = input("\n> ").strip()
            if query.lower() in {"exit", "quit", ""}:
                if query == "":
                    continue
                break
            _answer(engine, rag, query, args)
    except (KeyboardInterrupt, EOFError):
        print("\nKwaheri! (Goodbye)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
