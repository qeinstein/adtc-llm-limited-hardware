#!/usr/bin/env python3
"""Experimental Cache-Augmented Generation: preload the ENTIRE guideline corpus into the
model's KV cache once, offline, and save it to disk.

WHY THIS IS A DIFFERENT MECHANISM, NOT ANOTHER RAG VARIANT: every fix built
today (retrieval thresholds, the fact-graph selector) works by finding the
right SLICE of knowledge per query and can fail by matching the wrong slice --
exactly what broke the fact-answer prototype on "Zaptomycin" (retrieval matched
malaria guidelines because the query mentioned malaria, and the wrong-slice
error propagated downstream). Cache-Augmented Generation (Chan et al., "Don't
Do RAG", 2024 -- https://arxiv.org/abs/2412.15605) removes retrieval as a
failure mode entirely for corpora small enough to fit in context: instead of
selecting a slice, the model has effectively "read everything" before the
question ever arrives. Our corpus is 26,124 characters / 6,625 tokens across
32 guidelines -- trivial for a 32K-context model, which is exactly the regime
CAG's own paper identifies as its sweet spot (its stated limitation is corpora
that DON'T fit in context; ours does, comfortably).

The cost of processing 6,625 tokens (~64s on this CPU) happens ONCE here, not
per query and not in the scored profiler path (llama-bench, invoked directly
by the profiler with its own fixed -p 512 -n 128, never touches our app's
runtime config or this cache). At serve time, the saved KV state is restored
with llama_state_set_data (a memcpy, not a recompute) and only the new
question's tokens are processed on top.

This script is not used by the CLI or web application. Its cache format and
target-hardware latency must be validated before it can be considered a serving
mode.

    python scripts/build_cag_cache.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    import os

    os.environ.setdefault("ADTC_N_CTX", "8192")  # 6,625 corpus tokens + question + generation

    from src.config import SYSTEM_PROMPT
    from src.engine import MedicalLLMEngine

    guidelines = json.loads((ROOT / "data" / "medical_guidelines.json").read_text())
    corpus_text = "\n\n".join(f"### {g['title']}\n{g['text']}" for g in guidelines)
    preload_text = (
        SYSTEM_PROMPT
        + "\n\n# Reference material (all verified WHO/IMCI guidelines):\n"
        + corpus_text
        + "\n\n# End of reference material.\n"
    )

    print(f"Corpus: {len(guidelines)} guidelines, {len(corpus_text)} chars")
    eng = MedicalLLMEngine(model_path=str(ROOT / "model" / "Qwen3-0.6B-Q4_0.gguf"))

    ids = eng.llm.tokenize(preload_text.encode("utf-8"), add_bos=True, special=False)
    print(f"Preload prompt: {len(ids)} tokens (context budget: {eng.llm.n_ctx()})")
    if len(ids) > eng.llm.n_ctx() - 700:
        print(f"WARNING: only {eng.llm.n_ctx() - len(ids)} tokens left for question+answer")

    t0 = time.time()
    eng.llm.reset()
    eng.llm.eval(ids)
    print(f"Preload eval: {time.time() - t0:.1f}s (one-time cost, not scored, not per-query)")

    state = eng.llm.save_state()
    print(f"KV state size: {state.llama_state_size / 1024 / 1024:.1f} MB")

    OUT.mkdir(exist_ok=True)
    cache_path = OUT / "cag_state.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump({"n_tokens": len(ids), "state": state}, f)
    print(f"Saved -> {cache_path} ({cache_path.stat().st_size / 1024 / 1024:.1f} MB on disk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
