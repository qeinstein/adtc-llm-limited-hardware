"""ADTC 2026 — Offline Medical Advisor (Jamii Afya).

Package modules:
- config:      loads the profiler manifest (metadata.json) + runtime knobs
- retriever:   stdlib BM25 sparse retrieval over the bilingual clinical corpus
- compressor:  query-focused extractive context compression
- rag:         retrieve -> compress -> prompt assembly
- engine:      llama-cpp-python CPU serving wrapper (the interactive product)
- evaluator:   offline bilingual clinical concept-recall evaluation
"""

__version__ = "0.2.0"
