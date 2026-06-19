# System Implementation Plan: Custom Hybrid Serving Sandbox with Page-Cache Swapping

This document outlines the architecture and execution pipeline for our custom **Hybrid LLM Serving Sandbox** running a large model (Qwen 2.5 14B or Mistral Small 24B Q4_K_M) on a **5.5 GB RAM / Core i5** limit.

---

## 🎯 Architectural Goal
Build a highly custom, competitive serving sandbox that:
1. Serves a large-scale model (10 GB Qwen 2.5 14B or 15 GB Mistral Small 24B Q4_K_M) on a 5.5 GB RAM ceiling.
2. Implements a **Custom Page-Cache Paging Controller** (`src/memory_manager.py`) to bypass or evict model weight pages from physical RAM on-the-fly during decoding.
3. Keeps active Resident Set Size (RSS) **strictly under 5.5 GB** at all times.
4. Natively claims the **+15% African Alpha Bonus** using multilingual Swahili query evaluation.

---

## 🏗️ Core System Components

### 1. Dynamic Page-Cache Controller
*   **Module:** `src/memory_manager.py`
*   **Method:**
    *   *Linux (Target):* Periodically runs `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` and parses `/proc/self/maps` to call `madvise(addr, len, MADV_DONTNEED)` on the mapped GGUF file.
    *   *macOS (Local):* Uses `fcntl(fd, F_NOCACHE, 1)` on the model weight file descriptor to run Direct I/O and bypass OS page caching.
*   **Result:** The OS loads weights dynamically from SSD during matrix execution, and immediately drops them, keeping memory footprint low.

### 2. Hybrid Llama.cpp Bindings
*   **Module:** `src/inference.py`
*   **Method:** Hooks into the dynamic library wrapper to execute model layers at C++ speed, calling the memory manager after token decoding.

### 3. NumPy TurboQuant Engine & db_retriever
*   **Modules:** `src/turboquant_numpy.py` & `src/db_retriever.py`
*   **Method:** Manages the quantized Key/Value cache (Q4/Q8) and handles BM25 sparse RAG.

---

## 📁 Repository Structure

```
adtc-llm-limited-hardware/
├── plan.md                ← This document.
├── metadata.json          ← Submitter and model configuration metadata.
├── download_model.sh      ← Downloads the 14B or 24B GGUF weights.
├── src/
│   ├── main.py            ← Runner for prompt benchmarks and language checks.
│   ├── memory_manager.py  ← OS virtual memory and page-cache controller.
│   ├── inference.py       ← Dynamic bindings and attention hooks.
│   └── turboquant_numpy.py← Port of TurboQuant to NumPy.
└── venv/                  ← Python virtual environment.
```

---

## 🛠️ Execution checklist

*   **Phase 1:** Update the download script and metadata for the selected model.
*   **Phase 2:** Implement `src/memory_manager.py` with platform-specific system calls.
*   **Phase 3:** Update the inference runner to trigger page cache eviction.
*   **Phase 4:** Benchmark, profile RAM RSS, and verify the model runs successfully.
