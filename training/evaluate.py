#!/usr/bin/env python3
"""3-stage evaluation entrypoint: (A) native reference, (B) tuned merged BF16,
(C) tuned sparse deployment (Q2_K + K4/16). UNEXECUTED beyond stage-A/CPU
baselines (see the edge0base Kaggle kernel for the runnable CPU subset).

This script defines the track registry + paired-comparison math. Heavy
tracks (HealthBench, MedHELM, MedSafetyBench, MedHallu) run where GPU
inference exists; the script fails closed when a track's backend is absent.

Usage:
  python training/evaluate.py --stage A --out eval_runs/stageA.json
  python training/evaluate.py --compare eval_runs/stageA.json eval_runs/stageC.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

TRACKS = {
    # track: (kind, source, n_or_subset, backend)
    "mmlu200": ("mcq-logprob", "ikawrakow mmlu-test.bin (frozen harness)", 200, "cpu-gguf"),
    "afrimed-mcq": ("mcq-logprob", "afrimedqa_v2 test mcq sample seed 7", 500, "cpu-gguf"),
    "swahili18": ("keyword-recall", "data/swahili_eval_set.json", 18, "cpu-gguf"),
    "safety-bank": ("generation+rules", "edge0base SAFETY_BANK (35 prompts)", 35, "cpu-gguf"),
    "medqa-test": ("mcq-logprob", "GBaker test (1273)", 1273, "gpu-needed"),
    "medmcqa-valtest": ("mcq-logprob", "openlifescienceai val+test", 10333, "gpu-needed"),
    "pubmedqa-labeled": ("gen+judge", "qiaojin pqa_labeled (1k)", 1000, "gpu-needed"),
    "healthbench": ("harness", "openai/healthbench (if executable)", None, "gpu-needed"),
    "medhelm": ("harness", "stanford medhelm public tasks", None, "gpu-needed"),
    "medsafetybench": ("harness", "MedSafetyBench EVAL ONLY", None, "gpu-needed"),
    "medhallu": ("harness", "MedHallu/MHB if available", None, "gpu-needed"),
    "oasst-heldout": ("perplexity-pair", "OASST1 val slice (forgetting check)", 500, "gpu-needed"),
}


def paired_bootstrap(a: list[bool], b: list[bool], iters: int = 2000,
                     seed: int = 7) -> dict:
    """Paired accuracy delta with bootstrap CI (same items, same order)."""
    assert len(a) == len(b) and a
    rng = random.Random(seed)
    n = len(a)
    base = (sum(b) - sum(a)) / n
    deltas = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append((sum(b[i] for i in idx) - sum(a[i] for i in idx)) / n)
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[int(0.975 * iters) - 1]
    return {"delta": base, "ci95": [lo, hi],
            "n": n, "acc_a": sum(a) / n, "acc_b": sum(b) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "C"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        out = {}
        for track in a.get("tracks", {}):
            if track in b.get("tracks", {}) and "paired" in a["tracks"][track]:
                pa = a["tracks"][track]["paired"]
                pb = b["tracks"][track]["paired"]
                out[track] = paired_bootstrap(pa, pb)
        print(json.dumps(out, indent=1))
        return
    print("stage", args.stage, "tracks:")
    runnable, deferred = [], []
    for name, (kind, src, n, backend) in TRACKS.items():
        (runnable if backend == "cpu-gguf" else deferred).append(name)
        print(f"  {name}: {kind} [{backend}]")
    print(f"\nrunnable on CPU GGUF now: {runnable}")
    print(f"deferred to GPU inference: {deferred}")
    print("\nCPU subset runs via the edge0base kernel (see kaggle/native-sparse-edge0base-v1).")
    print("GPU tracks fail closed until inference compute exists. Nothing executed.")


if __name__ == "__main__":
    main()
