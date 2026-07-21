"""Accuracy prediction harness — mirrors how the adtc-profiler scores S_acc.

The profiler's automated accuracy is EleutherAI lm-eval-harness run on the RAW
GGUF (default task ``arc_easy``, ``acc_norm``, limit 50; audit uses a hidden
domain subset). RAG does NOT affect this number — it is pure base-model capability
— so we predict it here on the same footing, and additionally probe medical MCQA
tasks the hidden healthcare subset likely resembles.

Requires: a downloaded GGUF and ``pip install lm-eval`` + ``llama-cpp-python``.
Without them this prints exactly what to install and exits cleanly (no fabrication).

Usage:
    PYTHONPATH=. python -m src.accuracy --tasks arc_easy medmcqa --limit 50
    PYTHONPATH=. python -m src.accuracy --domain-eval      # our offline EN/SW recall
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src.config import EVAL_SET_PATH, resolve_model_path


def run_lm_eval(model_path: Path, task: str, limit: int = 50, seed: int = 42) -> dict:
    """Run lm-eval exactly as the profiler does; return {task, score, metric}."""
    if shutil.which("lm_eval") is None:
        raise RuntimeError("lm_eval not found. Install with: pip install lm-eval")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "res"
        cmd = [
            "lm_eval",
            "--model", "gguf",
            "--model_args", f"base_url=local,pretrained={model_path}",
            "--tasks", task,
            "--limit", str(limit),
            "--seed", str(seed),
            "--output_path", str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # lm-eval writes results_*.json under the output path
        files = list(Path(tmp).rglob("results*.json"))
        if not files:
            raise RuntimeError("lm-eval produced no results file")
        data = json.loads(files[0].read_text())
        res = data.get("results", {}).get(task, {})
        for key, metric in (("acc_norm,none", "acc_norm"), ("acc,none", "acc")):
            if key in res:
                return {"task": task, "score": round(float(res[key]) * 100, 2), "metric": metric}
        return {"task": task, "score": 0.0, "metric": "none"}


def _domain_eval() -> None:
    """Run our offline EN/SW clinical concept-recall eval against the live engine."""
    from src.engine import MedicalLLMEngine
    from src.evaluator import ConceptRecallEvaluator
    from src.rag import RAGPipeline

    engine = MedicalLLMEngine()
    rag = RAGPipeline()

    def answer(query: str) -> str:
        res = rag.build(query)
        return engine.generate(res.user_content, system_prompt=rag.system_prompt, max_tokens=256)["text"]

    ev = ConceptRecallEvaluator.from_json(EVAL_SET_PATH)
    report = ev.evaluate(answer, verbose=True)
    print(f"\nMean clinical concept recall: {report['mean_accuracy'] * 100:.1f}% over {report['n_cases']} cases")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Predict ADTC S_acc via lm-eval + domain eval")
    ap.add_argument("--tasks", nargs="+", default=["arc_easy"],
                    help="lm-eval tasks (e.g. arc_easy medmcqa pubmedqa mmlu_clinical_knowledge)")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domain-eval", action="store_true", help="Run our offline EN/SW concept recall")
    args = ap.parse_args(argv)

    model_path = resolve_model_path()
    if not model_path.exists():
        print(f"[accuracy] Model not found at {model_path}. Run ./download_model.sh first.")
        return 2

    if args.domain_eval:
        _domain_eval()
        return 0

    print(f"[accuracy] Predicting S_acc for {model_path.name} (mirrors adtc-profiler)\n")
    scores = []
    for task in args.tasks:
        try:
            r = run_lm_eval(model_path, task, args.limit, args.seed)
            scores.append(r["score"])
            print(f"  {r['task']:32s} {r['metric']}: {r['score']:.2f}")
        except Exception as e:
            print(f"  {task:32s} ERROR: {e}")
    if scores:
        print(f"\n  Mean over tasks (S_acc estimate): {sum(scores) / len(scores):.2f} / 100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
