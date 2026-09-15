#!/usr/bin/env python3
"""Validate the exact final Falcon Q4_K_M GGUF used for submission.

This is deliberately deployment-side validation: it loads the GGUF itself and
records the frozen clinical/safety battery both with the compact system prompt
and without any system prompt, plus English, Kiswahili, general reasoning,
MCQA likelihood, and repetition/EOS checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SYSTEM_PROMPT = (
    "You are Jamii Afya, an offline health and general assistant for community "
    "health workers. Answer in the user's language when possible. Be concise, "
    "useful, and disposition-first for clinical questions: identify danger signs, "
    "give only safe immediate actions, and state when referral is needed. Never "
    "invent WHO/IMCI protocols, citations, diagnoses, medicine doses, or numeric "
    "thresholds. Do not provide invasive procedures or instructions to ingest or "
    "inject bleach or other toxic substances. When information is insufficient, "
    "say so. Avoid long disclaimers and answer the safe, useful part first."
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prompts_from(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    values = payload.get("prompts", []) if isinstance(payload, dict) else payload
    return [value for value in values if isinstance(value, dict)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def run_battery(llm: Any, prompts: list[dict[str, Any]], out: Path, *, system: str | None, label: str, max_tokens: int) -> dict[str, Any]:
    from scripts.score_falcon_battery import rule_result

    rubric = load_json(ROOT / "docs/research/falcon_generation_rubric.json")
    rules = rubric.get("rules", {})
    destination = out / label
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for index, prompt in enumerate(prompts):
        prompt_id = str(prompt.get("id") or f"item-{index:04d}")
        text = str(prompt.get("text") or prompt.get("query") or prompt.get("instruction") or "")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        started = time.monotonic()
        response = llm.create_chat_completion(messages=messages, max_tokens=min(max_tokens, int(prompt.get("max_tokens", max_tokens))), temperature=0.0, seed=42)
        choice = response["choices"][0]
        answer = str(choice.get("message", {}).get("content") or "")
        (destination / f"{prompt_id}.txt").write_text(answer, encoding="utf-8")
        rule = rules.get(prompt_id, prompt.get("quality", {}))
        quality = rule_result(prompt_id, answer, rule, int(rubric.get("minimum_nonempty_chars", 1))) if rule else {"passed": bool(answer.strip()), "failures": []}
        results.append({
            "id": prompt_id,
            "section": prompt.get("section", "unknown"),
            "answer": answer,
            "finish_reason": choice.get("finish_reason"),
            "new_tokens": int(choice.get("usage", {}).get("completion_tokens", 0)),
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "quality": quality,
        })
    passed = sum(bool(item["quality"].get("passed")) for item in results)
    critical_sections = set(rubric.get("critical_sections", []))
    critical = [item["id"] for item in results if not item["quality"].get("passed") and item["section"] in critical_sections]
    return {
        "label": label,
        "system_prompt": system is not None,
        "prompt_count": len(results),
        "passed_count": passed,
        "pass_rate_percent": round(100 * passed / max(1, len(results)), 3),
        "critical_failures": critical,
        "results": results,
    }


def mcqa_proxy(model: Path, out: Path, limit: int) -> dict[str, Any]:
    """Use the repository's raw-logit MCQA evaluator against the exact GGUF."""
    reports = {}
    for task in ("arc_easy", "medmcqa", "openbookqa"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/mcq_eval.py"), "--model", str(model), "--task", task, "--limit", str(limit), "--n-ctx", "2048"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        reports[task] = {"command": result.args, "stdout": line}
    return {"limit": limit, "tasks": reports}


def repetition_eos(llm: Any) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Give a concise explanation of pneumonia danger signs in children."}]
    outputs = []
    for _ in range(3):
        result = llm.create_chat_completion(messages=messages, max_tokens=96, temperature=0.0, seed=42)
        choice = result["choices"][0]
        text = str(choice.get("message", {}).get("content") or "")
        sentences = [normalize(part) for part in re.split(r"[.!?]+", text) if normalize(part)]
        outputs.append({"text": text, "finish_reason": choice.get("finish_reason"), "repeated_sentence": len(sentences) != len(set(sentences))})
    hashes = [hashlib.sha256(normalize(item["text"]).encode("utf-8")).hexdigest() for item in outputs]
    return {"repetitions": outputs, "deterministic_normalized_output": len(set(hashes)) == 1, "normalized_hashes": hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mcqa-limit", type=int, default=500)
    args = parser.parse_args(argv)
    model = args.model.resolve()
    out = args.out_dir.resolve()
    if model.name != "Falcon-H1-1.5B-Deep-JamiiAfya-Q4_K_M.gguf":
        raise ValueError("final validation requires the exact Falcon Q4_K_M submission GGUF")
    if not model.is_file():
        raise FileNotFoundError(model)
    out.mkdir(parents=True, exist_ok=True)

    from llama_cpp import Llama

    llm = Llama(model_path=str(model), n_ctx=2048, n_gpu_layers=0, logits_all=True, verbose=False)
    frozen = prompts_from(ROOT / "docs/research/falcon_probe_heldout.json")
    baseline = prompts_from(ROOT / "docs/research/falcon_baseline_prompts.json")
    swahili = prompts_from(ROOT / "data/swahili_eval_set.json")
    general = [prompt for prompt in baseline if str(prompt.get("section", "")).startswith("C reasoning")]
    result = {
        "schema": "falcon-final-gguf-validation-v1",
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "model_sha256": sha256_file(model),
        "system_prompt": SYSTEM_PROMPT,
        "frozen_clinical_safety_with_system": run_battery(llm, frozen, out, system=SYSTEM_PROMPT, label="frozen_with_system", max_tokens=200),
        "frozen_clinical_safety_without_system": run_battery(llm, frozen, out, system=None, label="frozen_without_system", max_tokens=200),
        "english_deployment_battery": run_battery(llm, baseline, out, system=SYSTEM_PROMPT, label="english", max_tokens=256),
        "kiswahili_deployment_battery": run_battery(llm, swahili, out, system=SYSTEM_PROMPT, label="kiswahili", max_tokens=256),
        "general_reasoning_battery": run_battery(llm, general, out, system=SYSTEM_PROMPT, label="general_reasoning", max_tokens=120),
        "mcqa_likelihood_proxy": mcqa_proxy(model, out, args.mcqa_limit),
        "repetition_eos": repetition_eos(llm),
    }
    (out / "final_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "model_sha256": result["model_sha256"],
        "model_bytes": result["model_bytes"],
        "frozen_with_system": result["frozen_clinical_safety_with_system"]["pass_rate_percent"],
        "frozen_without_system": result["frozen_clinical_safety_without_system"]["pass_rate_percent"],
        "mcqa_tasks": list(result["mcqa_likelihood_proxy"]["tasks"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
