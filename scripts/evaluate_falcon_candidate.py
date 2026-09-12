#!/usr/bin/env python3
"""Evaluate the exact exported Falcon GGUF and preserve every raw response."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from scripts.mcq_eval import load_task, loglik


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", nargs="+", default=["medmcqa", "arc_easy"])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--battery", action="append", default=[])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    model = Path(args.model).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    events = out / "eval_events.jsonl"

    def emit(name: str, **fields):
        record = {"timestamp_utc": stamp(), "event": name, **fields}
        print(json.dumps(record, ensure_ascii=False), flush=True)
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    emit("evaluation_start", model=str(model), bytes=model.stat().st_size, sha256=file_sha(model), tasks=args.tasks, limit=args.limit)
    from llama_cpp import Llama

    llm = Llama(model_path=str(model), n_ctx=args.n_ctx, n_gpu_layers=0, n_threads=args.threads, logits_all=True, verbose=False)
    metrics = {"schema_version": "1.0.0", "model": str(model), "model_bytes": model.stat().st_size, "model_sha256": file_sha(model), "tasks": {}, "batteries": []}
    for task in args.tasks:
        emit("mcqa_start", task=task)
        items = load_task(task, args.limit, args.offset)
        acc = acc_norm = 0
        for index, (ctx, choices, gold) in enumerate(items):
            scores = []
            norms = []
            for choice in choices:
                score, _ = loglik(llm, ctx, choice)
                scores.append(score)
                norms.append(score / max(len(choice), 1))
            acc += max(range(len(scores)), key=scores.__getitem__) == gold
            acc_norm += max(range(len(norms)), key=norms.__getitem__) == gold
            if (index + 1) % 25 == 0 or index + 1 == len(items):
                emit("mcqa_progress", task=task, completed=index + 1, total=len(items))
        result = {"n": len(items), "acc": round(100 * acc / max(1, len(items)), 4), "acc_norm": round(100 * acc_norm / max(1, len(items)), 4), "offset": args.offset}
        metrics["tasks"][task] = result
        emit("mcqa_complete", task=task, **result)

    for battery_path in args.battery:
        battery_file = Path(battery_path).resolve()
        payload = json.loads(battery_file.read_text(encoding="utf-8"))
        prompts = payload.get("prompts", []) if isinstance(payload, dict) else payload
        name = battery_file.stem
        destination = out / name
        destination.mkdir(parents=True, exist_ok=True)
        index = []
        emit("generation_start", battery=str(battery_file), count=len(prompts))
        for prompt in prompts:
            pid = prompt["id"]
            result = llm.create_completion(prompt=prompt["text"], max_tokens=int(prompt.get("max_tokens", 256)), temperature=0.0, seed=args.seed)
            text = result["choices"][0]["text"]
            (destination / f"{pid}.txt").write_text(text, encoding="utf-8")
            index.append({"id": pid, "chars": len(text), "section": prompt.get("section", ""), "check": prompt.get("check", "")})
            emit("generation_item", battery=name, id=pid, chars=len(text))
        (destination / "_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        metrics["batteries"].append({"name": name, "count": len(index), "raw_dir": str(destination)})
        emit("generation_complete", battery=name, count=len(index))
    (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    emit("evaluation_complete", metrics=str(out / "eval_metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
