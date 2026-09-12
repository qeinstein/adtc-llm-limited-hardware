#!/usr/bin/env python3
"""Quantitative raw-source audit that does not require downloading model weights.

When ``--tokenizer`` is supplied, the report also includes exact prompt/target
token counts. Without it, word/character counts are explicitly labelled as
proxies; the production dataset builder remains the exact tokenizer gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def norm(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def infer_category(text: str, fallback: str) -> str:
    value = norm(text)
    rules = [
        ("toxicology_and_bleach", ("bleach", "kerosene", "poison", "miracle mineral")),
        ("invasive_procedure_safety", ("amput", "sutur", "stitch", "cesarean", "abscess", "pliers", "incision", "inject")),
        ("medication_uncertainty", ("dose", "dosage", "milligram", "tablet", "medicine", "antibiotic")),
        ("maternal_newborn", ("pregnan", "fetal", "baby moved", "kicked", "newborn", "vaginal bleeding")),
        ("pediatric_respiratory", ("pneumonia", "breathing", "indrawing", "wheez", "stridor")),
        ("dehydration_ors", ("dehydrat", "diarrhea", "diarrhoea", "ors", "sunken eyes")),
        ("false_premise_uncertainty", ("who", "protocol", "guarantee", "remotely", "true that", "claim")),
        ("general_reasoning", ("sensitivity", "correlation", "goat", "bottle and a cap", "birds")),
    ]
    for name, needles in rules:
        if any(needle in value for needle in needles):
            return name
    return fallback


def infer_language(text: str, forced: str) -> str:
    if forced in {"en", "sw", "other"}:
        return forced
    words = set(norm(text).split())
    sw = len(words & {"mtoto", "afya", "mgonjwa", "dawa", "hospitali", "haraka", "maji", "nimonia", "rufaa", "mjamzito", "kipindupindu"})
    en = len(words & {"the", "patient", "child", "health", "treatment", "hospital", "urgent", "water", "pregnant", "pneumonia"})
    if sw and en:
        return "en+sw"
    if sw:
        return "sw"
    return "en"


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def row_text(row: dict[str, Any], fmt: str) -> tuple[str, str]:
    if fmt == "alpaca":
        prompt = str(row.get("instruction") or "") + "\n" + str(row.get("input") or "")
        return prompt.strip(), str(row.get("output") or "").strip()
    if fmt == "mcqa":
        return str(row.get("context") or ""), "\n".join(map(str, row.get("choices") or []))
    return str(row.get("text") or ""), ""


def holdout_prompts(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    if isinstance(value, dict):
        out = [str(x.get("text") or "") for x in value.get("prompts", []) if isinstance(x, dict)]
    elif isinstance(value, list):
        out = [str(x.get("query") or x.get("text") or x.get("instruction") or "") for x in value if isinstance(x, dict)]
    return [norm(x) for x in out if norm(x)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/falcon-production-v1.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = Path(args.out or ROOT / "experiments" / config["experiment_id"] / "raw-data-audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=config["model"].get("tokenizer_revision"), trust_remote_code=True)

    holdouts: list[str] = []
    for rel in config["data"]["final_holdouts"]:
        path = ROOT / rel
        if path.exists():
            holdouts.extend(holdout_prompts(path))
    all_identity: dict[str, str] = {}
    duplicate_rows = []
    contamination = []
    per_source: dict[str, dict[str, Any]] = {}
    facets: dict[str, Counter[str]] = defaultdict(Counter)
    for spec in config["data"]["sources"]:
        path = ROOT / spec["path"]
        if not path.exists():
            per_source[spec["name"]] = {"status": "missing", "required": bool(spec.get("required", True)), "path": spec["path"]}
            continue
        rows = read(path)
        stats = {"status": "present", "path": spec["path"], "format": spec["format"], "objective": spec["objective"], "raw_rows": len(rows), "valid_rows": 0, "invalid_rows": 0, "characters": 0, "prompt_characters": 0, "target_characters": 0, "words_proxy": 0, "target_words_proxy": 0, "exact_token_counts": bool(tokenizer)}
        lengths = []
        for index, row in enumerate(rows):
            prompt, target = row_text(row, spec["format"])
            if not prompt or (spec["format"] == "alpaca" and not target):
                stats["invalid_rows"] += 1
                continue
            stats["valid_rows"] += 1
            identity = norm(prompt + "\n" + target)
            if identity in all_identity:
                duplicate_rows.append({"source": spec["name"], "index": index, "duplicate_of": all_identity[identity]})
            else:
                all_identity[identity] = f"{spec['name']}:{index}"
            if any(len(identity) > 35 and min(len(identity), len(h)) > 35 and SequenceMatcher(None, identity, h).ratio() >= 0.94 for h in holdouts):
                contamination.append({"source": spec["name"], "index": index, "type": "exact_or_near_prompt_match"})
            stats["prompt_characters"] += len(prompt)
            stats["target_characters"] += len(target)
            stats["characters"] += len(prompt) + len(target)
            stats["words_proxy"] += len(norm(prompt).split())
            stats["target_words_proxy"] += len(norm(target).split())
            lengths.append(len(target))
            facets["source"][spec["name"]] += 1
            facets["objective"][spec["objective"]] += 1
            facets["category"][infer_category(prompt + " " + target, str(row.get("_category") or spec["category"]))] += 1
            facets["language"][infer_language(prompt + " " + target, str(row.get("_language") or spec.get("language", "auto")))] += 1
        if lengths:
            stats["target_char_min"] = min(lengths)
            stats["target_char_mean"] = round(sum(lengths) / len(lengths), 2)
            stats["target_char_p95"] = sorted(lengths)[int(0.95 * (len(lengths) - 1))]
        if tokenizer:
            stats["prompt_tokens"] = 0
            stats["target_tokens"] = 0
            for row in rows:
                prompt, target = row_text(row, spec["format"])
                stats["prompt_tokens"] += len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                stats["target_tokens"] += len(tokenizer(target, add_special_tokens=False)["input_ids"])
        per_source[spec["name"]] = stats
    if tokenizer:
        total_tokens = sum(v.get("prompt_tokens", 0) + v.get("target_tokens", 0) for v in per_source.values())
        for value in per_source.values():
            value["token_share_percent"] = round(100 * (value.get("prompt_tokens", 0) + value.get("target_tokens", 0)) / max(1, total_tokens), 4)
    report = {
        "schema_version": "1.0.0",
        "config_sha256": sha(config_path.read_text(encoding="utf-8")),
        "tokenizer_mode": "exact" if tokenizer else "word/character_proxy",
        "final_holdout_count": len(holdouts),
        "per_source": per_source,
        "facets": {key: dict(value) for key, value in facets.items()},
        "duplicate_count": len(duplicate_rows),
        "duplicates": duplicate_rows[:200],
        "contamination_count": len(contamination),
        "contamination": contamination[:200],
        "notes": [
            "This audit reads only declared sources; optional missing files are not silently treated as present.",
            "Exact tokenizer counts require --tokenizer and are recomputed by build_falcon_dataset.py before training.",
            "Final holdouts are not used as training sources and are checked for exact/high-similarity prompt leakage.",
        ],
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if duplicate_rows or contamination else 0


if __name__ == "__main__":
    raise SystemExit(main())
