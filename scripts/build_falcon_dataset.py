#!/usr/bin/env python3
"""Build and audit the reproducible Falcon production train/dev dataset.

The builder is intentionally fail-closed. It never downloads evaluation data,
never turns a held-out prompt into training data, rejects over-length targets,
and writes token/facet counts so row counts cannot hide objective imbalance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_falcon_data import infer_category, infer_language
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def canonical(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: every JSONL row must be an object")
                rows.append(value)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON array")
    return [row for row in value if isinstance(row, dict)]


def language(text: str, forced: str) -> str:
    return infer_language(text, forced)


def holdout_texts(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    if isinstance(value, dict):
        for prompt in value.get("prompts", []):
            if isinstance(prompt, dict):
                out.append(str(prompt.get("text", "")))
    elif isinstance(value, list):
        for row in value:
            if isinstance(row, dict):
                out.append(str(row.get("query") or row.get("text") or row.get("instruction") or ""))
    return [canonical(x) for x in out if canonical(x)]


def near_holdout_index(holdouts: list[str]) -> dict[str, Any]:
    """Build a reusable index for the conservative holdout similarity gate.

    The old implementation rebuilt a set and compared every record with every
    holdout.  That made a 15k-row MCQA build spend minutes in quadratic
    SequenceMatcher work even though the holdout set is tiny.  Length is a
    lossless first filter for the ratio threshold: if the strings' lengths
    cannot achieve the threshold even with a perfect shorter-string match,
    SequenceMatcher cannot possibly accept them.
    """
    values = sorted({value for value in holdouts if value})
    return {"exact": frozenset(values), "by_length": [(len(value), value) for value in values]}


def near_holdout(text: str, holdouts: list[str] | dict[str, Any]) -> str | None:
    value = canonical(text)
    if not value:
        return None
    if isinstance(holdouts, dict):
        exact = holdouts["exact"]
        by_length = holdouts["by_length"]
    else:
        exact = frozenset(holdouts)
        by_length = [(len(other), other) for other in holdouts]
    if value in exact:
        return "exact"
    # The final holdouts are small; a high ratio is a useful conservative gate
    # against trivial paraphrase leakage, not a semantic similarity claim.
    value_len = len(value)
    for other_len, other in by_length:
        if min(value_len, other_len) < 35:
            continue
        # 2*min/(a+b) is the maximum possible SequenceMatcher ratio.  This
        # filter cannot discard a candidate that could reach 0.94.
        if 2.0 * min(value_len, other_len) < 0.94 * (value_len + other_len):
            continue
        if SequenceMatcher(None, value, other).ratio() >= 0.94:
            return "near"
    return None


def chat_prompt(tokenizer: Any, row: dict[str, Any], system: str) -> tuple[list[int], list[int]]:
    instruction = str(row.get("instruction") or "").strip()
    answer = str(row.get("output") or "").strip()
    if not instruction or not answer:
        raise ValueError("empty instruction/output")
    user = instruction
    extra = str(row.get("input") or "").strip()
    if extra:
        user = f"{instruction}\n\n{extra}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    target = tokenizer(answer, add_special_tokens=False)["input_ids"]
    target.append(int(tokenizer.eos_token_id))
    return list(prompt), target


def normalize_row(spec: dict[str, Any], row: dict[str, Any], tokenizer: Any, max_len: int, system: str) -> dict[str, Any]:
    fmt = spec["format"]
    base = {
        "source": str(row.get("_source") or spec["name"]),
        "category": infer_category(
            str(row.get("instruction") or row.get("context") or row.get("text") or "") + " " + str(row.get("output") or ""),
            str(row.get("_category") or spec["category"]),
        ),
        "provenance": str(row.get("_provenance") or spec["provenance"]),
        "license": str(row.get("_license") or spec["license"]),
        "language": language(
            str(row.get("instruction") or row.get("context") or row.get("text") or "") + " " + str(row.get("output") or ""),
            str(row.get("_language") or spec.get("language", "auto")),
        ),
        "objective": spec["objective"],
    }
    if fmt == "alpaca":
        prompt_ids, target_ids = chat_prompt(tokenizer, row, system)
        if len(target_ids) > max_len:
            raise ValueError(f"answer has {len(target_ids)} tokens > max_len={max_len}")
        prompt_ids = prompt_ids[-max(0, max_len - len(target_ids)):]
        if len(prompt_ids) + len(target_ids) > max_len:
            raise ValueError("prompt/answer packing exceeded max_len")
        content = str(row.get("instruction") or "").strip()
        answer = str(row.get("output") or "").strip()
        base.update({
            "format": "sft",
            "instruction": content,
            "input": str(row.get("input") or "").strip(),
            "output": answer,
            "prompt_tokens": len(prompt_ids),
            "target_tokens": len(target_ids),
            "loss_tokens": len(target_ids),
            "total_tokens": len(prompt_ids) + len(target_ids),
        })
        identity = content + "\n" + base["input"] + "\n" + answer
    elif fmt == "mcqa":
        context = str(row.get("context") or "").strip()
        choices = row.get("choices")
        gold = row.get("gold")
        if not context or not isinstance(choices, list) or not choices:
            raise ValueError("invalid MCQA context/choices")
        if not isinstance(gold, int) or not 0 <= gold < len(choices):
            raise ValueError("invalid MCQA gold index")
        ctx_ids = list(tokenizer(context, add_special_tokens=False)["input_ids"])
        choice_ids = [list(tokenizer(" " + str(choice), add_special_tokens=False)["input_ids"]) for choice in choices]
        if any(len(ctx_ids) + len(choice) > max_len for choice in choice_ids):
            raise ValueError("MCQA context plus a choice exceeds max_len; refusing to truncate answer choices")
        base.update({
            "format": "mcqa",
            "context": context,
            "choices": [str(x) for x in choices],
            "gold": gold,
            "prompt_tokens": len(ctx_ids) * len(choice_ids),
            "target_tokens": sum(len(x) for x in choice_ids),
            "loss_tokens": sum(len(x) for x in choice_ids),
            "total_tokens": len(ctx_ids) * len(choice_ids) + sum(len(x) for x in choice_ids),
        })
        identity = context + "\n" + "\n".join(map(str, choices)) + f"\n{gold}"
    else:
        raise ValueError(f"unsupported format: {fmt}")
    base["example_id"] = digest(spec["name"] + "\0" + identity)[:24]
    base["identity"] = canonical(identity)
    return base


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/falcon-production-v1.json"))
    ap.add_argument("--tokenizer", default=None, help="Override tokenizer id/revision for local tests")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--allow-missing-optional", action="store_true", default=True)
    args = ap.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir or ROOT / "experiments" / config["experiment_id"] / "data")
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    model = config["model"]["id"]
    revision = config["model"].get("tokenizer_revision")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or model, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_len = int(config["data"]["max_length"])
    holdouts: list[str] = []
    for rel in config["data"]["final_holdouts"]:
        path = ROOT / rel
        if path.exists():
            holdouts.extend(holdout_texts(path))
    holdout_index = near_holdout_index(holdouts)
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    rejected: list[dict[str, str]] = []
    missing: list[dict[str, Any]] = []
    for spec in config["data"]["sources"]:
        path = ROOT / spec["path"]
        if not path.exists():
            missing.append({"name": spec["name"], "path": spec["path"], "required": spec.get("required", True)})
            continue
        for index, raw in enumerate(read_records(path)):
            try:
                record = normalize_row(spec, raw, tokenizer, max_len, config["data"]["system_prompt"])
                text_for_holdout = str(record.get("instruction") or record.get("context") or "")
                leak = near_holdout(text_for_holdout, holdout_index)
                if leak:
                    raise ValueError(f"{leak} match against frozen final holdout")
                record["source_index"] = index
                rows.append(record)
                source_counts[record["source"]] += 1
            except (TypeError, ValueError, KeyError) as exc:
                rejected.append({"source": spec["name"], "index": str(index), "reason": str(exc)})
            if (index + 1) % 1000 == 0:
                print(
                    f"processed source={spec['name']} rows={index + 1} accepted={len(rows)} rejected={len(rejected)}",
                    flush=True,
                )

    # Global exact deduplication is performed before splitting so a duplicate
    # cannot land in train and dev under different source names.
    unique: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, str]] = []
    for row in rows:
        if row["identity"] in unique:
            duplicates.append({"kept": unique[row["identity"]]["example_id"], "dropped": row["example_id"]})
        else:
            unique[row["identity"]] = row
    rows = list(unique.values())
    dev_fraction = float(config["data"]["dev_fraction"])
    seed = str(config["data"]["split_seed"])
    for row in rows:
        bucket = int(digest(seed + "\0" + row["example_id"])[:8], 16) / 0xFFFFFFFF
        row["split"] = "dev" if bucket < dev_fraction else "train"
    if not any(row["split"] == "dev" for row in rows) and rows:
        rows[0]["split"] = "dev"
    for split in ("train", "dev"):
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in sorted((x for x in rows if x["split"] == split), key=lambda x: x["example_id"]):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    facets: dict[str, Counter[str]] = {key: Counter() for key in ("objective", "source", "category", "language", "split", "provenance")}
    tokens = Counter()
    loss_tokens = Counter()
    for row in rows:
        for key, counter in facets.items():
            counter[row[key]] += 1
        tokens[row["objective"]] += int(row["total_tokens"])
        loss_tokens[row["objective"]] += int(row["loss_tokens"])
    source_files = []
    for spec in config["data"]["sources"]:
        path = ROOT / spec["path"]
        entry = {"name": spec["name"], "path": spec["path"], "required": spec.get("required", True), "sha256": digest(path.read_text(encoding="utf-8")) if path.exists() else None}
        companion = path.with_name(path.stem + ".manifest.json")
        if companion.exists():
            entry["manifest_path"] = str(companion.relative_to(ROOT))
            entry["manifest_sha256"] = digest(companion.read_text(encoding="utf-8"))
        source_files.append(entry)
    manifest = {
        "schema_version": "1.0.0",
        "experiment_id": config["experiment_id"],
        "config_sha256": digest(config_path.read_text(encoding="utf-8")),
        "model": config["model"],
        "tokenizer": {"id": args.tokenizer or model, "revision": revision},
        "max_length": max_len,
        "counts": {"accepted": len(rows), "train": sum(x["split"] == "train" for x in rows), "dev": sum(x["split"] == "dev" for x in rows), "rejected": len(rejected), "exact_duplicates_dropped": len(duplicates)},
        "token_totals": dict(tokens),
        "token_shares_percent": {key: round(100 * value / max(1, sum(tokens.values())), 4) for key, value in tokens.items()},
        "loss_token_totals": dict(loss_tokens),
        "loss_token_shares_percent": {key: round(100 * value / max(1, sum(loss_tokens.values())), 4) for key, value in loss_tokens.items()},
        "facets": {key: dict(sorted(value.items())) for key, value in facets.items()},
        "missing_sources": missing,
        "rejected_examples": rejected,
        "duplicate_examples": duplicates[:100],
        "final_holdout_count": len(holdouts),
        "final_holdout_sha256": digest("\n".join(sorted(holdouts))),
        "source_files": source_files,
    }
    (out_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "counts": manifest["counts"], "token_totals": manifest["token_totals"], "facets": manifest["facets"], "missing_sources": missing}, indent=2))
    required_missing = [x for x in missing if x["required"]]
    return 2 if required_missing or rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
