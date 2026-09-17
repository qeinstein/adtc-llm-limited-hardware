"""Token-level audit of Jamii Afya training manifests.

The accounting intentionally mirrors scripts/train_lora.py. In particular, MCQA
contexts are counted once per answer choice because each choice is a separate
forward-pass row, while clinical rows receive the configured repeat multiplier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are Jamii Afya, an offline medical decision-support assistant for community "
    "health workers in rural African clinics. Answer in the question's language "
    "(English or Kiswahili). Always surface danger signs and when to refer."
)

SW_WORDS = {
    "na",
    "ya",
    "wa",
    "kwa",
    "ni",
    "katika",
    "hii",
    "hiyo",
    "nini",
    "jinsi",
    "mtoto",
    "afya",
    "mgonjwa",
    "dawa",
    "tafadhali",
    "asante",
    "habari",
    "ndiyo",
    "hapana",
    "kama",
    "lakini",
    "au",
    "ana",
    "nina",
    "una",
    "kwamba",
    "wakati",
    "maji",
    "chakula",
    "dalili",
    "matibabu",
    "hospitali",
    "haraka",
    "sana",
}
EN_WORDS = {
    "the",
    "and",
    "is",
    "of",
    "to",
    "in",
    "what",
    "how",
    "why",
    "when",
    "for",
    "with",
    "this",
    "that",
    "a",
    "an",
    "are",
    "should",
    "patient",
    "child",
    "health",
    "treatment",
    "question",
    "answer",
    "please",
    "can",
    "do",
    "does",
}
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class RowMetrics:
    source: str
    stream: str
    capability: str
    domain: str
    provenance: str
    language: str
    repeat: int
    prompt_original: int
    prompt_trained: int
    target_original: int
    target_trained: int
    total_trained: int
    answer_length: int
    truncated: bool
    over_max_len: bool
    tokens_lost: int
    canonical: str
    simhash: int


@dataclass
class Group:
    raw_rows: int = 0
    invalid_rows: int = 0
    rows: list[RowMetrics] = field(default_factory=list)


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return float(ordered[lo])
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def length_stats(values: list[int]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 2) if values else 0.0,
        "median": round(float(statistics.median(values)), 2) if values else 0.0,
        "p95": round(percentile(values, 0.95), 2),
    }


def normalize_text(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def simhash64(text: str) -> int:
    words = normalize_text(text).split()
    features = (
        words
        if len(words) < 3
        else [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    )
    if not features:
        return 0
    vector = [0] * 64
    for feature, count in Counter(features).items():
        value = int.from_bytes(
            hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            vector[bit] += count if value & (1 << bit) else -count
    return sum((1 << bit) for bit, score in enumerate(vector) if score >= 0)


def estimate_language(text: str, forced: str | None = None) -> str:
    if forced in {"en", "sw", "other"}:
        return forced
    words = normalize_text(text).split()
    sw = sum(word in SW_WORDS for word in words)
    en = sum(word in EN_WORDS for word in words)
    if sw >= 2 and sw > en:
        return "sw"
    if en >= 2 and en > sw:
        return "en"
    return "other_or_uncertain"


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Return parsed dict rows, raw nonblank records, and parse-invalid records."""
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        raw = invalid = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw += 1
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    invalid += 1
            except json.JSONDecodeError:
                invalid += 1
        return rows, raw, invalid
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path}: expected a JSON array")
    rows = [row for row in payload if isinstance(row, dict)]
    return rows, len(payload), len(payload) - len(rows)


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def _metadata(spec: dict[str, Any], row: dict[str, Any]) -> dict[str, str]:
    source = str(row.get("_source") or row.get("source") or spec["name"])
    override = spec.get("source_overrides", {}).get(source, {})
    return {
        "source": source,
        "stream": str(override.get("stream", spec["stream"])),
        "capability": str(
            override.get("capability", row.get("_category") or spec["capability"])
        ),
        "domain": str(override.get("domain", spec["domain"])),
        "provenance": str(
            override.get(
                "provenance",
                "synthetic" if row.get("_synthetic") else spec["provenance"],
            )
        ),
        "forced_language": str(
            override.get("language", row.get("_language") or spec.get("language", ""))
        ),
    }


def _canonical(fmt: str, row: dict[str, Any]) -> str:
    if fmt == "mcqa":
        obj = [row.get("context"), row.get("choices"), row.get("gold")]
    elif fmt == "alpaca":
        obj = [row.get("instruction"), row.get("input"), row.get("output")]
    else:
        obj = [row.get("text")]
    return normalize_text(json.dumps(obj, ensure_ascii=False, sort_keys=True))


def measure_row(
    tokenizer: Any, spec: dict[str, Any], row: dict[str, Any], max_len: int, eos_id: int
) -> RowMetrics | None:
    fmt = spec["format"]
    meta = _metadata(spec, row)
    repeat = max(1, int(spec.get("repeat", 1)))
    prompt_original = prompt_trained = target_original = target_trained = lost = 0
    truncated = False

    if fmt == "mcqa":
        context = row.get("context")
        choices = row.get("choices")
        gold = row.get("gold")
        if not isinstance(context, str) or not isinstance(choices, list) or not choices:
            return None
        if (
            not isinstance(gold, int)
            or not 0 <= gold < len(choices)
            or not all(isinstance(x, str) for x in choices)
        ):
            return None
        ctx_ids = _encode(tokenizer, context)
        trained_choices: list[list[int]] = []
        original_choices: list[list[int]] = []
        for choice in choices:
            ids = _encode(tokenizer, " " + choice)
            original_choices.append(ids)
            trained = ids
            if len(ctx_ids) + len(ids) > max_len:
                trained = ids[: max(1, max_len - len(ctx_ids))]
                truncated = True
            trained_choices.append(trained)
        prompt_original = prompt_trained = len(ctx_ids) * len(choices)
        target_original = sum(map(len, original_choices))
        target_trained = sum(map(len, trained_choices))
        lost = target_original - target_trained
        answer_len = len(original_choices[gold])
        text = context + "\n" + "\n".join(choices)
    elif fmt == "alpaca":
        instruction = (row.get("instruction") or "").strip()
        answer = (row.get("output") or "").strip()
        if not instruction or not answer:
            return None
        inp = (row.get("input") or "").strip()
        user = f"{instruction}\n\n{inp}" if inp else instruction
        messages = [
            {"role": "system", "content": spec.get("system_prompt", SYSTEM_PROMPT)},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ctx_ids = _encode(tokenizer, prompt)
        tgt_ids = _encode(tokenizer, answer) + [eos_id]
        original_ctx_len = len(ctx_ids)
        if len(ctx_ids) + len(tgt_ids) > max_len:
            keep = max_len - len(tgt_ids)
            ctx_ids = ctx_ids[-keep:]
            truncated = True
        prompt_original, prompt_trained = original_ctx_len, len(ctx_ids)
        target_original = target_trained = len(tgt_ids)
        lost = max(0, prompt_original - prompt_trained)
        answer_len = target_original
        text = user + "\n" + answer
    elif fmt == "corpus":
        text = (row.get("text") or "").strip()
        if not text:
            return None
        original = _encode(tokenizer, text)
        if len(original) < 8:
            return None
        trained = original[:max_len]
        truncated = len(trained) != len(original)
        split = max(1, len(trained) // 4)
        prompt_original = min(split, len(original))
        prompt_trained = split
        target_original = len(original) - prompt_original
        target_trained = len(trained) - split
        lost = len(original) - len(trained)
        answer_len = target_original
    else:
        raise ValueError(f"unsupported format: {fmt}")

    canonical = _canonical(fmt, row)
    language = estimate_language(text, meta["forced_language"] or None)
    over_max_len = prompt_trained + target_trained > max_len
    metric = RowMetrics(
        source=meta["source"],
        stream=meta["stream"],
        capability=meta["capability"],
        domain=meta["domain"],
        provenance=meta["provenance"],
        language=language,
        repeat=repeat,
        prompt_original=prompt_original,
        prompt_trained=prompt_trained,
        target_original=target_original,
        target_trained=target_trained,
        total_trained=prompt_trained + target_trained,
        answer_length=answer_len,
        truncated=truncated,
        over_max_len=over_max_len,
        tokens_lost=lost,
        canonical=canonical,
        simhash=simhash64(canonical),
    )
    return metric


def near_duplicate_count(rows: list[RowMetrics], threshold: int = 4) -> int:
    unique: dict[str, RowMetrics] = {row.canonical: row for row in rows}
    buckets: dict[tuple[int, int], list[RowMetrics]] = defaultdict(list)
    near: set[str] = set()
    for row in unique.values():
        candidates: dict[str, RowMetrics] = {}
        for band in range(4):
            key = (band, (row.simhash >> (band * 16)) & 0xFFFF)
            for candidate in buckets[key]:
                candidates[candidate.canonical] = candidate
        for candidate in candidates.values():
            if (row.simhash ^ candidate.simhash).bit_count() <= threshold:
                near.add(row.canonical)
                near.add(candidate.canonical)
        for band in range(4):
            key = (band, (row.simhash >> (band * 16)) & 0xFFFF)
            buckets[key].append(row)
    return len(near)


def summarize_rows(
    rows: list[RowMetrics], raw_rows: int, invalid_rows: int
) -> dict[str, Any]:
    unique = len({row.canonical for row in rows})
    exact = len(rows) - unique
    effective_examples = sum(row.repeat for row in rows)
    prompt = sum(row.prompt_trained * row.repeat for row in rows)
    target = sum(row.target_trained * row.repeat for row in rows)
    original = sum(
        (row.prompt_original + row.target_original) * row.repeat for row in rows
    )
    trained = prompt + target
    trunc_rows = sum(row.repeat for row in rows if row.truncated)
    return {
        "raw_rows": raw_rows,
        "valid_rows": len(rows),
        "invalid_rows": invalid_rows,
        "unique_rows": unique,
        "exact_duplicates": exact,
        "duplicate_rate_percent": round(100 * exact / len(rows), 3) if rows else 0.0,
        "near_duplicate_unique_rows": near_duplicate_count(rows),
        "effective_examples": effective_examples,
        "prompt_tokens": prompt,
        "target_tokens": target,
        "total_non_padding_tokens": trained,
        "original_tokens_before_truncation": original,
        "tokens_lost_to_truncation": max(0, original - trained),
        "tokens_lost_to_truncation_percent": round(
            100 * max(0, original - trained) / original, 3
        )
        if original
        else 0.0,
        "truncated_effective_examples": trunc_rows,
        "truncation_rate_percent": round(100 * trunc_rows / effective_examples, 3)
        if effective_examples
        else 0.0,
        "over_max_len_effective_examples": sum(
            row.repeat for row in rows if row.over_max_len
        ),
        "prompt_length_tokens": length_stats([row.prompt_original for row in rows]),
        "answer_length_tokens": length_stats([row.answer_length for row in rows]),
    }


def _facet(rows: list[RowMetrics], attr: str) -> dict[str, Any]:
    grouped: dict[str, list[RowMetrics]] = defaultdict(list)
    for row in rows:
        grouped[str(getattr(row, attr))].append(row)
    totals = {
        name: sum(r.total_trained * r.repeat for r in values)
        for name, values in grouped.items()
    }
    denominator = sum(totals.values())
    return {
        name: {
            "effective_examples": sum(r.repeat for r in grouped[name]),
            "total_non_padding_tokens": tokens,
            "token_share_percent": round(100 * tokens / denominator, 3)
            if denominator
            else 0.0,
        }
        for name, tokens in sorted(totals.items())
    }


def audit_manifest(manifest_path: Path, tokenizer: Any) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.resolve().parents[
        int(manifest.get("repository_root_parents", 2))
    ]
    max_len = int(manifest["max_len"])
    eos_id = int(tokenizer.eos_token_id)
    groups: dict[str, Group] = defaultdict(Group)
    missing: list[dict[str, Any]] = []
    all_rows: list[RowMetrics] = []

    for spec in manifest["sources"]:
        path = root / spec["path"]
        if not path.exists():
            missing.append(
                {
                    "name": spec["name"],
                    "path": spec["path"],
                    "required": bool(spec.get("required", True)),
                }
            )
            continue
        parsed, _raw, parse_invalid = _read_rows(path)
        raw_by_source: Counter[str] = Counter()
        invalid_by_source: Counter[str] = Counter()
        for row in parsed:
            source = _metadata(spec, row)["source"]
            raw_by_source[source] += 1
            metric = measure_row(tokenizer, spec, row, max_len, eos_id)
            if metric is None:
                invalid_by_source[source] += 1
                continue
            groups[source].rows.append(metric)
            all_rows.append(metric)
        fallback = spec["name"]
        groups[fallback].invalid_rows += parse_invalid
        groups[fallback].raw_rows += parse_invalid
        for source, count in raw_by_source.items():
            groups[source].raw_rows += count
            groups[source].invalid_rows += invalid_by_source[source]

    per_source = {
        name: summarize_rows(group.rows, group.raw_rows, group.invalid_rows)
        for name, group in sorted(groups.items())
    }
    total = summarize_rows(
        all_rows,
        sum(g.raw_rows for g in groups.values()),
        sum(g.invalid_rows for g in groups.values()),
    )
    total_tokens = total["total_non_padding_tokens"]
    for summary in per_source.values():
        summary["token_share_percent"] = (
            round(100 * summary["total_non_padding_tokens"] / total_tokens, 3)
            if total_tokens
            else 0.0
        )

    return {
        "schema_version": "1.0.0",
        "audit_id": manifest["audit_id"],
        "manifest": str(manifest_path),
        "tokenizer": manifest["tokenizer"],
        "tokenizer_revision": manifest.get("tokenizer_revision"),
        "max_len": max_len,
        "accounting": {
            "token_share_basis": "effective non-padding forward-pass tokens after loader truncation and repeat multipliers",
            "mcqa_prompt_accounting": "context tokens counted once per candidate choice, matching train_lora.py",
            "answer_length_stat": "gold choice for MCQA; full target for chat; untruncated modeled suffix for corpus",
            "language_rule": "forced manifest/row label when present, otherwise documented English/Kiswahili stopword heuristic",
            "near_duplicate_rule": "64-bit word-trigram SimHash; Hamming distance <= 4 among exact-unique rows",
        },
        "missing_sources": missing,
        "total": total,
        "per_source": per_source,
        "facets": {
            "stream": _facet(all_rows, "stream"),
            "capability": _facet(all_rows, "capability"),
            "domain": _facet(all_rows, "domain"),
            "language": _facet(all_rows, "language"),
            "provenance": _facet(all_rows, "provenance"),
        },
    }


CSV_FIELDS = [
    "source",
    "raw_rows",
    "valid_rows",
    "unique_rows",
    "effective_examples",
    "prompt_tokens",
    "target_tokens",
    "total_non_padding_tokens",
    "token_share_percent",
    "duplicate_rate_percent",
    "near_duplicate_unique_rows",
    "truncation_rate_percent",
    "tokens_lost_to_truncation_percent",
]


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for source, values in report["per_source"].items():
            writer.writerow(
                {
                    "source": source,
                    **{key: values[key] for key in CSV_FIELDS if key != "source"},
                }
            )

    lines = [
        f"# Training-data token audit: {report['audit_id']}",
        "",
        f"Tokenizer: `{report['tokenizer']}` at revision `{report.get('tokenizer_revision')}`  ",
        f"Maximum sequence length: {report['max_len']}  ",
        "Token shares are effective non-padding forward-pass tokens after repeats and loader truncation.",
        "",
        "| Source | Unique examples | Effective examples | Prompt tokens | Target tokens | Total tokens | % total | Duplicate rate | Truncation rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, values in report["per_source"].items():
        lines.append(
            f"| {source} | {values['unique_rows']:,} | {values['effective_examples']:,} | "
            f"{values['prompt_tokens']:,} | {values['target_tokens']:,} | "
            f"{values['total_non_padding_tokens']:,} | {values['token_share_percent']:.3f}% | "
            f"{values['duplicate_rate_percent']:.3f}% | {values['truncation_rate_percent']:.3f}% |"
        )
    lines.extend(["", "## Aggregate facets", ""])
    for facet, values in report["facets"].items():
        lines.extend(
            [
                f"### {facet}",
                "",
                "| Bucket | Effective examples | Tokens | Token share |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, value in values.items():
            lines.append(
                f"| {name} | {value['effective_examples']:,} | {value['total_non_padding_tokens']:,} | {value['token_share_percent']:.3f}% |"
            )
        lines.append("")
    if report["missing_sources"]:
        lines.extend(["## Missing inputs", ""])
        for item in report["missing_sources"]:
            lines.append(
                f"- `{item['path']}` ({'required' if item['required'] else 'optional'}; source `{item['name']}`)"
            )
        lines.append("")
    lines.extend(
        [
            "## Method",
            "",
            *[f"- {key}: {value}" for key, value in report["accounting"].items()],
            "",
        ]
    )
    (output_dir / "audit.md").write_text("\n".join(lines), encoding="utf-8")
