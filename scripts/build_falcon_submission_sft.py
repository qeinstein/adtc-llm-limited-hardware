#!/usr/bin/env python3
"""Build the fixed, audited Falcon-H1 submission SFT corpus.

The output is deliberately an ordinary assistant-completion corpus.  MCQA rows
are converted to one user turn plus one concise correct answer; no listwise
loss, answer-choice forwards, chain-of-thought, validation rows, or test rows
are admitted here.

The builder uses the pinned Falcon tokenizer to measure assistant-token mass.
That measurement drives the additive Kiswahili oversampling and is recorded in
the sidecar manifest.  The trainer later packs these rows into 512-token
sequences and samples the named mixture buckets by the same supervised-token
unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_falcon_data import infer_language, quality_flags, row_text
from scripts.build_falcon_dataset import holdout_texts, near_holdout, near_holdout_index
from scripts.falcon_format import render_completion


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

TARGET_BUCKETS = (
    "clinical_safety",
    "normal_clinical",
    "general_conversational",
    "mcqa_sft",
    "kiswahili_boost",
)
PUBLIC_TRAIN_SPLITS = {"train", "auxiliary_train"}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        values = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            values.append(value)
        return values
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: expected a JSON array of objects")
    return value


def _is_train_only_split(value: Any) -> bool:
    return str(value or "").strip().casefold() in PUBLIC_TRAIN_SPLITS


def verify_public_mcqa_manifest(path: Path) -> dict[str, Any]:
    """Reject MCQA material unless its repository manifest proves train-only provenance."""
    manifest_path = path.with_name(path.stem + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"MCQA manifest is required for train-only provenance: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = str(manifest.get("contamination_policy", "")).casefold()
    if "train" not in policy or "test" not in policy or "validation" not in policy:
        raise ValueError("MCQA manifest does not declare a train-only/test-validation exclusion policy")
    for name, source in (manifest.get("source_provenance") or {}).items():
        split = str((source or {}).get("split", "")).casefold()
        if split not in PUBLIC_TRAIN_SPLITS:
            raise ValueError(f"MCQA source {name!r} has forbidden split provenance {split!r}")
    skipped = [item.get("name") for item in manifest.get("source_results", []) if item.get("status") != "ok"]
    if skipped:
        raise ValueError(f"MCQA manifest records skipped sources: {skipped}")
    return manifest


def _assert_row_does_not_name_eval_split(row: dict[str, Any], source: str, index: int) -> None:
    """Guard against accidentally passing a benchmark validation/test row."""
    for key in ("split", "_split", "dataset_split", "_dataset_split"):
        if key in row and not _is_train_only_split(row[key]):
            raise ValueError(f"{source}:{index}: forbidden non-training split {key}={row[key]!r}")
    if row.get("_synthetic") is True:
        raise ValueError(f"{source}:{index}: synthetic material is not part of the audited submission corpus")


def _language_for(row: dict[str, Any], prompt: str, answer: str, declared: str) -> str:
    forced = str(row.get("_language") or "").strip().casefold()
    if forced in {"en", "sw", "other"}:
        return forced
    inferred = infer_language(f"{prompt}\n{answer}", declared)
    return inferred if inferred in {"en", "sw", "en+sw", "other"} else "en"


def is_kiswahili(language: str, prompt: str = "", answer: str = "") -> bool:
    """Return true for rows that contribute Kiswahili assistant-token exposure."""
    value = str(language).casefold()
    if value == "sw" or value.startswith("sw+") or value.endswith("+sw"):
        return True
    # Existing audited rows without a language tag are classified using the
    # repository's conservative lexical detector.  A bilingual row is useful
    # to the Kiswahili target even when it contains an English clinical term.
    return infer_language(f"{prompt}\n{answer}", "auto") in {"sw", "en+sw"}


def mcqa_answer(row: dict[str, Any]) -> str:
    """Convert one choice-list record into a concise supervised answer."""
    choices = row.get("choices")
    gold = row.get("gold")
    if not isinstance(choices, list) or not choices:
        raise ValueError("MCQA row has no choices")
    if not isinstance(gold, int) or not 0 <= gold < len(choices):
        raise ValueError("MCQA row has an invalid gold index")
    answer = str(choices[gold]).strip()
    if not answer:
        raise ValueError("MCQA gold choice is empty")
    fmt = str(row.get("format") or "").casefold()
    if fmt == "letter":
        letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[gold]
        return f"{letter}. {answer}"
    return answer


def _sft_record(
    *,
    source: str,
    index: int,
    instruction: str,
    answer: str,
    category: str,
    language: str,
    system_prompt: str | None,
    rendering: str,
    source_row: dict[str, Any],
) -> dict[str, Any]:
    if not instruction.strip() or not answer.strip():
        raise ValueError(f"{source}:{index}: empty instruction or answer")
    identity = "\0".join((source, str(index), rendering, instruction.strip(), answer.strip()))
    return {
        "example_id": digest(identity)[:24],
        "format": "sft",
        "source": source,
        "source_index": index,
        "category": category,
        "mix_bucket": category,
        "language": language,
        "rendering": rendering,
        "system_prompt": system_prompt,
        "instruction": instruction.strip(),
        "input": str(source_row.get("input") or "").strip(),
        "output": answer.strip(),
        "assistant_only_loss": True,
        "chain_of_thought": False,
    }


def _normal_user(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction") or "").strip()
    extra = str(row.get("input") or "").strip()
    return f"{instruction}\n\n{extra}" if extra else instruction


def _load_holdout_index(config: dict[str, Any]) -> dict[str, Any]:
    values: list[str] = []
    for relative in config["data"].get("frozen_holdouts", []):
        path = ROOT / str(relative)
        if path.is_file():
            values.extend(holdout_texts(path))
    return near_holdout_index(values)


def _assert_no_holdout(prompt: str, holdouts: dict[str, Any], source: str, index: int) -> None:
    match = near_holdout(prompt, holdouts)
    if match:
        raise ValueError(f"{source}:{index}: {match} match against frozen holdout")


def build_rows(config: dict[str, Any], tokenizer: Any, mcqa_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build base rows plus the required additive Kiswahili boost."""
    holdouts = _load_holdout_index(config)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for spec in config["data"]["sources"]:
        path = ROOT / str(spec["path"])
        if spec["name"] == "mcqa_sft":
            path = mcqa_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"required audited training source missing: {path}")
        if spec["name"] == "mcqa_sft":
            verify_public_mcqa_manifest(path)
        for index, raw in enumerate(read_records(path)):
            try:
                _assert_row_does_not_name_eval_split(raw, str(spec["name"]), index)
                if spec["format"] == "alpaca":
                    prompt, answer = row_text(raw, "alpaca")
                    if any(flag in set(spec.get("exclude_quality_flags", [])) for flag in quality_flags(prompt, answer, "alpaca")):
                        raise ValueError("quality policy excluded row")
                    language = _language_for(raw, prompt, answer, str(spec.get("language", "auto")))
                    _assert_no_holdout(prompt, holdouts, str(spec["name"]), index)
                    if spec.get("category") == "clinical_safety":
                        # Both renderings carry the exact same assistant answer;
                        # only the presence of the compact system turn differs.
                        rows.append(_sft_record(
                            source=str(spec["name"]), index=index, instruction=prompt,
                            answer=answer, category="clinical_safety", language=language,
                            system_prompt=SYSTEM_PROMPT, rendering="with_system",
                            source_row=raw,
                        ))
                        rows.append(_sft_record(
                            source=str(spec["name"]), index=index, instruction=prompt,
                            answer=answer, category="clinical_safety", language=language,
                            system_prompt=None, rendering="without_system",
                            source_row=raw,
                        ))
                    else:
                        rows.append(_sft_record(
                            source=str(spec["name"]), index=index, instruction=prompt,
                            answer=answer, category=str(spec["category"]), language=language,
                            system_prompt=SYSTEM_PROMPT, rendering="with_system",
                            source_row=raw,
                        ))
                elif spec["format"] == "mcqa":
                    context = str(raw.get("context") or "").strip()
                    answer = mcqa_answer(raw)
                    if not context or not answer:
                        raise ValueError("invalid MCQA context or answer")
                    # The MCQA builder's context is already the question plus
                    # options.  No system prompt keeps this ordinary SFT row
                    # close to the public train-only scoring shape.
                    _assert_no_holdout(context, holdouts, str(spec["name"]), index)
                    rows.append(_sft_record(
                        source=str(spec["name"]), index=index, instruction=context,
                        answer=answer, category="mcqa_sft", language="en",
                        system_prompt=None, rendering="without_system", source_row=raw,
                    ))
                else:
                    raise ValueError(f"unsupported source format {spec['format']!r}")
            except (KeyError, TypeError, ValueError) as exc:
                if spec.get("category") == "clinical_safety":
                    raise ValueError(
                        f"clinical safety source row {spec['name']}:{index} was rejected; "
                        "every audited safety row must have both renderings"
                    ) from exc
                rejected.append({"source": spec["name"], "index": index, "reason": str(exc)})

    if not rows:
        raise ValueError("no submission SFT rows were accepted")

    # Count exact assistant targets, including the template's assistant-turn
    # terminator, before adding the language boost.  This is the only quantity
    # used for the 10% Kiswahili floor.
    for row in rows:
        messages = []
        if row["system_prompt"]:
            messages.append({"role": "system", "content": row["system_prompt"]})
        messages.append({"role": "user", "content": row["instruction"]})
        _, target = render_completion(tokenizer, messages, row["output"])
        row["assistant_tokens"] = len(target)
        row["kiswahili"] = is_kiswahili(row["language"], row["instruction"], row["output"])

    base_total = sum(int(row["assistant_tokens"]) for row in rows)
    base_sw = sum(int(row["assistant_tokens"]) for row in rows if row["kiswahili"])
    threshold = float(config["data"]["mixture"]["minimum_total_kiswahili_assistant_token_share"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"invalid Kiswahili threshold {threshold}")
    # Also make the additive bucket approximately 5% of supervised mass.  If
    # the natural language share is below 10%, the floor wins and extra rows
    # are intentionally added; this is the requested additive targeting.
    nominal_boost = math.ceil((0.05 * base_total) / 0.95)
    floor_boost = max(0, math.ceil((threshold * base_total - base_sw) / (1.0 - threshold)))
    boost_target = max(nominal_boost, floor_boost)
    candidates = sorted(
        (row for row in rows if row["kiswahili"]),
        key=lambda row: (str(row["source"]), int(row["source_index"]), str(row["rendering"]), str(row["example_id"])),
    )
    if boost_target and not candidates:
        raise ValueError("cannot meet Kiswahili supervised-token floor: no audited Kiswahili training rows")
    boosted: list[dict[str, Any]] = []
    boost_tokens = 0
    boost_index = 0
    while boost_tokens < boost_target:
        source_row = candidates[boost_index % len(candidates)]
        copy = dict(source_row)
        copy["example_id"] = digest(f"{source_row['example_id']}\0kiswahili_boost\0{boost_index}")[:24]
        copy["mix_bucket"] = "kiswahili_boost"
        copy["rendering"] = f"{source_row['rendering']};kiswahili_boost"
        copy["oversampled_from"] = source_row["example_id"]
        boosted.append(copy)
        boost_tokens += int(copy["assistant_tokens"])
        boost_index += 1
    rows.extend(boosted)

    total_tokens = sum(int(row["assistant_tokens"]) for row in rows)
    sw_tokens = sum(int(row["assistant_tokens"]) for row in rows if row["kiswahili"])
    if sw_tokens / max(1, total_tokens) < threshold:
        raise AssertionError("internal Kiswahili oversampling calculation failed")
    buckets = Counter(str(row["mix_bucket"]) for row in rows)
    bucket_tokens = Counter()
    for row in rows:
        bucket_tokens[str(row["mix_bucket"])] += int(row["assistant_tokens"])
    manifest = {
        "schema": "falcon-submission-sft-v1",
        "system_prompt": SYSTEM_PROMPT,
        "base_rows": len(rows) - len(boosted),
        "boost_rows": len(boosted),
        "rows": len(rows),
        "rejected_rows": rejected,
        "row_counts_by_mix_bucket": dict(sorted(buckets.items())),
        "assistant_tokens_by_mix_bucket": dict(sorted(bucket_tokens.items())),
        "assistant_tokens_total": total_tokens,
        "kiswahili_assistant_tokens": sw_tokens,
        "kiswahili_assistant_token_share": sw_tokens / max(1, total_tokens),
        "minimum_kiswahili_assistant_token_share": threshold,
        "additive_boost_target_tokens": boost_target,
        "additive_boost_tokens": boost_tokens,
        "target_unit": "assistant_supervised_tokens",
        "source_files": {
            str(spec["name"]): str((mcqa_path if spec["name"] == "mcqa_sft" else ROOT / spec["path"]).resolve())
            for spec in config["data"]["sources"]
        },
        "source_sha256": {
            str(spec["name"]): digest((mcqa_path if spec["name"] == "mcqa_sft" else ROOT / spec["path"]).read_text(encoding="utf-8"))
            for spec in config["data"]["sources"]
        },
        "safety_rendering_policy": "two_renderings_per_clinical_safety_row_same_answer_with_and_without_system",
        "mcqa_policy": "ordinary_sft_concise_correct_choice_no_chain_of_thought",
    }
    return rows, manifest


def write_output(path: Path, rows: Iterable[dict[str, Any]], manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (str(row["mix_bucket"]), str(row["example_id"])))
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = dict(manifest)
    manifest["output"] = str(path)
    manifest["output_sha256"] = digest(path.read_text(encoding="utf-8"))
    path.with_name(path.stem + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/falcon-production-v1.json"))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--mcqa", default=None, help="Existing public train-only MCQA JSONL; defaults to config data source")
    parser.add_argument("--out", default=str(ROOT / "output/falcon-submission-sft-v1.jsonl"))
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mcqa_path = Path(args.mcqa or ROOT / "output/accuracy_sft.jsonl").resolve()
    tokenizer_name = args.tokenizer or config["model"]["id"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=config["model"].get("tokenizer_revision", config["model"]["revision"]),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows, manifest = build_rows(config, tokenizer, mcqa_path)
    write_output(Path(args.out).resolve(), rows, manifest)
    print(json.dumps({key: manifest[key] for key in (
        "rows", "base_rows", "boost_rows", "row_counts_by_mix_bucket",
        "assistant_tokens_by_mix_bucket", "kiswahili_assistant_token_share",
        "rejected_rows",
    )}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
