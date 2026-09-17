#!/usr/bin/env python3
"""Apply the fixed post-training Falcon candidate-selection rule.

Only three trained adapters are eligible: Stage1-step96, Stage2-final, and
Stage3-final.  The frozen clinical/safety battery is consulted after training,
in deterministic Stage3 -> Stage2 -> Stage1 order.  No development score,
loss, arbitrary checkpoint, or hyperparameter is used to choose the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.select_falcon_candidate import frozen_gate_passes
from scripts.verify_falcon_promotion import build_promotion_manifest, verify_promoted_adapter


CANDIDATE_ORDER = ("Stage3-final", "Stage2-final", "Stage1-step96")
CANDIDATE_STEPS = {"Stage1-step96": 96, "Stage2-final": 24, "Stage3-final": 16}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidates(run_dir: Path) -> dict[str, Path]:
    trajectory_path = run_dir / "trajectory_manifest.json"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"training trajectory manifest missing: {trajectory_path}")
    manifest = json.loads(trajectory_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates") or {}
    expected = set(CANDIDATE_ORDER)
    if set(candidates) != expected:
        raise ValueError(f"trajectory candidates must be exactly {sorted(expected)}")
    result = {name: Path(str(candidates[name])).resolve() for name in CANDIDATE_ORDER}
    missing = [str(path) for path in result.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"trained candidate adapter(s) missing: {missing}")
    return result


def evaluate_candidate(
    *,
    label: str,
    adapter: Path,
    config: Path,
    battery: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Run the existing frozen battery and its deterministic scorer once."""
    empty_data = out_dir / "empty-dev-data"
    empty_data.mkdir(parents=True, exist_ok=True)
    (empty_data / "dev.jsonl").write_text("", encoding="utf-8")
    generation_dir = out_dir / label
    generation_dir.mkdir(parents=True, exist_ok=True)
    evaluate = [
        sys.executable, "-u", str(ROOT / "scripts/evaluate_falcon_hf.py"),
        "--config", str(config), "--data-dir", str(empty_data),
        "--output-dir", str(generation_dir), "--adapter", str(adapter),
        "--max-new-tokens", "200", "--battery", str(battery),
    ]
    subprocess.run(evaluate, cwd=ROOT, check=True)
    quality_path = generation_dir / "frozen-quality.json"
    score = [
        sys.executable, "-u", str(ROOT / "scripts/score_falcon_battery.py"),
        "--battery", str(battery), "--generation-dir",
        str(generation_dir / battery.stem), "--out", str(quality_path), "--report-only",
    ]
    subprocess.run(score, cwd=ROOT, check=True)
    report = json.loads(quality_path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "adapter": str(adapter),
        "report": str(quality_path),
        "passed": frozen_gate_passes(report),
        "critical_failures": list(report.get("critical_failures", [])),
        "pass_rate_percent": report.get("pass_rate_percent"),
        "report_summary": report,
    }


def select_frozen_candidate(
    run_dir: Path,
    config: Path,
    battery: Path,
    out_dir: Path,
) -> dict[str, Any]:
    candidates = load_candidates(run_dir)
    evaluations: list[dict[str, Any]] = []
    for label in CANDIDATE_ORDER:
        result = evaluate_candidate(
            label=label,
            adapter=candidates[label],
            config=config,
            battery=battery,
            out_dir=out_dir,
        )
        evaluations.append(result)
        # A passing Stage3 is final.  A failed candidate is never tuned or
        # replaced with an arbitrary nearby checkpoint; continue only through
        # the predetermined fallback order.
        if result["passed"]:
            selected = result
            break
    else:
        selected = None
    selection = {
        "schema": "falcon-submission-selection-v1",
        "criterion": "frozen_clinical_safety_gate_only",
        "candidate_order": list(CANDIDATE_ORDER),
        "candidates_considered": evaluations,
        "selected_candidate": selected["label"] if selected else "stock-falcon-h1",
        "selected_adapter": selected["adapter"] if selected else None,
        "stock_fallback": {
            "model": "tiiuae/Falcon-H1-1.5B-Deep-Instruct",
            "revision": "b6648636ddc906688974282de6e7a243395f5423",
        } if selected is None else None,
    }
    if selected is not None:
        adapter = Path(selected["adapter"]).resolve()
        frozen_report = selected["report_summary"]
        frozen_report["report_sha256"] = sha256_file(Path(selected["report"]))
        config_sha = sha256_file(config)
        data_manifest = ROOT / "output" / "falcon-submission-sft-v1.manifest.json"
        promotion = build_promotion_manifest(
            adapter,
            quality_selection={
                "status": "selected_and_frozen_gate_passed",
                "selected_step": CANDIDATE_STEPS[selected["label"]],
                "selected_eval_loss": None,
                "selection_criterion": "frozen_clinical_safety_gate_only",
                "minimum_pass_rate_percent": 100.0,
                "selected_dev_validation_pass_rate_percent": None,
                "frozen_gate": {"status": "passed"},
            },
            frozen_report=frozen_report,
            experiment_id="falcon-submission-sft-v1",
            stage=selected["label"],
            repo_sha=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip(),
            config_sha256=config_sha,
            data_manifest_sha256=sha256_file(data_manifest) if data_manifest.is_file() else None,
            minimum_pass_rate=100.0,
        )
        promotion_path = adapter / "promotion_manifest.json"
        promotion_path.write_text(json.dumps(promotion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        verify_promoted_adapter(adapter, minimum_pass_rate=100.0)
        selection["promotion_manifest"] = str(promotion_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", default=str(ROOT / "configs/falcon-production-v1.json"), type=Path)
    parser.add_argument("--battery", default=str(ROOT / "docs/research/falcon_probe_heldout.json"), type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = select_frozen_candidate(args.run_dir.resolve(), args.config.resolve(), args.battery.resolve(), args.out_dir.resolve())
    print(json.dumps({key: result[key] for key in ("selected_candidate", "selected_adapter", "criterion")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
