import json
import sys
from pathlib import Path

from scripts.benchmark_falcon_deployment import bench_once, parse_time_report
from scripts.audit_falcon_data import near_duplicate_prompt_pairs, quality_flags
from scripts.build_falcon_dataset import read_records
from scripts.falcon_trainability_probe import tiny_rows
from scripts.score_falcon_battery import rule_result


def test_deployment_time_report_parser(tmp_path: Path):
    report = tmp_path / "time.log"
    report.write_text(
        "Maximum resident set size (kbytes): 123456\n"
        "Minor (reclaiming a frame) page faults: 789\n"
        "Major (page faults): 3\n"
        "Elapsed (wall clock) time (seconds): 1:02.50\n"
    )
    assert parse_time_report(report) == {
        "max_rss_kb": 123456,
        "minor_page_faults": 789,
        "major_page_faults": 3,
        "elapsed": "1:02.50",
    }


def test_deployment_wrapper_survives_dead_pid_race(tmp_path: Path):
    payload = json.dumps([
        {"n_gen": 0, "n_prompt": 512, "avg_ts": 100.0},
        {"n_gen": 128, "n_prompt": 0, "avg_ts": 20.0},
    ])
    result = bench_once(
        [sys.executable, "-c", f"import time; print({payload!r}); time.sleep(0.02)"],
        tmp_path / "time.log",
        0.005,
    )
    assert result["decode_tps"] == 20.0
    assert result["peak_tree_rss_mb_sampled"] >= 0


def test_trainability_fixture_contains_sft_and_mcqa():
    rows = tiny_rows()
    assert len(rows) == 32
    assert {row["format"] for row in rows} == {"sft", "mcqa"}
    assert sum(row["format"] == "sft" for row in rows) == 16
    assert sum(row["format"] == "mcqa" for row in rows) == 16
    assert len({row["example_id"] for row in rows}) == len(rows)


def test_prompt_search_sets_are_not_frozen_batteries():
    final_ids = {item["id"] for item in json.loads(Path("docs/research/falcon_probe_heldout.json").read_text())["prompts"]}
    dev_ids = {item["id"] for item in json.loads(Path("docs/research/falcon_prompt_dev.json").read_text())["prompts"]}
    validation_ids = {item["id"] for item in json.loads(Path("docs/research/falcon_prompt_validation.json").read_text())["prompts"]}
    assert not final_ids & (dev_ids | validation_ids)
    assert dev_ids.isdisjoint(validation_ids)


def test_prompt_weight_matrix_is_bounded_and_selection_gated():
    notebook = json.loads(Path("kaggle/phase04-falcon-prompt-weight-matrix/phase04_falcon_prompt_weight_matrix.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "falcon_trainability_probe.py" in source
    assert "falcon_probe_heldout.json" in source
    assert "No validation-selected prompt" in source
    assert "persist_checkpoint.py" not in source


def test_production_notebook_can_pin_selected_system_prompt():
    notebook = json.loads(Path("kaggle/phase04-falcon-production/phase04_falcon_production.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "FALCON_SYSTEM_PROMPT_FILE" in source
    assert "system_prompt_id" in source
    assert "falcon-selected-prompt-config.json" in source


def test_hallucinated_official_criteria_are_a_hard_veto():
    rubric = json.loads(Path("docs/research/falcon_generation_rubric.json").read_text())
    text = (
        "Grade-Blue is not part of WHO terminology, but the WHO 2021 criteria "
        "for diagnosing mild pneumonia are as follows."
    )
    result = rule_result("h10", text, rubric["rules"]["h10"], 12)
    assert result["passed"] is False
    assert any(failure.startswith("forbidden:") for failure in result["failures"])


def test_data_quality_flags_are_review_only_and_cover_known_risks():
    flags = quality_flags(
        "Multiple choice: what dose is recommended under WHO protocol?",
        "Give 10 mg; never swallow bleach.",
        "alpaca",
    )
    assert "mcqa_shaped_sft" in flags
    assert "authority_or_protocol_claim" in flags
    assert "numeric_medication_guidance" in flags
    assert "toxin_or_disinfectant_content" in flags


def test_near_duplicate_prompt_audit_is_bounded_and_excludes_exact_pairs():
    pairs = near_duplicate_prompt_pairs([
        ("a:1", "A child has severe chest indrawing and needs urgent referral today"),
        ("b:2", "A child has severe chest indrawing and needs urgent referral now"),
        ("c:3", "A child has severe chest indrawing and needs urgent referral today"),
    ])
    assert len(pairs) == 2
    assert {tuple(item[key] for key in ("left", "right")) for item in pairs} == {("a:1", "b:2"), ("b:2", "c:3")}


def test_production_policy_marks_mcqa_shaped_clinical_rows_for_exclusion():
    config = json.loads(Path("configs/falcon-production-v1.json").read_text())
    source = next(item for item in config["data"]["sources"] if item["name"] == "project_clinical_generation")
    rows = read_records(Path(source["path"]))
    assert source["exclude_quality_flags"] == ["mcqa_shaped_sft"]
    assert sum("Multiple choice" in row["instruction"] or "Chagua jibu" in row["instruction"] for row in rows) == 15
