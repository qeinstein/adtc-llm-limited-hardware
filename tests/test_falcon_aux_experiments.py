import json
from pathlib import Path

from scripts.benchmark_falcon_deployment import parse_time_report
from scripts.falcon_trainability_probe import tiny_rows


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


def test_trainability_fixture_contains_sft_and_mcqa():
    rows = tiny_rows()
    assert len(rows) == 12
    assert {row["format"] for row in rows} == {"sft", "mcqa"}
    assert len({row["example_id"] for row in rows}) == len(rows)


def test_prompt_search_sets_are_not_frozen_batteries():
    final_ids = {item["id"] for item in json.loads(Path("docs/research/falcon_probe_heldout.json").read_text())["prompts"]}
    dev_ids = {item["id"] for item in json.loads(Path("docs/research/falcon_prompt_dev.json").read_text())["prompts"]}
    validation_ids = {item["id"] for item in json.loads(Path("docs/research/falcon_prompt_validation.json").read_text())["prompts"]}
    assert not final_ids & (dev_ids | validation_ids)
    assert dev_ids.isdisjoint(validation_ids)
