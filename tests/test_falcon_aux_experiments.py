import json
import sys
from pathlib import Path

from scripts.benchmark_falcon_deployment import bench_once, parse_time_report
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
