from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.native_sparse_experiments.route_trace import (
    QWEN35_LAYERS,
    TraceFormatError,
    analyze_routes,
    lru_replay,
    read_trace,
    validate_record,
)


FIXTURES = Path(__file__).parent / "fixtures"


def make_record(token: int, shift: int = 0) -> dict:
    layers = []
    for layer in range(QWEN35_LAYERS):
        start = (shift + layer) % 32
        layers.append([((start + expert) % 32) for expert in range(8)])
    return {"token": token, "layers": layers}


def test_fixture_validates_exact_qwen35_shape() -> None:
    routes = read_trace(FIXTURES / "valid_one_token.json")
    assert len(routes) == 1
    assert len(routes[0].layers) == 40
    assert all(len(experts) == 8 for experts in routes[0].layers)


def test_wrong_top_k_is_rejected() -> None:
    with pytest.raises(TraceFormatError, match="expected 40"):
        read_trace(FIXTURES / "invalid_top_k.json")


def test_duplicate_and_out_of_range_experts_are_rejected() -> None:
    duplicate = make_record(0)
    duplicate["layers"][0][7] = duplicate["layers"][0][0]
    with pytest.raises(TraceFormatError, match="duplicate"):
        validate_record(duplicate)

    out_of_range = make_record(0)
    out_of_range["layers"][0][0] = 256
    with pytest.raises(TraceFormatError, match="outside"):
        validate_record(out_of_range)


def test_lru_replay_counts_global_bundle_hits_and_misses() -> None:
    first = validate_record(make_record(0))
    second = validate_record(make_record(1))
    stats = lru_replay([first, second], capacity=QWEN35_LAYERS * 8)
    assert stats["request_count"] == 640
    assert stats["miss_count"] == 320
    assert stats["hit_count"] == 320
    assert stats["misses_by_token"] == [320, 0]


def test_analysis_reports_overlap_popularity_and_fresh_bytes() -> None:
    routes = [validate_record(make_record(0)), validate_record(make_record(1))]
    report = analyze_routes(routes, capacities=[0, 320], expert_bundle_bytes=100)
    assert report["shape"] == {"layers": 40, "top_k": 8, "expert_count": 256, "tokens": 2}
    assert report["request_count"] == 640
    assert report["uncached_bytes_per_token"] == 40 * 8 * 100
    assert report["layer_overlap"]["mean_shared_experts"] == 8
    assert report["layer_overlap"]["mean_retention_fraction"] == 1
    assert report["token_overlap"]["adjacent_pairs"] == 1
    assert report["lru"][0]["fresh_expert_bytes_per_token"] == 32_000
    assert report["lru"][1]["fresh_expert_bytes_per_token"] == 16_000


def test_json_array_input_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    path.write_text(json.dumps([make_record(0)]), encoding="utf-8")
    assert len(read_trace(path)) == 1
