from __future__ import annotations

from prefetch_oracle import (
    LAYERS,
    OracleSimulator,
    PromptTrace,
    RouteStage,
    SimConfig,
    simulate_pair,
)


def _prompt(tokens: int = 3) -> tuple[PromptTrace, ...]:
    stages = []
    token_layers = []
    ordinal = 0
    for token in range(tokens):
        layers = []
        for layer in range(LAYERS):
            # Distinct per-token routes make cold traffic visible while the
            # one-token offset remains exactly known to the oracle.
            experts = tuple((layer + token + i) % 256 for i in range(8))
            layers.append(experts)
            stages.append(RouteStage(0, token, layer, experts, ordinal))
            ordinal += 1
        token_layers.append(tuple(layers))
    return (PromptTrace(0, tuple(token_layers), tuple(stages)),)


def test_oracle_prefetch_hides_a_future_layer_read() -> None:
    traces = _prompt(3)
    config = SimConfig(
        capacity_bytes=10_000_000,
        bundle_bytes=1_000,
        bandwidth_mb_s=10.0,
        latency_ms=0.0,
        reads_per_bundle=1,
        compute_ms_per_token=400.0,
        token_lead=1,
        layer_lead=0,
    )
    result = simulate_pair(traces, config)
    baseline = result["baseline"]
    assert result["prefetch_enabled"] is True
    assert result["prefetch_useful_reads"] > 0
    assert result["prefetch_precision"] > 0.0
    assert result["unhidden_io_stall_ms_per_token"] < baseline["unhidden_io_stall_ms_per_token"]
    assert result["simulated_tok_s"] > baseline["simulated_tok_s"]


def test_prefetch_cache_is_bounded_and_pollution_is_reported() -> None:
    traces = _prompt(4)
    config = SimConfig(
        capacity_bytes=1_000,
        bundle_bytes=1_000,
        bandwidth_mb_s=1_000.0,
        latency_ms=0.0,
        reads_per_bundle=1,
        compute_ms_per_token=400.0,
        token_lead=1,
        layer_lead=0,
    )
    result = OracleSimulator(config).run(traces, prefetch=True)
    assert result["capacity_slots"] == 1
    assert result["max_cache_occupancy_slots"] <= 1
    assert result["prefetch_wasted_bytes"] > 0
    assert result["prefetch_cache_pollution_evictions"] > 0


def test_zero_capacity_streams_demand_without_retention() -> None:
    traces = _prompt(1)
    config = SimConfig(
        capacity_bytes=0,
        bundle_bytes=1_000,
        bandwidth_mb_s=1_000.0,
        latency_ms=0.0,
        reads_per_bundle=1,
        compute_ms_per_token=400.0,
        token_lead=1,
        layer_lead=0,
    )
    result = OracleSimulator(config).run(traces, prefetch=True)
    assert result["capacity_slots"] == 0
    assert result["max_cache_occupancy_slots"] == 0
    assert result["demand_hits"] == 0
    assert result["demand_misses"] == 320

