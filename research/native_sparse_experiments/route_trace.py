#!/usr/bin/env python3
"""Analyze selected-expert traces for a Qwen3.5-style routed model.

The analyzer does not run a model. It consumes JSONL records like::

    {"token": 0, "layers": [[0, 1, 2, 3, 4, 5, 6, 7], ... 40 lists ...]}

Each layer must contain exactly eight distinct expert IDs in ``[0, 256)``.
Cache keys are global ``(layer, expert_id)`` bundles, matching the global-LRU
design in the research notes.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

QWEN35_LAYERS = 40
QWEN35_EXPERTS = 256
QWEN35_TOP_K = 8
DEFAULT_EXPERT_BUNDLE_BYTES = 1_700_000


class TraceFormatError(ValueError):
    """Raised when a trace does not have the expected routed shape."""


@dataclass(frozen=True)
class TokenRoutes:
    """Validated routes for one decoded token."""

    token: int
    layers: tuple[tuple[int, ...], ...]

    @property
    def requests(self) -> Iterator[tuple[int, int]]:
        for layer, experts in enumerate(self.layers):
            for expert in experts:
                yield layer, expert


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceFormatError(f"{where} must be an integer")
    return value


def validate_record(
    record: dict[str, Any],
    *,
    layers: int = QWEN35_LAYERS,
    top_k: int = QWEN35_TOP_K,
    expert_count: int = QWEN35_EXPERTS,
    ordinal: int = 0,
) -> TokenRoutes:
    """Validate and normalize one trace record.

    ``layers`` may be a list of expert lists or objects containing an
    ``experts`` list. Extra object fields are ignored intentionally.
    """

    if not isinstance(record, dict):
        raise TraceFormatError(f"record {ordinal} must be a JSON object")
    raw_token = record.get("token", record.get("token_id", ordinal))
    token = _as_int(raw_token, f"record {ordinal}.token")
    raw_layers = record.get("layers")
    if not isinstance(raw_layers, list):
        raise TraceFormatError(f"record {ordinal}.layers must be a list")
    if len(raw_layers) != layers:
        raise TraceFormatError(
            f"record {ordinal} has {len(raw_layers)} layers; expected {layers}"
        )

    normalized: list[tuple[int, ...]] = []
    for layer_idx, raw_layer in enumerate(raw_layers):
        raw_experts = raw_layer.get("experts") if isinstance(raw_layer, dict) else raw_layer
        if not isinstance(raw_experts, list):
            raise TraceFormatError(
                f"record {ordinal}.layers[{layer_idx}] must contain an experts list"
            )
        if len(raw_experts) != top_k:
            raise TraceFormatError(
                f"record {ordinal}.layers[{layer_idx}] has {len(raw_experts)} experts; "
                f"expected top_k={top_k}"
            )
        experts = tuple(
            _as_int(value, f"record {ordinal}.layers[{layer_idx}].experts[{i}]")
            for i, value in enumerate(raw_experts)
        )
        if len(set(experts)) != top_k:
            raise TraceFormatError(
                f"record {ordinal}.layers[{layer_idx}] contains duplicate expert IDs"
            )
        bad = [expert for expert in experts if not 0 <= expert < expert_count]
        if bad:
            raise TraceFormatError(
                f"record {ordinal}.layers[{layer_idx}] expert IDs {bad} outside "
                f"[0, {expert_count})"
            )
        normalized.append(experts)
    return TokenRoutes(token=token, layers=tuple(normalized))


def read_trace(
    path: str | Path,
    *,
    layers: int = QWEN35_LAYERS,
    top_k: int = QWEN35_TOP_K,
    expert_count: int = QWEN35_EXPERTS,
) -> list[TokenRoutes]:
    """Read JSONL or a JSON array and validate every token record."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise TraceFormatError(f"trace is empty: {source}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records: list[Any] = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TraceFormatError(f"invalid JSON on line {line_no}: {exc}") from exc
    else:
        records = parsed if isinstance(parsed, list) else [parsed]

    routes = [
        validate_record(
            record,
            layers=layers,
            top_k=top_k,
            expert_count=expert_count,
            ordinal=ordinal,
        )
        for ordinal, record in enumerate(records)
    ]
    if not routes:
        raise TraceFormatError(f"trace is empty: {source}")
    return routes


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def lru_replay(routes: Sequence[TokenRoutes], capacity: int) -> dict[str, Any]:
    """Replay a global bundle LRU and return request/hit/miss statistics."""

    if capacity < 0:
        raise ValueError("LRU capacity must be non-negative")
    cache: collections.OrderedDict[tuple[int, int], None] = collections.OrderedDict()
    requests = hits = misses = 0
    misses_by_token: list[int] = []
    for token_routes in routes:
        token_misses = 0
        for key in token_routes.requests:
            requests += 1
            if key in cache:
                hits += 1
                cache.move_to_end(key)
                continue
            misses += 1
            token_misses += 1
            if capacity:
                cache[key] = None
                cache.move_to_end(key)
                if len(cache) > capacity:
                    cache.popitem(last=False)
        misses_by_token.append(token_misses)
    unique_pairs = {key for route in routes for key in route.requests}
    return {
        "capacity_bundles": capacity,
        "request_count": requests,
        "hit_count": hits,
        "miss_count": misses,
        "hit_rate": hits / requests if requests else 0.0,
        "miss_rate": misses / requests if requests else 0.0,
        "misses_by_token": misses_by_token,
        "fresh_requests_per_token": misses / len(routes) if routes else 0.0,
        "max_cache_occupancy_bundles": min(capacity, len(unique_pairs)),
    }


def analyze_routes(
    routes: Sequence[TokenRoutes],
    *,
    capacities: Iterable[int] = (0, 64, 128, 256, 512, 1024),
    expert_bundle_bytes: int = DEFAULT_EXPERT_BUNDLE_BYTES,
) -> dict[str, Any]:
    """Compute route popularity, temporal overlap, and cache/I/O metrics."""

    if expert_bundle_bytes <= 0:
        raise ValueError("expert_bundle_bytes must be positive")
    if not routes:
        raise ValueError("at least one token route is required")

    popularity: collections.Counter[int] = collections.Counter()
    layer_popularity: dict[str, dict[str, int]] = {}
    unique_pairs: set[tuple[int, int]] = set()
    token_request_counts: list[int] = []
    layer_overlap_counts: list[int] = []
    layer_overlap_ratios: list[float] = []
    token_overlap_counts: list[int] = []
    token_overlap_ratios: list[float] = []

    previous: TokenRoutes | None = None
    for current in routes:
        token_request_counts.append(sum(len(experts) for experts in current.layers))
        for layer, expert in current.requests:
            popularity[expert] += 1
            layer_counts = layer_popularity.setdefault(str(layer), {})
            expert_key = str(expert)
            layer_counts[expert_key] = layer_counts.get(expert_key, 0) + 1
            unique_pairs.add((layer, expert))
        if previous is not None:
            prev_all = set(previous.requests)
            curr_all = set(current.requests)
            overlap = len(prev_all & curr_all)
            union = len(prev_all | curr_all)
            token_overlap_counts.append(overlap)
            token_overlap_ratios.append(overlap / union if union else 0.0)
            for prev_layer, curr_layer in zip(previous.layers, current.layers):
                count = len(set(prev_layer) & set(curr_layer))
                layer_overlap_counts.append(count)
                layer_overlap_ratios.append(count / len(curr_layer))
        previous = current

    requests = sum(token_request_counts)
    capacities_out: list[dict[str, Any]] = []
    seen_capacities: set[int] = set()
    for raw_capacity in capacities:
        capacity = int(raw_capacity)
        if capacity in seen_capacities:
            continue
        seen_capacities.add(capacity)
        replay = lru_replay(routes, capacity)
        replay["fresh_expert_bytes_total"] = replay["miss_count"] * expert_bundle_bytes
        replay["fresh_expert_bytes_per_token"] = replay["fresh_expert_bytes_total"] / len(routes)
        capacities_out.append(replay)

    return {
        "shape": {
            "layers": len(routes[0].layers),
            "top_k": len(routes[0].layers[0]),
            "expert_count": QWEN35_EXPERTS,
            "tokens": len(routes),
        },
        "request_count": requests,
        "requests_per_token": requests / len(routes),
        "requests_per_token_min": min(token_request_counts),
        "requests_per_token_max": max(token_request_counts),
        "unique_layer_expert_bundles": len(unique_pairs),
        "unique_expert_ids": len(popularity),
        "expert_bundle_bytes": expert_bundle_bytes,
        "uncached_bytes_per_token": QWEN35_LAYERS * QWEN35_TOP_K * expert_bundle_bytes,
        "expert_popularity": dict(sorted(popularity.items(), key=lambda item: (-item[1], item[0]))),
        "layer_expert_popularity": layer_popularity,
        "token_overlap": {
            "adjacent_pairs": max(0, len(routes) - 1),
            "mean_shared_layer_expert_requests": _mean(token_overlap_counts),
            "mean_jaccard": _mean(token_overlap_ratios),
            "p50_shared_layer_expert_requests": _percentile(token_overlap_counts, 0.50),
        },
        "layer_overlap": {
            "comparisons": len(layer_overlap_counts),
            "mean_shared_experts": _mean(layer_overlap_counts),
            "mean_retention_fraction": _mean(layer_overlap_ratios),
            "p50_shared_experts": _percentile(layer_overlap_counts, 0.50),
        },
        "lru": capacities_out,
    }


def _parse_capacities(value: str) -> list[int]:
    try:
        capacities = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("capacities must be comma-separated integers") from exc
    if any(capacity < 0 for capacity in capacities):
        raise argparse.ArgumentTypeError("capacities must be non-negative")
    return capacities


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path, help="JSONL or JSON route trace")
    parser.add_argument("--output", type=Path, help="write analysis JSON to this path")
    parser.add_argument(
        "--capacities",
        type=_parse_capacities,
        default=[0, 64, 128, 256, 512, 1024],
        help="global LRU capacities in expert bundles (default: 0,64,128,256,512,1024)",
    )
    parser.add_argument(
        "--expert-bundle-bytes",
        type=int,
        default=DEFAULT_EXPERT_BUNDLE_BYTES,
        help=f"bytes per quantized layer/expert bundle (default: {DEFAULT_EXPERT_BUNDLE_BYTES})",
    )
    args = parser.parse_args(argv)
    routes = read_trace(args.trace)
    report = analyze_routes(
        routes,
        capacities=args.capacities,
        expert_bundle_bytes=args.expert_bundle_bytes,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

