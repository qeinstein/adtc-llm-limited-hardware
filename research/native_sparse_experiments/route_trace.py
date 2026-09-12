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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

QWEN35_LAYERS = 40
QWEN35_EXPERTS = 256
QWEN35_TOP_K = 8
DEFAULT_EXPERT_BUNDLE_BYTES = 1_700_000
BundleBytes = int | Sequence[int] | Mapping[int, int]


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


def _normalize_bundle_bytes(bundle_bytes: BundleBytes | None) -> tuple[int, ...]:
    """Return one positive bundle size per layer.

    A scalar preserves the original API. A sequence or integer-keyed mapping
    allows exact capacity-byte accounting when packed layer sizes differ.
    """

    if bundle_bytes is None:
        bundle_bytes = DEFAULT_EXPERT_BUNDLE_BYTES
    if isinstance(bundle_bytes, bool):
        raise ValueError("bundle bytes must be positive integers")
    if isinstance(bundle_bytes, int):
        values = (bundle_bytes,) * QWEN35_LAYERS
    elif isinstance(bundle_bytes, Mapping):
        def value_for(layer: int) -> Any:
            if layer in bundle_bytes:
                return bundle_bytes[layer]
            return bundle_bytes.get(str(layer))

        missing = [layer for layer in range(QWEN35_LAYERS) if value_for(layer) is None]
        if missing:
            raise ValueError(f"bundle byte mapping is missing layers {missing}")
        values = tuple(value_for(layer) for layer in range(QWEN35_LAYERS))
    else:
        values = tuple(bundle_bytes)
        if len(values) != QWEN35_LAYERS:
            raise ValueError(f"bundle byte sequence must have {QWEN35_LAYERS} entries")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("bundle bytes must be positive integers")
    return values


def _replay_byte_fields(
    *,
    misses_by_key: collections.Counter[tuple[int, int]],
    misses_by_token: Sequence[int],
    bundle_bytes: Sequence[int],
    token_count: int,
) -> dict[str, Any]:
    fresh_bytes = sum(count * bundle_bytes[layer] for (layer, _), count in misses_by_key.items())
    return {
        "fresh_expert_bytes_total": fresh_bytes,
        "fresh_expert_bytes_per_token": fresh_bytes / token_count if token_count else 0.0,
        "misses_by_token": list(misses_by_token),
    }


def _replay_summary(
    *,
    policy: str,
    capacity: int,
    requests: int,
    hits: int,
    misses: int,
    misses_by_token: Sequence[int],
    unique_pairs: set[tuple[int, int]],
    capacity_bytes: int | None,
    max_resident_bytes: int,
    token_count: int,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "capacity_bundles": capacity,
        "capacity_bytes": capacity_bytes,
        "request_count": requests,
        "hit_count": hits,
        "miss_count": misses,
        "hit_rate": hits / requests if requests else 0.0,
        "miss_rate": misses / requests if requests else 0.0,
        "misses_by_token": list(misses_by_token),
        "fresh_requests_per_token": misses / token_count if token_count else 0.0,
        "max_cache_occupancy_bundles": min(capacity, len(unique_pairs)),
        "max_resident_bytes": max_resident_bytes,
    }


def lru_replay(
    routes: Sequence[TokenRoutes],
    capacity: int,
    *,
    bundle_bytes: BundleBytes | None = None,
) -> dict[str, Any]:
    """Replay the original global bundle LRU.

    With non-uniform layer sizes, ``capacity_bytes`` is ``None`` because a
    bundle-count budget has no single fixed byte size; ``max_resident_bytes``
    records the exact observed resident bytes instead.
    """

    if capacity < 0:
        raise ValueError("LRU capacity must be non-negative")
    sizes = _normalize_bundle_bytes(bundle_bytes)
    cache: collections.OrderedDict[tuple[int, int], None] = collections.OrderedDict()
    requests = hits = misses = resident_bytes = max_resident_bytes = 0
    misses_by_key: collections.Counter[tuple[int, int]] = collections.Counter()
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
            misses_by_key[key] += 1
            if capacity:
                cache[key] = None
                cache.move_to_end(key)
                resident_bytes += sizes[key[0]]
                if len(cache) > capacity:
                    evicted, _ = cache.popitem(last=False)
                    resident_bytes -= sizes[evicted[0]]
            max_resident_bytes = max(max_resident_bytes, resident_bytes)
        misses_by_token.append(token_misses)
    unique_pairs = {key for route in routes for key in route.requests}
    capacity_bytes = capacity * sizes[0] if len(set(sizes)) == 1 else None
    report = _replay_summary(
        policy="global_lru",
        capacity=capacity,
        requests=requests,
        hits=hits,
        misses=misses,
        misses_by_token=misses_by_token,
        unique_pairs=unique_pairs,
        capacity_bytes=capacity_bytes,
        max_resident_bytes=max_resident_bytes,
        token_count=len(routes),
    )
    report["capacity_bytes_lower_bound"] = capacity * min(sizes)
    report["capacity_bytes_upper_bound"] = capacity * max(sizes)
    report.update(_replay_byte_fields(
        misses_by_key=misses_by_key,
        misses_by_token=misses_by_token,
        bundle_bytes=sizes,
        token_count=len(routes),
    ))
    return report


def _partition_capacities(total_capacity: int, layer_count: int = QWEN35_LAYERS) -> list[int]:
    if total_capacity < 0:
        raise ValueError("LRU capacity must be non-negative")
    base, remainder = divmod(total_capacity, layer_count)
    # Remainders go to lower layer indices, making the policy reproducible.
    return [base + (layer < remainder) for layer in range(layer_count)]


def partitioned_lru_replay(
    routes: Sequence[TokenRoutes],
    total_capacity: int,
    *,
    bundle_bytes: BundleBytes | None = None,
) -> dict[str, Any]:
    """Replay independent per-layer LRUs under one bundle budget."""

    sizes = _normalize_bundle_bytes(bundle_bytes)
    capacities = _partition_capacities(total_capacity)
    caches = [collections.OrderedDict() for _ in range(QWEN35_LAYERS)]
    requests = hits = misses = resident_bytes = max_resident_bytes = 0
    misses_by_key: collections.Counter[tuple[int, int]] = collections.Counter()
    misses_by_token: list[int] = []
    unique_pairs = {key for route in routes for key in route.requests}
    for token_routes in routes:
        token_misses = 0
        for key in token_routes.requests:
            requests += 1
            layer, _ = key
            cache = caches[layer]
            if key in cache:
                hits += 1
                cache.move_to_end(key)
                continue
            misses += 1
            token_misses += 1
            misses_by_key[key] += 1
            if capacities[layer]:
                cache[key] = None
                cache.move_to_end(key)
                resident_bytes += sizes[layer]
                if len(cache) > capacities[layer]:
                    evicted, _ = cache.popitem(last=False)
                    resident_bytes -= sizes[evicted[0]]
            max_resident_bytes = max(max_resident_bytes, resident_bytes)
        misses_by_token.append(token_misses)
    report = _replay_summary(
        policy="partitioned_lru",
        capacity=total_capacity,
        requests=requests,
        hits=hits,
        misses=misses,
        misses_by_token=misses_by_token,
        unique_pairs=unique_pairs,
        capacity_bytes=sum(capacities[layer] * sizes[layer] for layer in range(QWEN35_LAYERS)),
        max_resident_bytes=max_resident_bytes,
        token_count=len(routes),
    )
    report["per_layer_capacities_bundles"] = capacities
    report.update(_replay_byte_fields(
        misses_by_key=misses_by_key,
        misses_by_token=misses_by_token,
        bundle_bytes=sizes,
        token_count=len(routes),
    ))
    return report


def static_popularity_replay(
    routes: Sequence[TokenRoutes],
    total_capacity: int,
    *,
    bundle_bytes: BundleBytes | None = None,
) -> dict[str, Any]:
    """Replay a hindsight static cache ordered by per-layer popularity.

    This is an offline oracle upper-control: it sees the complete trace before
    choosing the most-requested ``(layer, expert)`` bundles, so it is not a
    deployable predictor. Ties are deterministic by layer then expert ID.
    """

    if total_capacity < 0:
        raise ValueError("cache capacity must be non-negative")
    sizes = _normalize_bundle_bytes(bundle_bytes)
    popularity: collections.Counter[tuple[int, int]] = collections.Counter(
        key for route in routes for key in route.requests
    )
    selected = sorted(popularity, key=lambda key: (-popularity[key], key[0], key[1]))[:total_capacity]
    selected_set = set(selected)
    requests = hits = misses = 0
    misses_by_key: collections.Counter[tuple[int, int]] = collections.Counter()
    misses_by_token: list[int] = []
    for token_routes in routes:
        token_misses = 0
        for key in token_routes.requests:
            requests += 1
            if key in selected_set:
                hits += 1
            else:
                misses += 1
                token_misses += 1
                misses_by_key[key] += 1
        misses_by_token.append(token_misses)
    selected_bytes = sum(sizes[layer] for layer, _ in selected)
    report = _replay_summary(
        policy="static_popularity_oracle",
        capacity=total_capacity,
        requests=requests,
        hits=hits,
        misses=misses,
        misses_by_token=misses_by_token,
        unique_pairs=set(popularity),
        capacity_bytes=selected_bytes,
        max_resident_bytes=selected_bytes,
        token_count=len(routes),
    )
    report["selected_bundle_count"] = len(selected)
    report["selected_bundle_bytes"] = selected_bytes
    report["selected_bundles_by_layer"] = dict(
        sorted(collections.Counter(layer for layer, _ in selected).items())
    )
    report["oracle_hindsight"] = True
    report["deployable_prediction"] = False
    report.update(_replay_byte_fields(
        misses_by_key=misses_by_key,
        misses_by_token=misses_by_token,
        bundle_bytes=sizes,
        token_count=len(routes),
    ))
    return report


def analyze_routes(
    routes: Sequence[TokenRoutes],
    *,
    capacities: Iterable[int] = (0, 64, 128, 256, 512, 1024),
    expert_bundle_bytes: BundleBytes = DEFAULT_EXPERT_BUNDLE_BYTES,
) -> dict[str, Any]:
    """Compute route popularity, temporal overlap, and cache/I/O metrics."""

    if not routes:
        raise ValueError("at least one token route is required")
    bundle_sizes = _normalize_bundle_bytes(expert_bundle_bytes)

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
    capacity_values: list[int] = []
    seen_capacities: set[int] = set()
    for raw_capacity in capacities:
        capacity = int(raw_capacity)
        if capacity in seen_capacities:
            continue
        seen_capacities.add(capacity)
        capacity_values.append(capacity)
        capacities_out.append(lru_replay(routes, capacity, bundle_bytes=bundle_sizes))
    partitioned_out = [
        partitioned_lru_replay(routes, capacity, bundle_bytes=bundle_sizes)
        for capacity in capacity_values
    ]
    static_out = [
        static_popularity_replay(routes, capacity, bundle_bytes=bundle_sizes)
        for capacity in capacity_values
    ]
    bundle_bytes_output: int | list[int]
    if len(set(bundle_sizes)) == 1:
        bundle_bytes_output = bundle_sizes[0]
    else:
        bundle_bytes_output = list(bundle_sizes)

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
        "expert_bundle_bytes": bundle_bytes_output,
        "uncached_bytes_per_token": sum(
            len(routes[0].layers[layer]) * bundle_sizes[layer]
            for layer in range(len(routes[0].layers))
        ),
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
        "partitioned_lru": partitioned_out,
        "static_popularity_oracle": static_out,
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
    parser.add_argument(
        "--bundle-by-layer-file",
        type=Path,
        help="JSON file containing a 40-item byte list or layer->byte mapping",
    )
    args = parser.parse_args(argv)
    routes = read_trace(args.trace)
    bundle_bytes: BundleBytes = args.expert_bundle_bytes
    if args.bundle_by_layer_file:
        bundle_bytes = json.loads(args.bundle_by_layer_file.read_text(encoding="utf-8"))
    report = analyze_routes(
        routes,
        capacities=args.capacities,
        expert_bundle_bytes=bundle_bytes,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
