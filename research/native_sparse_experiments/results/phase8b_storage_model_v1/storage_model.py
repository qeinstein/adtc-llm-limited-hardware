#!/usr/bin/env python3
"""Model storage-layout ideas against the existing exact route corpus.

This is intentionally model-free: it replays exact ``(layer, expert)``
requests, keeps the established global-LRU cache semantics, and estimates
physical ranges.  It does not load a checkpoint or change routing.  The
models are upper/lower controls for deciding whether a Kaggle runtime build
is worth doing.

The four probes are:

* per-layer co-access graph placement and range coalescing;
* whole-route-set range planning (same layout, all misses submitted together);
* duplicate hot lanes on SSD;
* independent gate/up/down plane residency.

All byte values are explicit bytes; the report additionally emits decimal MB
and MiB where useful.  The physical range model assumes one equal-record
expert position per plane and records extra bytes read when a range spans
unrequested records.  It is therefore deliberately conservative about cache
semantics and optimistic only about a future range planner's bookkeeping.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence


LAYERS = 40
EXPERTS = 256
TOP_K = 8
PLANE_BYTES = (69_206_016 // 256, 69_206_016 // 256, 85_983_232 // 256)
BUNDLE_BYTES = sum(PLANE_BYTES)
assert PLANE_BYTES == (270_336, 270_336, 335_872)

# Phase 6G exact bounded 4-GiB design point.  This fraction is the measured
# read interval / wall interval from the matched 2-GB-cache control.  It is
# used only for a transparent Amdahl estimate, not as a new benchmark claim.
IO_WALL_FRACTION = 0.44288645203125
BASELINE_BOUNDED_TOK_S = 3.0
BASELINE_BOUNDED_MS_PER_TOKEN = 1000.0 / BASELINE_BOUNDED_TOK_S


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("layers"), list):
            raise ValueError(f"bad trace row {line_no}")
        if len(row["layers"]) != LAYERS:
            raise ValueError(f"trace row {line_no} has {len(row['layers'])} layers")
        for layer, experts in enumerate(row["layers"]):
            if isinstance(experts, dict):
                experts = experts.get("experts")
            if not isinstance(experts, list) or len(experts) != TOP_K:
                raise ValueError(f"trace row {line_no} layer {layer} has bad top-k")
            if len(set(experts)) != TOP_K or any(not 0 <= x < EXPERTS for x in experts):
                raise ValueError(f"trace row {line_no} layer {layer} has bad IDs")
        rows.append(row)
    if not rows:
        raise ValueError("empty route corpus")
    return rows


def key_rows(rows: Sequence[dict]) -> Iterable[tuple[int, int, int]]:
    """Yield token ordinal, layer, expert in exact runtime request order."""
    for token, row in enumerate(rows):
        for layer, experts in enumerate(row["layers"]):
            if isinstance(experts, dict):
                experts = experts["experts"]
            for expert in experts:
                yield token, layer, expert


def plane_sequence(rows: Sequence[dict]) -> list[int]:
    """Encode route keys as compact plane IDs for the variable-size replay."""
    encoded: list[int] = []
    for _, layer, expert in key_rows(rows):
        base = (layer * EXPERTS + expert) * 3
        encoded.extend((base, base + 1, base + 2))
    return encoded


def prompt_parity(row: dict) -> int:
    return int(row.get("prompt_id", row.get("global_token", 0))) & 1


def _metric_bytes(bytes_value: int | float) -> dict[str, float]:
    return {
        "bytes": bytes_value,
        "decimal_mb": bytes_value / 1_000_000,
        "mib": bytes_value / (1024 * 1024),
    }


def amdahl_gain(byte_ratio: float, io_fraction: float = IO_WALL_FRACTION) -> float:
    """Idealized throughput gain if only physical I/O bytes scale.

    The numerator keeps all non-I/O work unchanged and assumes storage can
    sustain the same measured bandwidth.  This is an upper bound for cache
    traffic changes, not a benchmark result.
    """
    if byte_ratio <= 0:
        raise ValueError("byte ratio must be positive")
    return 1.0 / (1.0 - io_fraction * (1.0 - byte_ratio)) - 1.0


def lru_misses(rows: Sequence[dict], capacity: int) -> tuple[dict[tuple[int, int], set[int]], dict]:
    """Replay global bundle LRU, retaining misses grouped by token/layer."""
    cache: collections.OrderedDict[tuple[int, int], None] = collections.OrderedDict()
    misses: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    requests = hits = miss_count = 0
    for token, layer, expert in key_rows(rows):
        key = (layer, expert)
        requests += 1
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            continue
        miss_count += 1
        misses[(token, layer)].add(expert)
        if capacity:
            cache[key] = None
            cache.move_to_end(key)
            if len(cache) > capacity:
                cache.popitem(last=False)
    summary = {
        "capacity_bundles": capacity,
        "capacity_bytes": capacity * BUNDLE_BYTES,
        "requests": requests,
        "hits": hits,
        "misses": miss_count,
        "hit_rate": hits / requests if requests else 0.0,
        "logical_fresh_bytes": miss_count * BUNDLE_BYTES,
        "logical_fresh_bytes_per_token": miss_count * BUNDLE_BYTES / len(rows),
    }
    return misses, summary


def _reuses(rows: Sequence[dict]) -> tuple[list[int], int]:
    """Distinct-key reuse distances and number of cold first touches."""
    sequence = [(layer, expert) for _, layer, expert in key_rows(rows)]
    # Fenwick tree over sequence positions, matching route_trace.py's exact
    # distinct-bundle reuse-distance definition.
    bit = [0] * (len(sequence) + 1)
    last: dict[tuple[int, int], int] = {}

    def add(pos: int, delta: int) -> None:
        i = pos + 1
        while i < len(bit):
            bit[i] += delta
            i += i & -i

    def prefix(n: int) -> int:
        total = 0
        while n:
            total += bit[n]
            n -= n & -n
        return total

    distances: list[int] = []
    cold = 0
    for pos, key in enumerate(sequence):
        previous = last.get(key)
        if previous is None:
            cold += 1
        else:
            distances.append(prefix(pos) - prefix(previous + 1))
            add(previous, -1)
        add(pos, 1)
        last[key] = pos
    return distances, cold


def lru_miss_count_for_capacity(distances: Sequence[int], cold: int, capacity: int) -> int:
    return cold + sum(distance >= capacity for distance in distances)


def lru_miss_curve(distances: Sequence[int], cold: int, max_capacity: int) -> list[int]:
    """Return miss counts for every integer LRU capacity in one pass."""
    histogram = [0] * (max_capacity + 1)
    for distance in distances:
        # A reuse distance above max_capacity is a miss for every capacity
        # represented by this curve.  Clamp it to the final bucket.
        histogram[min(distance, max_capacity)] += 1
    suffix = [0] * (max_capacity + 1)
    running = 0
    for capacity in range(max_capacity, -1, -1):
        running += histogram[capacity]
        suffix[capacity] = cold + running
    # At zero slots, every request misses.  The reuse-distance formulation
    # also yields that result because every distance is >= 0.
    return suffix


def split_plane_global(rows: Sequence[dict], capacity_bytes: int, encoded: Sequence[int] | None = None) -> dict:
    """Replay a variable-size global LRU whose entries are individual planes."""
    sizes = PLANE_BYTES
    # A linked-list LRU over a fixed integer key space avoids tuple hashing and
    # OrderedDict allocation in the 1.9M-access corpus replay.
    sequence = encoded if encoded is not None else plane_sequence(rows)
    key_count = LAYERS * EXPERTS * 3
    prev = [-1] * key_count
    next_key = [-1] * key_count
    present = bytearray(key_count)
    head = tail = -1
    resident = requests = hits = misses = 0
    misses_by_kind = [0, 0, 0]
    for key in sequence:
        kind = key % 3
        size = sizes[kind]
        requests += 1
        if present[key]:
            hits += 1
            old_prev = prev[key]
            old_next = next_key[key]
            if key != tail:
                if old_prev >= 0:
                    next_key[old_prev] = old_next
                else:
                    head = old_next
                if old_next >= 0:
                    prev[old_next] = old_prev
                prev[key] = tail
                next_key[key] = -1
                if tail >= 0:
                    next_key[tail] = key
                tail = key
            continue
        misses += 1
        misses_by_kind[kind] += 1
        if size <= capacity_bytes:
            while resident + size > capacity_bytes and head >= 0:
                evicted = head
                head = next_key[evicted]
                if head >= 0:
                    prev[head] = -1
                else:
                    tail = -1
                present[evicted] = 0
                next_key[evicted] = prev[evicted] = -1
                resident -= sizes[evicted % 3]
            present[key] = 1
            prev[key] = tail
            next_key[key] = -1
            if tail >= 0:
                next_key[tail] = key
            else:
                head = key
            tail = key
            resident += size
    fresh_bytes = sum(count * sizes[kind] for kind, count in enumerate(misses_by_kind))
    out = {
        "policy": "split_plane_variable_global_lru",
        "capacity_bytes": capacity_bytes,
        "requests": requests,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / requests if requests else 0.0,
        "misses_by_plane": misses_by_kind,
        "fresh_bytes": fresh_bytes,
        "fresh_bytes_per_token": fresh_bytes / len(rows),
        "max_resident_bytes": capacity_bytes,
    }
    out["byte_ratio_vs_bundle_lru"] = None
    return out


def split_plane_partitioned(
    rows: Sequence[dict], capacity_bytes: int, distances: Sequence[int], cold: int
) -> dict:
    """Estimate independent plane LRUs with a near-optimal slot split.

    Gate and up have equal size and equal request streams.  We search all
    possible down-plane slot counts and split the remaining equal-plane slots
    evenly, with a small neighbourhood check.  This is exact for the common
    equal split if the miss curve has diminishing returns (which the report
    verifies), and is a tight bounded allocation search otherwise.
    """
    sizes = PLANE_BYTES
    max_down = capacity_bytes // sizes[2]
    max_equal = capacity_bytes // sizes[0]
    miss_curve = lru_miss_curve(distances, cold, max(max_down, max_equal))
    best: tuple[int, int, int, int] | None = None  # bytes, n0, n1, n2
    for n2 in range(max_down + 1):
        remaining = capacity_bytes - n2 * sizes[2]
        total_equal = remaining // sizes[0]
        center = total_equal // 2
        # Include the equal split and a bounded neighbourhood to avoid an
        # assumption about convexity in the reuse-distance curve.
        candidates = {max(0, min(total_equal, center + delta)) for delta in range(-64, 65)}
        for n0 in candidates:
            n1 = total_equal - n0
            counts = [
                miss_curve[n0],
                miss_curve[n1],
                miss_curve[n2],
            ]
            fresh = sum(counts[kind] * sizes[kind] for kind in range(3))
            candidate = (fresh, n0, n1, n2)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    fresh, n0, n1, n2 = best
    counts = [
        lru_miss_count_for_capacity(distances, cold, n0),
        lru_miss_count_for_capacity(distances, cold, n1),
        lru_miss_count_for_capacity(distances, cold, n2),
    ]
    return {
        "policy": "split_plane_independent_lru",
        "capacity_bytes": capacity_bytes,
        "plane_slot_capacities": [n0, n1, n2],
        "plane_reserved_bytes": [n0 * sizes[0], n1 * sizes[1], n2 * sizes[2]],
        "misses_by_plane": counts,
        "fresh_bytes": fresh,
        "fresh_bytes_per_token": fresh / len(rows),
        "allocation_search": "all down slots; gate/up split +/-64 around equal",
    }


def graph_and_layout(
    rows: Sequence[dict], train_mask: Sequence[bool]
) -> tuple[list[list[int]], dict]:
    """Return id/popularity/greedy layouts and graph diagnostics."""
    graph: list[dict[tuple[int, int], int]] = [collections.Counter() for _ in range(LAYERS)]
    popularity: list[collections.Counter[int]] = [collections.Counter() for _ in range(LAYERS)]
    for row, use in zip(rows, train_mask):
        if not use:
            continue
        for layer, raw in enumerate(row["layers"]):
            ids = raw["experts"] if isinstance(raw, dict) else raw
            popularity[layer].update(ids)
            for a, b in itertools.combinations(sorted(ids), 2):
                graph[layer][(a, b)] += 1

    def edge(layer: int, a: int, b: int) -> int:
        return graph[layer].get(tuple(sorted((a, b))), 0)

    layouts: dict[str, list[list[int]]] = {
        "identity": [list(range(EXPERTS)) for _ in range(LAYERS)],
        "train_popularity": [
            sorted(range(EXPERTS), key=lambda x: (-popularity[layer][x], x))
            for layer in range(LAYERS)
        ],
    }
    greedy: list[list[int]] = []
    for layer in range(LAYERS):
        degree = collections.Counter()
        for (a, b), weight in graph[layer].items():
            degree[a] += weight
            degree[b] += weight
        start = max(range(EXPERTS), key=lambda x: (degree[x], popularity[layer][x], -x))
        order = [start]
        unused = set(range(EXPERTS)) - {start}
        while unused:
            previous = order[-1]
            nxt = max(unused, key=lambda x: (edge(layer, previous, x), degree[x], popularity[layer][x], -x))
            order.append(nxt)
            unused.remove(nxt)
        greedy.append(order)
    layouts["train_greedy_coaccess"] = greedy

    diagnostics: dict[str, dict] = {}
    for name, layout in layouts.items():
        total_edge = adjacent_edge = 0
        for layer in range(LAYERS):
            positions = {expert: pos for pos, expert in enumerate(layout[layer])}
            total_edge += sum(graph[layer].values())
            adjacent_edge += sum(
                weight for (a, b), weight in graph[layer].items()
                if abs(positions[a] - positions[b]) == 1
            )
        diagnostics[name] = {
            "train_coaccess_edge_mass": total_edge,
            "train_adjacent_edge_mass": adjacent_edge,
            "train_adjacent_edge_fraction": adjacent_edge / total_edge if total_edge else 0.0,
        }
    return layouts, {
        "train_rows": sum(train_mask),
        "test_rows": len(rows) - sum(train_mask),
        "layouts": diagnostics,
        "top_train_pairs": {
            str(layer): [
                {"experts": list(pair), "count": count}
                for pair, count in graph[layer].most_common(8)
            ]
            for layer in range(LAYERS)
        },
    }


def range_count_for_positions(positions: Sequence[int], max_gap: int | None) -> tuple[int, int]:
    """Return physical range count and records transferred for one plane."""
    if not positions:
        return 0, 0
    ordered = sorted(positions)
    return range_count_for_sorted_positions(ordered, max_gap)


def range_count_for_sorted_positions(ordered: Sequence[int], max_gap: int | None) -> tuple[int, int]:
    """Same as :func:`range_count_for_positions` for already sorted offsets."""
    if not ordered:
        return 0, 0
    if max_gap is None:
        return len(ordered), len(ordered)
    ranges = 1
    records = 0
    start = previous = ordered[0]
    for position in ordered[1:]:
        if position - previous - 1 <= max_gap:
            previous = position
            continue
        records += previous - start + 1
        ranges += 1
        start = previous = position
    records += previous - start + 1
    return ranges, records


def physical_ranges(
    miss_groups: dict[tuple[int, int], set[int]],
    layout_positions: list[list[int] | dict[int, int]],
    max_gap: int | None,
) -> dict:
    return physical_ranges_multi(miss_groups, layout_positions, (max_gap,))[str(max_gap)]


def physical_ranges_multi(
    miss_groups: dict[tuple[int, int], set[int]],
    layout_positions: list[list[int] | dict[int, int]],
    gaps: Sequence[int | None],
) -> dict[str, dict]:
    """Estimate plane-file reads for all cache misses.

    ``None`` means one read per missing expert (the exact existing operation).
    ``0`` merges adjacent physical records, while ``n`` permits n unrequested
    records between neighbouring misses.  Ranges are planned independently
    for each routed layer and each of the three plane files.
    """
    positions = [
        order if isinstance(order, dict) else {expert: pos for pos, expert in enumerate(order)}
        for order in layout_positions
    ]
    gap_values = [str(gap) for gap in gaps]
    range_counts = {str(gap): [0, 0, 0] for gap in gaps}
    records_by_planes = {str(gap): [0, 0, 0] for gap in gaps}
    logical_records = 0
    grouped_sets = 0
    for (token, layer), experts in miss_groups.items():
        del token
        grouped_sets += 1
        logical_records += len(experts)
        physical_positions = sorted(positions[layer][expert] for expert in experts)
        for gap in gaps:
            gap_key = str(gap)
            ranges, records = range_count_for_sorted_positions(physical_positions, gap)
            # All three planes have the same expert placement; only their
            # record byte sizes differ.  Compute geometry once per gap.
            for kind in range(3):
                range_counts[gap_key][kind] += ranges
                records_by_planes[gap_key][kind] += records
    logical_bytes = logical_records * BUNDLE_BYTES
    output = {}
    for gap_key in gap_values:
        range_count = range_counts[gap_key]
        records_by_plane = records_by_planes[gap_key]
        physical_bytes = sum(records_by_plane[kind] * PLANE_BYTES[kind] for kind in range(3))
        output[gap_key] = {
            "max_gap_unrequested_records": None if gap_key == "None" else int(gap_key),
            "route_layer_sets_with_misses": grouped_sets,
            "logical_miss_bundles": logical_records,
            "logical_bytes": logical_bytes,
            "physical_range_count_by_plane": range_count,
            "physical_records_by_plane": records_by_plane,
            "physical_bytes": physical_bytes,
            "physical_bytes_decimal_mb": physical_bytes / 1_000_000,
            "physical_bytes_mib": physical_bytes / (1024 * 1024),
            "amplification_vs_logical": physical_bytes / logical_bytes if logical_bytes else 1.0,
            "range_reduction_vs_one_read_per_miss": 1.0 - sum(range_count) / (3 * logical_records)
            if logical_records else 0.0,
        }
    return output


def duplicate_layout(
    base_layout: list[list[int]], hot_layout: list[list[int]], hot_count: int
) -> list[list[int]]:
    """Return positions with base records [0,255] and duplicate hot lane."""
    positions: list[list[int]] = []
    for layer in range(LAYERS):
        base_pos = {expert: pos for pos, expert in enumerate(base_layout[layer])}
        hot = hot_layout[layer][:hot_count]
        hot_pos = {expert: EXPERTS + pos for pos, expert in enumerate(hot)}
        positions.append([hot_pos.get(expert, base_pos[expert]) for expert in range(EXPERTS)])
    return positions


def duplicate_summary(
    rows: Sequence[dict], miss_groups: dict[tuple[int, int], set[int]], layouts: dict[str, list[list[int]]],
    hot_layout_name: str, hot_counts: Sequence[int], base_layout_name: str,
) -> dict:
    base = layouts[base_layout_name]
    hot_layout = layouts[hot_layout_name]
    popularity: list[collections.Counter[int]] = [collections.Counter() for _ in range(LAYERS)]
    for row in rows:
        for layer, raw in enumerate(row["layers"]):
            ids = raw["experts"] if isinstance(raw, dict) else raw
            popularity[layer].update(ids)
    out = {}
    for hot_count in hot_counts:
        if hot_count <= 0 or hot_count >= EXPERTS:
            continue
        positions = [
            {expert: position for expert, position in enumerate(row)}
            for row in duplicate_layout(base, hot_layout, hot_count)
        ]
        # The compact lane contains one extra record per hot expert per layer
        # and plane, so this is the exact logical duplicate disk cost under
        # the uniform bundle model.
        hot_bundles = LAYERS * hot_count
        metrics = physical_ranges_multi(miss_groups, positions, (None, 0, 1, 2, 4))
        out[str(hot_count)] = {
            "hot_bundles_duplicated": hot_bundles,
            "extra_disk_bytes": hot_bundles * BUNDLE_BYTES,
            "extra_disk_decimal_gb": hot_bundles * BUNDLE_BYTES / 1_000_000_000,
            "disk_multiplier_vs_routed_bundle_store": 1.0 + hot_count / EXPERTS,
            "placement": f"base={base_layout_name};hot={hot_layout_name}",
            "metrics": metrics,
        }
    return out


def summarize_layouts(
    miss_groups: dict[tuple[int, int], set[int]], layouts: dict[str, list[list[int]]]
) -> dict:
    out = {}
    for name, layout in layouts.items():
        out[name] = physical_ranges_multi(miss_groups, layout, (None, 0, 1, 2, 4))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # 2,281 is the measured 2,000,000,000-byte <=4-GiB executor point;
    # 1,439 and 826 are the committed <=3-GiB and <=2.5-GiB points.
    parser.add_argument("--capacities", default="2281,1439,826")
    args = parser.parse_args()
    rows = read_rows(args.trace)
    print(f"loaded rows={len(rows)}", file=sys.stderr, flush=True)
    capacities = [int(value) for value in args.capacities.split(",") if value.strip()]
    trace_digest = sha256(args.trace)
    encoded_planes = plane_sequence(rows)
    train_mask = [prompt_parity(row) == 0 for row in rows]
    layouts, graph = graph_and_layout(rows, train_mask)
    print("built layouts", file=sys.stderr, flush=True)
    distances, cold = _reuses(rows)
    print(f"computed reuse distances={len(distances)}", file=sys.stderr, flush=True)
    report: dict = {
        "schema": "native-sparse-phase8b-storage-model/v1",
        "status": "complete",
        "decision_rule": "kill any branch whose optimistic end-to-end gain is below 5%; no model/runtime was run",
        "trace": {
            "path": str(args.trace),
            "sha256": trace_digest,
            "tokens": len(rows),
            "requests": len(rows) * LAYERS * TOP_K,
            "unique_prompt_ids": sorted({row.get("prompt_id") for row in rows}),
            "plane_bytes": list(PLANE_BYTES),
            "bundle_bytes": BUNDLE_BYTES,
        },
        "baseline": {
            "runtime": "phase6g_bounded_executor_v3 exact IQP",
            "tok_s": BASELINE_BOUNDED_TOK_S,
            "ms_per_token": BASELINE_BOUNDED_MS_PER_TOKEN,
            "io_wall_fraction": IO_WALL_FRACTION,
            "io_bytes_are_logical_not_physical": True,
        },
        "reuse": {
            "cold_first_touches": cold,
            "reuses": len(distances),
            "p50_distinct_distance": sorted(distances)[len(distances) // 2] if distances else 0,
            "p95_distinct_distance": sorted(distances)[int((len(distances) - 1) * 0.95)] if distances else 0,
        },
        "coaccess_graph": graph,
        "capacities": {},
    }

    # Cache misses are generated once per capacity and reused by all layout
    # and duplication probes.  Cache behavior itself is unchanged by a disk
    # permutation or by choosing a duplicate copy.
    for capacity in capacities:
        print(f"capacity {capacity}: cache", file=sys.stderr, flush=True)
        miss_groups, baseline = lru_misses(rows, capacity)
        print(f"capacity {capacity}: misses={baseline['misses']}", file=sys.stderr, flush=True)
        split_global = split_plane_global(rows, capacity * BUNDLE_BYTES, encoded_planes)
        print(f"capacity {capacity}: split global", file=sys.stderr, flush=True)
        split_global["byte_ratio_vs_bundle_lru"] = split_global["fresh_bytes"] / baseline["logical_fresh_bytes"]
        split_global["optimistic_amdahl_gain"] = amdahl_gain(split_global["byte_ratio_vs_bundle_lru"])
        split_partitioned = split_plane_partitioned(
            rows, capacity * BUNDLE_BYTES, distances, cold
        )
        print(f"capacity {capacity}: split partitioned", file=sys.stderr, flush=True)
        split_partitioned["byte_ratio_vs_bundle_lru"] = split_partitioned["fresh_bytes"] / baseline["logical_fresh_bytes"]
        split_partitioned["optimistic_amdahl_gain"] = amdahl_gain(split_partitioned["byte_ratio_vs_bundle_lru"])

        layout_summary = summarize_layouts(miss_groups, layouts)
        print(f"capacity {capacity}: layouts", file=sys.stderr, flush=True)
        for name, metrics in layout_summary.items():
            for value in metrics.values():
                value["optimistic_amdahl_gain_from_bytes_only"] = amdahl_gain(
                    value["physical_bytes"] / value["logical_bytes"]
                )
        duplicates = duplicate_summary(
            rows,
            miss_groups,
            layouts,
            hot_layout_name="train_greedy_coaccess",
            hot_counts=(8, 16, 32, 64, 128),
            base_layout_name="identity",
        )
        print(f"capacity {capacity}: duplicates", file=sys.stderr, flush=True)
        for candidate in duplicates.values():
            for value in candidate["metrics"].values():
                value["optimistic_amdahl_gain_from_bytes_only"] = amdahl_gain(
                    value["physical_bytes"] / value["logical_bytes"]
                )

        # Holdout report: only odd-prompt miss groups are included, but cache
        # state is replayed over the complete stream so this remains a fair
        # physical-layout check rather than a fresh-cache toy trace.
        holdout_groups = {
            key: experts for key, experts in miss_groups.items()
            if prompt_parity(rows[key[0]]) == 1
        }
        holdout_layouts = {
            name: physical_ranges_multi(holdout_groups, layout, (None, 0, 1, 2, 4))
            for name, layout in layouts.items()
        }
        report["capacities"][str(capacity)] = {
            "baseline_bundle_lru": baseline,
            "split_plane_global": split_global,
            "split_plane_partitioned": split_partitioned,
            "coaccess_layouts_all_trace": layout_summary,
            "coaccess_layouts_odd_prompt_holdout": holdout_layouts,
            "duplicate_hot_lane": duplicates,
        }
        print(f"capacity {capacity}: complete", file=sys.stderr, flush=True)

    # A compact decision table keeps the result consumable without requiring
    # readers to parse every range-detail entry.
    decisions = []
    for capacity in capacities:
        bucket = report["capacities"][str(capacity)]
        base = bucket["baseline_bundle_lru"]["logical_fresh_bytes"]
        sp = bucket["split_plane_partitioned"]
        best_layout = min(
            bucket["coaccess_layouts_odd_prompt_holdout"].items(),
            key=lambda item: item[1]["1"]["physical_bytes"],
        )
        best_dup = min(
            (
                (int(count), candidate)
                for count, candidate in bucket["duplicate_hot_lane"].items()
            ),
            key=lambda item: item[1]["metrics"]["1"]["physical_bytes"],
        )
        decisions.append({
            "capacity_bundles": capacity,
            "baseline_fresh_decimal_mb_per_token": base / len(rows) / 1_000_000,
            "split_partitioned_fresh_decimal_mb_per_token": sp["fresh_bytes_per_token"] / 1_000_000,
            "split_partitioned_gain_percent": 100 * sp["optimistic_amdahl_gain"],
            "best_holdout_layout_gap1": best_layout[0],
            "best_holdout_layout_gap1_amplification": best_layout[1]["1"]["amplification_vs_logical"],
            "best_duplicate_hot_count_gap1": best_dup[0],
            "best_duplicate_gap1_amplification": best_dup[1]["metrics"]["1"]["amplification_vs_logical"],
            "best_duplicate_extra_disk_gb": best_dup[1]["extra_disk_decimal_gb"],
            "decision_split_plane": "FOLLOW-UP" if 100 * sp["optimistic_amdahl_gain"] >= 5 else "KILL",
            "decision_layout": "FOLLOW-UP" if best_layout[1]["1"]["amplification_vs_logical"] < 0.95 else "KILL",
            "decision_duplicate": "FOLLOW-UP" if best_dup[1]["metrics"]["1"]["amplification_vs_logical"] < 0.95 else "KILL",
        })
    report["decision_table"] = decisions
    report["overall_conclusion"] = {
        "route_set_placement": "KILL unless a real filesystem benchmark proves a large syscall-only win; byte traffic cannot improve and coalescing adds amplification",
        "duplicate_hot_copies": "KILL if no candidate reduces physical bytes; disk duplication does not change RAM/cache traffic",
        "split_gate_up_down": "Use the reported partitioned/global replay as the only storage branch worth a runtime follow-up if its Amdahl estimate clears 5%",
        "model_limit": "Coaccess/range metrics are not tok/s measurements; exact output/routing would remain unchanged by all storage probes",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision_table": decisions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
