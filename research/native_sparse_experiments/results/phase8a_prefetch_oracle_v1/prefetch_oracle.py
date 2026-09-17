#!/usr/bin/env python3
"""Trace-only oracle for future expert prefetching.

This is deliberately a model-free storage simulator.  It consumes exact
``(token, layer, top-8)`` routes and never predicts or changes a route.  A
single serialized storage service models an NVMe queue; reads overlap the
fixed per-layer compute clock, while a layer waits for all of its selected
experts.  This is an upper-bound probe for an oracle that knows the future
routes, not a runtime implementation.

The canonical corpus is phase5e's 2,016-token, 32-prompt JSONL artifact.  A
prefetch offset is expressed as ``token_lead * 40 + layer_lead`` flattened
layer stages within the same prompt.  For example, ``token_lead=1,
layer_lead=0`` asks for the same layer of the next token, and
``token_lead=0, layer_lead=1`` asks for the next layer.

All byte counters are logical cache payload bytes.  ``physical`` bytes are
only an explicitly labelled amplification estimate; this simulator does not
claim to measure NVMe traffic.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence


LAYERS = 40
TOP_K = 8
EXPERTS = 256


class CorpusFormatError(ValueError):
    """Raised when a route corpus record does not have the native shape."""


@dataclass(frozen=True)
class RouteStage:
    prompt_id: int
    token: int
    layer: int
    experts: tuple[int, ...]
    ordinal: int


@dataclass(frozen=True)
class PromptTrace:
    prompt_id: int
    tokens: tuple[tuple[tuple[int, ...], ...], ...]
    stages: tuple[RouteStage, ...]


def _validate_layers(raw: object, *, where: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, list) or len(raw) != LAYERS:
        raise CorpusFormatError(f"{where} must contain exactly {LAYERS} layers")
    result: list[tuple[int, ...]] = []
    for layer, values in enumerate(raw):
        if not isinstance(values, list) or len(values) != TOP_K:
            raise CorpusFormatError(f"{where}[{layer}] must contain {TOP_K} experts")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in values):
            raise CorpusFormatError(f"{where}[{layer}] experts must be integers")
        if len(set(values)) != TOP_K:
            raise CorpusFormatError(f"{where}[{layer}] contains duplicate experts")
        if any(x < 0 or x >= EXPERTS for x in values):
            raise CorpusFormatError(f"{where}[{layer}] contains an out-of-range expert")
        result.append(tuple(values))
    return tuple(result)


def read_corpus(path: str | Path) -> tuple[PromptTrace, ...]:
    """Load phase5e JSONL and retain prompt boundaries.

    Keeping prompt boundaries prevents an oracle from prefetching a future
    route from another independent prompt.  Cache state itself is retained
    between prompts by default, matching the existing global-LRU replay.
    """

    by_prompt: "collections.OrderedDict[int, list[tuple[int, tuple[tuple[int, ...], ...]]]]" = (
        collections.OrderedDict()
    )
    source = Path(path)
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusFormatError(f"invalid JSON on line {line_no}: {exc}") from exc
        if not isinstance(record, dict):
            raise CorpusFormatError(f"line {line_no} must be a JSON object")
        prompt_id = record.get("prompt_id")
        token = record.get("token")
        if isinstance(prompt_id, bool) or not isinstance(prompt_id, int):
            raise CorpusFormatError(f"line {line_no}.prompt_id must be an integer")
        if isinstance(token, bool) or not isinstance(token, int):
            raise CorpusFormatError(f"line {line_no}.token must be an integer")
        layers = _validate_layers(record.get("layers"), where=f"line {line_no}.layers")
        by_prompt.setdefault(prompt_id, []).append((token, layers))

    traces: list[PromptTrace] = []
    ordinal = 0
    for prompt_id, records in by_prompt.items():
        records.sort(key=lambda item: item[0])
        if [token for token, _ in records] != list(range(len(records))):
            raise CorpusFormatError(f"prompt {prompt_id} tokens are not contiguous from zero")
        stages: list[RouteStage] = []
        token_layers: list[tuple[tuple[int, ...], ...]] = []
        for token, layers in records:
            token_layers.append(layers)
            for layer, experts in enumerate(layers):
                stages.append(RouteStage(prompt_id, token, layer, experts, ordinal))
                ordinal += 1
        traces.append(PromptTrace(prompt_id, tuple(token_layers), tuple(stages)))
    if not traces:
        raise CorpusFormatError(f"corpus is empty: {source}")
    return tuple(traces)


@dataclass(frozen=True)
class SimConfig:
    """Physical and runtime assumptions for one trace replay."""

    capacity_bytes: int = 2_000_000_000
    bundle_bytes: int = 876_544
    bandwidth_mb_s: float = 767.0
    latency_ms: float = 0.05
    reads_per_bundle: int = 3
    compute_ms_per_token: float = 1000.0 / 4.433333333333333
    token_lead: int = 0
    layer_lead: int = 0
    physical_amplification: float = 1.075
    reset_cache_per_prompt: bool = False

    @property
    def capacity_slots(self) -> int:
        return self.capacity_bytes // self.bundle_bytes

    @property
    def compute_ms_per_layer(self) -> float:
        return self.compute_ms_per_token / LAYERS

    @property
    def read_service_ms(self) -> float:
        if self.bandwidth_mb_s <= 0:
            raise ValueError("bandwidth_mb_s must be positive")
        if self.bundle_bytes <= 0 or self.reads_per_bundle <= 0:
            raise ValueError("bundle_bytes and reads_per_bundle must be positive")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        return (self.bundle_bytes / (self.bandwidth_mb_s * 1_000_000.0) * 1000.0
                + self.latency_ms * self.reads_per_bundle)

    def validate(self) -> None:
        if self.capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        if self.token_lead < 0 or self.layer_lead < 0:
            raise ValueError("prefetch leads must be non-negative")
        if self.physical_amplification < 1.0:
            raise ValueError("physical_amplification must be >= 1")
        if self.token_lead == 0 and self.layer_lead == 0:
            raise ValueError("an oracle prefetch run needs a positive lead")
        _ = self.read_service_ms
        if self.compute_ms_per_token <= 0:
            raise ValueError("compute_ms_per_token must be positive")


@dataclass
class _Read:
    key: tuple[int, int]
    source: str
    complete_ms: float
    admitted: bool
    used: bool = False


@dataclass
class _Entry:
    key: tuple[int, int]
    source: str
    used: bool = False
    read: _Read | None = None


@dataclass
class _RunState:
    resident: "collections.OrderedDict[tuple[int, int], _Entry]" = field(
        default_factory=collections.OrderedDict
    )
    pending: dict[tuple[int, int], _Read] = field(default_factory=dict)
    reads: list[_Read] = field(default_factory=list)
    storage_available_ms: float = 0.0
    completion_cursor: int = 0
    pending_admitted_count: int = 0
    now_ms: float = 0.0


class OracleSimulator:
    """Replay demand routes with optional perfect future route prefetch."""

    def __init__(self, config: SimConfig):
        config.validate()
        self.config = config
        self.state = _RunState()
        self._stats: collections.Counter[str] = collections.Counter()
        self._wait_ms_by_token: collections.Counter[int] = collections.Counter()
        self._request_miss_positions: list[tuple[int, tuple[int, int]]] = []
        self._prefetch_use_positions: list[tuple[int, tuple[int, int]]] = []
        self._stage_request_count = 0
        self._max_occupancy = 0

    @property
    def _slots(self) -> int:
        return self.config.capacity_slots

    def _evict_one(self) -> None:
        if not self.state.resident:
            return
        key, entry = self.state.resident.popitem(last=False)
        self._stats["evictions"] += 1
        if entry.source == "prefetch" and not entry.used:
            self._stats["prefetch_pollution_evictions"] += 1
            self._stats["prefetch_wasted_reads"] += 1
            self._stats["prefetch_wasted_bytes"] += self.config.bundle_bytes

    def _admit_completion(self, read: _Read) -> None:
        if not read.admitted or self._slots == 0:
            return
        while len(self.state.resident) >= self._slots:
            self._evict_one()
        entry = _Entry(read.key, read.source, used=read.used, read=read)
        self.state.resident[read.key] = entry
        self.state.resident.move_to_end(read.key)
        self._stats["cache_insertions"] += 1
        self._max_occupancy = max(self._max_occupancy, len(self.state.resident))

    def _complete_until(self, time_ms: float) -> None:
        """Publish storage completions through ``time_ms`` in issue order."""

        while self.state.completion_cursor < len(self.state.reads):
            read = self.state.reads[self.state.completion_cursor]
            if read.complete_ms > time_ms + 1e-9:
                break
            self.state.completion_cursor += 1
            self.state.pending.pop(read.key, None)
            if read.admitted:
                self.state.pending_admitted_count -= 1
            self._admit_completion(read)
            if read.source == "prefetch" and not read.used:
                # Keep the read's eventual usefulness separate from a cache
                # insertion.  It may be marked used by a later demand hit.
                self._stats["prefetch_completed"] += 1

    def _issue(self, key: tuple[int, int], source: str, issue_ms: float) -> _Read:
        if key in self.state.pending:
            return self.state.pending[key]
        start = max(issue_ms, self.state.storage_available_ms)
        complete = start + self.config.read_service_ms
        self.state.storage_available_ms = complete
        admitted = self._reserve(source)
        if not admitted and self._slots > 0 and source == "demand":
            # Demand reads must make progress, but a full set of in-flight
            # prefetches cannot be evicted.  The read is streamed to compute
            # and is intentionally not retained in the bounded cache.
            self._stats["demand_unadmitted_reads"] += 1
        read = _Read(key, source, complete, admitted)
        self.state.pending[key] = read
        if admitted:
            self.state.pending_admitted_count += 1
        self.state.reads.append(read)
        self._stats["reads_issued"] += 1
        self._stats[f"{source}_reads"] += 1
        self._stats[f"{source}_bytes"] += self.config.bundle_bytes
        return read

    def _touch(self, key: tuple[int, int]) -> None:
        entry = self.state.resident[key]
        entry.used = True
        if entry.source == "prefetch":
            # A prefetched read that completed before demand is a useful
            # prefetch hit.  Keep the read object marked so precision/recall
            # count it even though the cache entry was subsequently touched.
            if entry.read is not None:
                entry.read.used = True
        self.state.resident.move_to_end(key)

    def _reserve(self, source: str) -> bool:
        """Reserve one bounded slot, evicting the global-LRU victim."""

        if self._slots == 0:
            return False
        occupied = len(self.state.resident) + self.state.pending_admitted_count
        while occupied >= self._slots and self.state.resident:
            self._evict_one()
            occupied -= 1
        # In-flight reads are pinned until completion.  If they fill the
        # cache, the new read is streamed and its bytes are still charged.
        return occupied < self._slots

    def _demand_stage(self, stage: RouteStage, *, baseline: bool) -> float:
        stage_start = self.state.now_ms
        self._complete_until(stage_start)
        wait_until = stage_start
        waiting: list[_Read] = []
        for expert in stage.experts:
            key = (stage.layer, expert)
            entry = self.state.resident.get(key)
            if entry is not None:
                self._stats["demand_hits"] += 1
                if entry.source == "prefetch":
                    self._stats["prefetch_ready_hits"] += 1
                    self._prefetch_use_positions.append((stage.ordinal, key))
                self._touch(key)
                continue
            pending = self.state.pending.get(key)
            if pending is None:
                self._stats["demand_misses"] += 1
                self._request_miss_positions.append((stage.ordinal, key))
                pending = self._issue(key, "demand", stage_start)
            else:
                self._stats["demand_inflight_waits"] += 1
                if pending.source == "prefetch":
                    self._stats["prefetch_wait_hits"] += 1
                    self._prefetch_use_positions.append((stage.ordinal, key))
            waiting.append(pending)
        if waiting:
            wait_until = max(read.complete_ms for read in waiting)
            for read in waiting:
                read.used = True
            self._complete_until(wait_until)
            # Unadmitted reads are consumed directly; admitted reads now have
            # entries unless a later completion evicted them in this stage.
            for read in waiting:
                entry = self.state.resident.get(read.key)
                if entry is not None:
                    entry.used = True
                    self.state.resident.move_to_end(read.key)
        wait_ms = max(0.0, wait_until - stage_start)
        # Waiting is part of the decode clock.  Without advancing ``now_ms``
        # here, every later layer would incorrectly re-count an absolute
        # storage completion timestamp as fresh stall.
        self.state.now_ms = wait_until
        self._stats["demand_wait_ms"] += wait_ms
        self._stats["demand_layers"] += 1
        self._wait_ms_by_token[stage.token] += wait_ms
        return wait_ms

    def _prefetch_stage(
        self,
        stage: RouteStage,
        prompt: PromptTrace,
        target_override: tuple[int, int, tuple[int, ...]] | None = None,
    ) -> None:
        if target_override is None:
            lead = self.config.token_lead * LAYERS + self.config.layer_lead
            if lead <= 0:
                return
            target_ordinal = (stage.token * LAYERS + stage.layer) + lead
            if target_ordinal >= len(prompt.stages):
                self._stats["prefetch_targets_out_of_range"] += 1
                return
            target = prompt.stages[target_ordinal]
            # The offset is flattened only inside a prompt.  This explicit
            # check guards against malformed/non-contiguous prompt records.
            if target.prompt_id != stage.prompt_id:
                self._stats["prefetch_targets_out_of_range"] += 1
                return
            target_layer, target_experts = target.layer, target.experts
        else:
            target_ordinal, target_layer, target_experts = target_override
            if target_ordinal < 0 or target_ordinal >= len(prompt.stages):
                self._stats["prefetch_targets_out_of_range"] += 1
                return
            if len(target_experts) != TOP_K or len(set(target_experts)) != TOP_K:
                raise CorpusFormatError("prefetch plan target must contain 8 distinct experts")
            if any(expert < 0 or expert >= EXPERTS for expert in target_experts):
                raise CorpusFormatError("prefetch plan contains an out-of-range expert")
        for expert in target_experts:
            key = (target_layer, expert)
            if key in self.state.resident:
                # A predictor often names a key already resident for the
                # current layer (copy-last is the simplest example).  A
                # deployment can promote it in the LRU without issuing a
                # duplicate read; this is still speculative cache policy,
                # never routing.
                self._touch(key)
                self._stats["prefetch_resident_promotions"] += 1
                continue
            if key in self.state.pending:
                self._stats["prefetch_duplicate_or_resident"] += 1
                continue
            self._issue(key, "prefetch", self.state.now_ms)
            self._stats["prefetch_targets"] += 1

    def _clear_prompt(self) -> None:
        # Reads already submitted still consume storage.  Draining here keeps
        # cross-prompt state deterministic when the user asks for isolated
        # sessions, and does not count the idle drain as decode stall.
        self._complete_until(self.state.storage_available_ms)
        self.state.resident.clear()
        self.state.pending.clear()
        self.state.pending_admitted_count = 0

    def run(
        self,
        traces: Sequence[PromptTrace],
        *,
        prefetch: bool,
        prefetch_plan: dict[int, tuple[int, int, tuple[int, ...]]] | None = None,
    ) -> dict[str, object]:
        """Run one exact demand trace and return auditable counters."""

        for prompt in traces:
            if self.config.reset_cache_per_prompt and prompt is not traces[0]:
                self._clear_prompt()
            for stage in prompt.stages:
                self._demand_stage(stage, baseline=not prefetch)
                # Current-layer demand is submitted before future reads.  The
                # future queue then advances while this layer computes.
                if prefetch:
                    self._prefetch_stage(
                        stage,
                        prompt,
                        prefetch_plan.get(stage.ordinal) if prefetch_plan else None,
                    )
                self.state.now_ms += self.config.compute_ms_per_layer
        self._complete_until(self.state.storage_available_ms)
        # Every prefetched read not consumed before the end is wasted traffic.
        for read in self.state.reads:
            if read.source == "prefetch" and not read.used and not read.admitted:
                self._stats["prefetch_wasted_reads"] += 1
                self._stats["prefetch_wasted_bytes"] += self.config.bundle_bytes
            elif read.source == "prefetch" and not read.used and read.admitted:
                # It remains resident but does not help this finite trace.
                self._stats["prefetch_wasted_reads"] += 1
                self._stats["prefetch_wasted_bytes"] += self.config.bundle_bytes

        requests = len(traces) * 0
        requests = sum(len(prompt.stages) * TOP_K for prompt in traces)
        tokens = sum(len(prompt.tokens) for prompt in traces)
        read_bytes = int(self._stats["reads_issued"] * self.config.bundle_bytes)
        demand_bytes = int(self._stats["demand_bytes"])
        prefetch_bytes = int(self._stats["prefetch_bytes"])
        wait_ms = float(self._stats["demand_wait_ms"])
        compute_ms = tokens * self.config.compute_ms_per_token
        decode_ms = compute_ms + wait_ms
        result: dict[str, object] = {
            "tokens": tokens,
            "route_stages": len(traces) * 0 + sum(len(prompt.stages) for prompt in traces),
            "demand_requests": requests,
            "prefetch_enabled": prefetch,
            "prefetch_token_lead": self.config.token_lead if prefetch else 0,
            "prefetch_layer_lead": self.config.layer_lead if prefetch else 0,
            "capacity_bytes": self.config.capacity_bytes,
            "capacity_slots": self._slots,
            "bundle_bytes": self.config.bundle_bytes,
            "bandwidth_mb_s_decimal": self.config.bandwidth_mb_s,
            "latency_ms_per_plane": self.config.latency_ms,
            "reads_per_bundle": self.config.reads_per_bundle,
            "read_service_ms": self.config.read_service_ms,
            "compute_ms_per_token": self.config.compute_ms_per_token,
            "demand_hits": int(self._stats["demand_hits"]),
            "demand_misses": int(self._stats["demand_misses"]),
            "demand_inflight_waits": int(self._stats["demand_inflight_waits"]),
            "demand_unadmitted_reads": int(self._stats["demand_unadmitted_reads"]),
            "demand_hit_rate": self._stats["demand_hits"] / requests if requests else 0.0,
            "reads_issued": int(self._stats["reads_issued"]),
            "demand_reads": int(self._stats["demand_reads"]),
            "prefetch_reads": int(self._stats["prefetch_reads"]),
            "prefetch_targets": int(self._stats["prefetch_targets"]),
            "prefetch_duplicate_or_resident": int(self._stats["prefetch_duplicate_or_resident"]),
            "prefetch_targets_out_of_range": int(self._stats["prefetch_targets_out_of_range"]),
            "logical_read_bytes": read_bytes,
            "logical_demand_bytes": demand_bytes,
            "logical_prefetch_bytes": prefetch_bytes,
            "logical_bytes_per_token": read_bytes / tokens if tokens else 0.0,
            "logical_demand_bytes_per_token": demand_bytes / tokens if tokens else 0.0,
            "logical_prefetch_bytes_per_token": prefetch_bytes / tokens if tokens else 0.0,
            "estimated_physical_bytes_per_token": (
                read_bytes * self.config.physical_amplification / tokens if tokens else 0.0
            ),
            "demand_wait_ms_total": wait_ms,
            "unhidden_io_stall_ms_per_token": wait_ms / tokens if tokens else 0.0,
            "compute_ms_total": compute_ms,
            "decode_ms_total": decode_ms,
            "simulated_tok_s": 1000.0 * tokens / decode_ms if decode_ms else 0.0,
            "cache_evictions": int(self._stats["evictions"]),
            "max_cache_occupancy_slots": self._max_occupancy,
            "prefetch_cache_pollution_evictions": int(self._stats["prefetch_pollution_evictions"]),
            "prefetch_wasted_reads": int(self._stats["prefetch_wasted_reads"]),
            "prefetch_wasted_bytes": int(self._stats["prefetch_wasted_bytes"]),
            "prefetch_wasted_bytes_per_token": (
                self._stats["prefetch_wasted_bytes"] / tokens if tokens else 0.0
            ),
            "prefetch_useful_reads": int(sum(
                1 for read in self.state.reads if read.source == "prefetch" and read.used
            )),
            "prefetch_precision": (
                sum(1 for read in self.state.reads if read.source == "prefetch" and read.used)
                / self._stats["prefetch_reads"]
                if self._stats["prefetch_reads"] else 0.0
            ),
            "prefetch_wait_hits": int(self._stats["prefetch_wait_hits"]),
            "prefetch_ready_hits": int(self._stats["prefetch_ready_hits"]),
            "prefetch_use_positions": [
                [ordinal, layer, expert]
                for ordinal, (layer, expert) in self._prefetch_use_positions
            ],
        }
        result["stage_wait_ms_p50"] = _percentile(
            [self._wait_ms_by_token[token] for token in sorted(self._wait_ms_by_token)], 0.50
        )
        result["stage_wait_ms_p95"] = _percentile(
            [self._wait_ms_by_token[token] for token in sorted(self._wait_ms_by_token)], 0.95
        )
        result["demand_miss_positions"] = [
            [ordinal, layer, expert] for ordinal, (layer, expert) in self._request_miss_positions
        ]
        return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def simulate_pair(
    traces: Sequence[PromptTrace],
    config: SimConfig,
    *,
    prefetch_plan: dict[int, tuple[int, int, tuple[int, ...]]] | None = None,
) -> dict[str, object]:
    """Run no-prefetch and one perfect-offset arm with shared assumptions."""

    baseline_config = SimConfig(**{**config.__dict__, "token_lead": 0, "layer_lead": 1})
    # SimConfig validation intentionally requires a positive lead for a
    # prefetch arm; baseline does not issue prefetch, so the placeholder lead
    # is harmless and keeps one immutable configuration type.
    baseline = OracleSimulator(baseline_config).run(traces, prefetch=False)
    oracle = OracleSimulator(config).run(traces, prefetch=True, prefetch_plan=prefetch_plan)
    baseline_misses = {
        (int(row[0]), (int(row[1]), int(row[2])))
        for row in baseline["demand_miss_positions"]  # type: ignore[index]
    }
    oracle_misses = {
        (int(row[0]), (int(row[1]), int(row[2])))
        for row in oracle["demand_miss_positions"]  # type: ignore[index]
    }
    useful = int(oracle["prefetch_useful_reads"])
    oracle["baseline_demand_misses"] = len(baseline_misses)
    oracle["baseline_demand_hit_rate"] = 1.0 - len(baseline_misses) / int(baseline["demand_requests"])
    oracle["prefetch_recall_vs_baseline_misses"] = (
        useful / len(baseline_misses) if baseline_misses else 0.0
    )
    prefetch_use_positions = {
        (int(row[0]), (int(row[1]), int(row[2])))
        for row in oracle["prefetch_use_positions"]  # type: ignore[index]
    }
    useful_on_baseline_miss = len(prefetch_use_positions & baseline_misses)
    oracle["prefetch_useful_baseline_misses"] = useful_on_baseline_miss
    oracle["prefetch_precision_vs_baseline_misses"] = (
        useful_on_baseline_miss / int(oracle["prefetch_reads"])
        if int(oracle["prefetch_reads"])
        else 0.0
    )
    oracle["prefetch_recall_vs_baseline_misses"] = (
        useful_on_baseline_miss / len(baseline_misses) if baseline_misses else 0.0
    )
    oracle["demand_misses_saved_vs_baseline"] = max(
        0, len(baseline_misses) - len(oracle_misses)
    )
    oracle["baseline"] = baseline
    return oracle


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--capacity-bytes", type=int, action="append", default=None)
    parser.add_argument("--capacity-gib", type=float, action="append", default=None)
    parser.add_argument("--token-lead", type=int, default=1)
    parser.add_argument("--layer-lead", type=int, default=0)
    parser.add_argument("--bandwidth-mb-s", type=float, default=767.0)
    parser.add_argument("--latency-ms", type=float, default=0.05)
    parser.add_argument("--reads-per-bundle", type=int, default=3)
    parser.add_argument("--compute-ms-token", type=float, default=1000.0 / 4.433333333333333)
    parser.add_argument("--bundle-bytes", type=int, default=876_544)
    parser.add_argument("--physical-amplification", type=float, default=1.075)
    parser.add_argument("--reset-cache-per-prompt", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _args()
    traces = read_corpus(args.corpus)
    capacities = list(args.capacity_bytes or [])
    capacities.extend(int(gib * (1024**3)) for gib in (args.capacity_gib or []))
    if not capacities:
        capacities = [2_000_000_000]
    rows: list[dict[str, object]] = []
    for capacity in capacities:
        config = SimConfig(
            capacity_bytes=capacity,
            bundle_bytes=args.bundle_bytes,
            bandwidth_mb_s=args.bandwidth_mb_s,
            latency_ms=args.latency_ms,
            reads_per_bundle=args.reads_per_bundle,
            compute_ms_per_token=args.compute_ms_token,
            token_lead=args.token_lead,
            layer_lead=args.layer_lead,
            physical_amplification=args.physical_amplification,
            reset_cache_per_prompt=args.reset_cache_per_prompt,
        )
        row = simulate_pair(traces, config)
        # Demand miss positions are useful for internal checks but make a
        # report needlessly large; aggregate counts are retained in JSON.
        row.pop("demand_miss_positions", None)
        row.pop("prefetch_use_positions", None)
        baseline = row.get("baseline")
        if isinstance(baseline, dict):
            baseline.pop("demand_miss_positions", None)
        rows.append(row)
    report = {
        "schema": "native-sparse-prefetch-oracle/v1",
        "corpus": {
            "path": str(args.corpus),
            "prompts": len(traces),
            "tokens": sum(len(prompt.tokens) for prompt in traces),
            "route_stages": sum(len(prompt.stages) for prompt in traces),
            "requests": sum(len(prompt.stages) * TOP_K for prompt in traces),
            "layers": LAYERS,
            "top_k": TOP_K,
        },
        "assumptions": {
            "storage": "one serialized NVMe service; asynchronous reads overlap compute",
            "future_knowledge": "perfect exact selected IDs; routing is never changed",
            "physical_bytes": "logical bytes times explicitly labelled amplification estimate",
        },
        "arms": rows,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
