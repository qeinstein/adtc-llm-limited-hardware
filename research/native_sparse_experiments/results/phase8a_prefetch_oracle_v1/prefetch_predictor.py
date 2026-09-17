#!/usr/bin/env python3
"""Small deployable route-only prefetch predictor.

The predictor sees only already-produced routes.  It learns a transition
table on an earlier prompt split, keyed by ``(layer, current top-8 set)``;
the value is the empirical next-token same-layer expert frequency.  Unseen
conditions fall back to copying the current top-8 IDs.  Predicted IDs are
used only to issue speculative reads; the simulator's demand routes remain
the exact corpus routes.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

from prefetch_oracle import (
    LAYERS,
    TOP_K,
    PromptTrace,
    RouteStage,
    SimConfig,
    read_corpus,
    simulate_pair,
)


Condition = tuple[int, tuple[int, ...]]
Plan = dict[int, tuple[int, int, tuple[int, ...]]]


class TransitionPredictor:
    """Empirical same-layer transition model with deterministic tie breaks."""

    def __init__(self, traces: Sequence[PromptTrace]):
        self.transitions: dict[Condition, collections.Counter[int]] = {}
        self.layer_global: dict[int, collections.Counter[int]] = {
            layer: collections.Counter() for layer in range(LAYERS)
        }
        self.observations = 0
        for prompt in traces:
            for token in range(len(prompt.tokens) - 1):
                for layer in range(LAYERS):
                    current = tuple(sorted(prompt.tokens[token][layer]))
                    nxt = prompt.tokens[token + 1][layer]
                    counter = self.transitions.setdefault((layer, current), collections.Counter())
                    counter.update(nxt)
                    self.layer_global[layer].update(nxt)
                    self.observations += 1

    @staticmethod
    def _choose(counter: collections.Counter[int], fallback: Sequence[int]) -> tuple[int, ...]:
        selected = [expert for expert, _ in sorted(counter.items(), key=lambda p: (-p[1], p[0]))]
        for expert in sorted(fallback):
            if expert not in selected:
                selected.append(expert)
        return tuple(selected[:TOP_K])

    def predict(self, stage: RouteStage) -> tuple[int, ...]:
        condition = (stage.layer, tuple(sorted(stage.experts)))
        counter = self.transitions.get(condition)
        if counter:
            return self._choose(counter, stage.experts)
        # Copy-last is the robust fallback and has no extra model state.
        return tuple(stage.experts)

    def build_plan(
        self,
        traces: Sequence[PromptTrace],
        *,
        prompt_ids: set[int] | None = None,
        mode: str = "transition",
    ) -> tuple[Plan, dict[str, float | int]]:
        plan: Plan = {}
        correct = total = exact = fallback = 0
        for prompt in traces:
            if prompt_ids is not None and prompt.prompt_id not in prompt_ids:
                continue
            for local_idx, stage in enumerate(prompt.stages):
                target_idx = local_idx + LAYERS
                if target_idx >= len(prompt.stages):
                    continue
                target = prompt.stages[target_idx]
                if mode == "copy_last":
                    predicted = tuple(stage.experts)
                    fallback += 1
                elif mode == "transition":
                    condition = (stage.layer, tuple(sorted(stage.experts)))
                    if condition in self.transitions:
                        predicted = self.predict(stage)
                    else:
                        predicted = tuple(stage.experts)
                        fallback += 1
                else:
                    raise ValueError(f"unknown predictor mode: {mode}")
                predicted_set = set(predicted)
                target_set = set(target.experts)
                overlap = len(predicted_set & target_set)
                correct += overlap
                total += TOP_K
                exact += int(predicted_set == target_set)
                plan[stage.ordinal] = (target_idx, target.layer, predicted)
        return plan, {
            "planned_stages": len(plan),
            "planned_experts": total,
            "correct_target_experts": correct,
            "target_exact_sets": exact,
            "fallback_copy_last_stages": fallback,
            "target_precision": correct / total if total else 0.0,
            "target_recall": correct / total if total else 0.0,
            "target_exact_set_rate": exact / len(plan) if plan else 0.0,
        }


def _metric_row(
    name: str,
    traces: Sequence[PromptTrace],
    predictor: TransitionPredictor,
    *,
    capacity_bytes: int,
    mode: str,
    bundle_bytes: int,
    bandwidth_mb_s: float,
    latency_ms: float,
    compute_ms_token: float,
) -> dict[str, object]:
    plan, plan_stats = predictor.build_plan(traces, mode=mode)
    config = SimConfig(
        capacity_bytes=capacity_bytes,
        bundle_bytes=bundle_bytes,
        bandwidth_mb_s=bandwidth_mb_s,
        latency_ms=latency_ms,
        reads_per_bundle=3,
        compute_ms_per_token=compute_ms_token,
        token_lead=1,
        layer_lead=0,
    )
    simulated = simulate_pair(traces, config, prefetch_plan=plan)
    tokens = int(simulated["tokens"])
    wrong_experts = int(plan_stats["planned_experts"]) - int(plan_stats["correct_target_experts"])
    result = {
        "name": name,
        "mode": mode,
        "prompts": len(traces),
        "tokens": tokens,
        "capacity_bytes": capacity_bytes,
        "capacity_slots": simulated["capacity_slots"],
        "plan": plan_stats,
        "plan_wrong_target_bytes_per_token": wrong_experts * bundle_bytes / tokens if tokens else 0.0,
        "demand_misses": simulated["demand_misses"],
        "demand_hit_rate": simulated["demand_hit_rate"],
        "logical_bytes_per_token": simulated["logical_bytes_per_token"],
        "logical_prefetch_bytes_per_token": simulated["logical_prefetch_bytes_per_token"],
        "unhidden_io_stall_ms_per_token": simulated["unhidden_io_stall_ms_per_token"],
        "simulated_tok_s": simulated["simulated_tok_s"],
        "prefetch_reads": simulated["prefetch_reads"],
        "prefetch_useful_reads": simulated["prefetch_useful_reads"],
        "prefetch_precision_vs_baseline_misses": simulated["prefetch_precision_vs_baseline_misses"],
        "prefetch_recall_vs_baseline_misses": simulated["prefetch_recall_vs_baseline_misses"],
        "prefetch_wasted_bytes_per_token": simulated["prefetch_wasted_bytes_per_token"],
        "prefetch_cache_pollution_evictions": simulated["prefetch_cache_pollution_evictions"],
        "baseline": {
            "simulated_tok_s": simulated["baseline"]["simulated_tok_s"],
            "unhidden_io_stall_ms_per_token": simulated["baseline"]["unhidden_io_stall_ms_per_token"],
            "logical_bytes_per_token": simulated["baseline"]["logical_bytes_per_token"],
            "demand_misses": simulated["baseline"]["demand_misses"],
        },
    }
    result["net_speedup_vs_baseline"] = (
        float(result["simulated_tok_s"]) / float(result["baseline"]["simulated_tok_s"])
        if float(result["baseline"]["simulated_tok_s"]) else 0.0
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--capacity-bytes", type=int, default=724_025_344)
    parser.add_argument("--bundle-bytes", type=int, default=876_544)
    parser.add_argument("--bandwidth-mb-s", type=float, default=767.0)
    parser.add_argument("--latency-ms", type=float, default=0.05)
    parser.add_argument("--compute-ms-token", type=float, default=1000.0 / 4.433333333333333)
    parser.add_argument("--train-prompts", type=int, default=16)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    traces = read_corpus(args.corpus)
    ordered = list(traces)
    train = tuple(ordered[:args.train_prompts])
    test = tuple(ordered[args.train_prompts:])
    predictor = TransitionPredictor(train)
    rows = [
        _metric_row("heldout_copy_last", test, predictor, capacity_bytes=args.capacity_bytes,
                    mode="copy_last", bundle_bytes=args.bundle_bytes,
                    bandwidth_mb_s=args.bandwidth_mb_s, latency_ms=args.latency_ms,
                    compute_ms_token=args.compute_ms_token),
        _metric_row("heldout_transition_frequency", test, predictor, capacity_bytes=args.capacity_bytes,
                    mode="transition", bundle_bytes=args.bundle_bytes,
                    bandwidth_mb_s=args.bandwidth_mb_s, latency_ms=args.latency_ms,
                    compute_ms_token=args.compute_ms_token),
    ]
    # In-sample figures are useful diagnostics only; held-out rows are the
    # deployability result and the only rows used for a branch decision.
    rows.extend([
        _metric_row("train_copy_last_diagnostic", train, predictor, capacity_bytes=args.capacity_bytes,
                    mode="copy_last", bundle_bytes=args.bundle_bytes,
                    bandwidth_mb_s=args.bandwidth_mb_s, latency_ms=args.latency_ms,
                    compute_ms_token=args.compute_ms_token),
        _metric_row("train_transition_diagnostic", train, predictor, capacity_bytes=args.capacity_bytes,
                    mode="transition", bundle_bytes=args.bundle_bytes,
                    bandwidth_mb_s=args.bandwidth_mb_s, latency_ms=args.latency_ms,
                    compute_ms_token=args.compute_ms_token),
    ])
    report = {
        "schema": "native-sparse-prefetch-predictor/v1",
        "corpus": {
            "path": str(args.corpus),
            "prompts": len(traces),
            "tokens": sum(len(prompt.tokens) for prompt in traces),
            "train_prompts": len(train),
            "heldout_prompts": len(test),
            "train_prompt_ids": [prompt.prompt_id for prompt in train],
            "heldout_prompt_ids": [prompt.prompt_id for prompt in test],
        },
        "predictor": {
            "kind": "transition_frequency_conditioned_on_layer_and_current_top8",
            "fallback": "copy_last_same_layer",
            "training_transition_observations": predictor.observations,
            "condition_count": len(predictor.transitions),
            "routing_unchanged": True,
        },
        "assumptions": {
            "storage": "same single serialized NVMe service as oracle simulator",
            "prefetch_lead": {"token": 1, "layer": 0},
            "predictor_never_sees_heldout_future_routes": True,
        },
        "results": rows,
    }
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
