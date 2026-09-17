from __future__ import annotations

from prefetch_oracle import LAYERS, PromptTrace, RouteStage
from prefetch_predictor import TransitionPredictor


def _traces() -> tuple[PromptTrace, ...]:
    traces = []
    ordinal = 0
    for prompt_id in range(2):
        tokens = []
        stages = []
        for token in range(3):
            layers = []
            for layer in range(LAYERS):
                current = tuple((layer + token + i) % 256 for i in range(8))
                layers.append(current)
                stages.append(RouteStage(prompt_id, token, layer, current, ordinal))
                ordinal += 1
            tokens.append(tuple(layers))
        traces.append(PromptTrace(prompt_id, tuple(tokens), tuple(stages)))
    return tuple(traces)


def test_copy_last_plan_has_one_stage_per_nonfinal_token_layer() -> None:
    traces = _traces()
    predictor = TransitionPredictor(traces[:1])
    plan, stats = predictor.build_plan(traces[1:], mode="copy_last")
    assert len(plan) == 2 * LAYERS
    assert stats["planned_experts"] == 2 * LAYERS * 8
    assert stats["target_recall"] == 7.0 / 8.0


def test_transition_predictor_is_deterministic_and_fills_top8() -> None:
    traces = _traces()
    predictor = TransitionPredictor(traces[:1])
    stage = traces[1].stages[0]
    predicted = predictor.predict(stage)
    assert len(predicted) == 8
    assert len(set(predicted)) == 8
    assert predicted == predictor.predict(stage)
