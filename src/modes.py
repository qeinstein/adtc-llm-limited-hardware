"""Fast / Medium / High reasoning modes — the ONE canonical definition.

Modes change reasoning effort ONLY: thinking allowance, protected
final-answer allowance, and a short effort-steering line. Everything
safety-relevant (system prompt, guidance retrieval, safety rules, output
lint, sampling temperature) is identical across modes.

Budget mechanics (server-agnostic, enforced in code):
- Phase 1 requests thinking_allowance + answer_allowance tokens.
- If the model spends everything thinking (empty answer), phase 2 feeds
  the thinking back and demands the final answer within answer_allowance.
- Total completion per turn is therefore bounded for every mode,
  including High. An empty final answer is never returned.

Budgets fit the serving context (4096): worst case is High phase 1
(1536+768=2304) + a ~1400-token prompt < 4096, and phase 2 re-sends
with its own bounded window.
"""

from __future__ import annotations

DEFAULT_MODE = "medium"

# thinking_tokens: reasoning allowance. answer_tokens: protected final
# answer (phase 1 share AND the phase-2 regen cap).
MODES: dict[str, dict[str, object]] = {
    "fast": {
        "thinking_tokens": 128,
        "answer_tokens": 256,
        "steering": ("Deliberate briefly, then answer directly and "
                     "concisely."),
    },
    "medium": {
        "thinking_tokens": 512,
        "answer_tokens": 512,
        "steering": "",
    },
    "high": {
        "thinking_tokens": 1536,
        "answer_tokens": 768,
        "steering": ("Think carefully step by step before answering. "
                     "Consider danger signs, alternative causes, and what "
                     "information is missing, then give a complete final "
                     "answer."),
    },
}

PHASE2_INSTRUCTION = (
    "Based on your reasoning above, provide your final answer now. "
    "Be direct and complete; do not continue deliberating."
)


def normalize_mode(name: str | None) -> str:
    """Validate a user-supplied mode. Empty/None -> default (medium)."""
    if name is None or not str(name).strip():
        return DEFAULT_MODE
    key = str(name).strip().lower()
    if key not in MODES:
        raise ValueError(
            f"unknown mode {name!r} (want one of {sorted(MODES)})")
    return key


def budgets(mode: str) -> tuple[int, int]:
    """(thinking_allowance, answer_allowance) for a normalized mode."""
    m = MODES[normalize_mode(mode)]
    return int(m["thinking_tokens"]), int(m["answer_tokens"])


def phase1_max_tokens(mode: str) -> int:
    think, answer = budgets(mode)
    return think + answer


def steering(mode: str) -> str:
    """Short effort-steering line appended to the system prompt."""
    return str(MODES[normalize_mode(mode)]["steering"])


def needs_answer_regen(thinking: str, answer: str) -> bool:
    """True when phase 1 produced reasoning but no usable final answer."""
    return bool((thinking or "").strip()) and not (answer or "").strip()


def phase2_messages(messages: list[dict], thinking: str,
                    max_thinking_chars: int = 6000) -> list[dict]:
    """Build the bounded phase-2 request: thinking back as context, then
    a final-answer instruction. Thinking is tail-truncated so phase 2
    always fits the context window."""
    think = (thinking or "").strip()
    if len(think) > max_thinking_chars:
        think = think[-max_thinking_chars:]
    return (list(messages)
            + [{"role": "assistant", "content": think},
               {"role": "user", "content": PHASE2_INSTRUCTION}])


def describe(mode: str) -> dict:
    """Mode summary for API responses / telemetry."""
    think, answer = budgets(mode)
    return {"mode": normalize_mode(mode),
            "thinking_tokens": think, "answer_tokens": answer,
            "max_tokens": think + answer}
