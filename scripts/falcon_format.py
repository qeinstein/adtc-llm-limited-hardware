"""Exact Falcon-H1 chat formatting shared by training and evaluation.

Falcon-H1's template terminates an assistant turn with ``<|im_end|>``.  That
boundary is not interchangeable with the tokenizer's generic
``<|end_of_text|>`` token: using the latter teaches a different sequence from
the one used by chat generation.  This module derives the completion target
from the tokenizer's own full conversation rendering so the prompt and target
cannot silently drift apart.
"""

from __future__ import annotations

from typing import Any


def generation_stop_ids(tokenizer: Any, generation_config: Any = None) -> list[int]:
    """Return real Falcon stop IDs without treating unknown tokens as stops.

    ``convert_tokens_to_ids`` is not an existence check: some tokenizer
    implementations map an unknown string to ``unk_token_id``.  Falcon also
    uses a distinct PAD ID, so adding such a fallback to ``eos_token_id`` can
    stop generation immediately and leave only padded output.
    """
    configured = getattr(generation_config, "eos_token_id", []) if generation_config is not None else []
    raw = [configured] if isinstance(configured, int) else list(configured or [])
    stop_ids: list[int] = []
    for value in raw:
        if isinstance(value, int) and value >= 0:
            stop_ids.append(int(value))
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and eos >= 0:
        stop_ids.append(int(eos))
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    im_end = vocab.get("<|im_end|>")
    forbidden = {getattr(tokenizer, "unk_token_id", None), getattr(tokenizer, "pad_token_id", None)}
    if isinstance(im_end, int) and im_end >= 0 and im_end not in forbidden:
        stop_ids.append(int(im_end))
    return sorted(set(stop_ids))


def _apply_template(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    """Render a Falcon chat sequence across tokenizer API variants."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def render_completion(
    tokenizer: Any,
    messages: list[dict[str, str]],
    answer: str,
) -> tuple[list[int], list[int]]:
    """Return prompt IDs and the exact assistant-completion target IDs.

    The target includes the assistant-turn terminator emitted by the official
    chat template (normally ``<|im_end|>\\n``).  A prefix mismatch is a hard
    error: silently falling back to a generic EOS token would reintroduce the
    bug this helper exists to prevent.
    """
    answer = str(answer).strip()
    if not answer:
        raise ValueError("empty assistant answer")
    prompt_text = _apply_template(tokenizer, messages, add_generation_prompt=True)
    full_text = _apply_template(
        tokenizer,
        messages + [{"role": "assistant", "content": answer}],
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "Falcon chat template changed between generation and full rendering; "
            "refusing to train on misaligned prompt/target text"
        )
    target_text = full_text[len(prompt_text) :]
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    target_ids = list(tokenizer(target_text, add_special_tokens=False)["input_ids"])
    if not target_ids:
        raise ValueError("chat template produced an empty assistant target")
    return prompt_ids, target_ids
