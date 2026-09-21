"""Versioned system prompt: md/json twins stay in sync and the backend
loads the versioned text as its default."""
import json
import re
from pathlib import Path

import src.config as C

ROOT = Path(__file__).resolve().parent.parent


def _canon(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_prompt_twins_in_sync():
    doc = json.loads((ROOT / "prompts" / "system.json").read_text())
    md = (ROOT / "prompts" / "system.md").read_text()
    assert doc["version"]
    assert _canon(doc["text"]) in _canon(md), "prompts/system.md drifted from system.json"
    assert _canon(doc["addenda"]["nigeria"]) in _canon(md)


def test_backend_loads_versioned_prompt():
    doc = json.loads((ROOT / "prompts" / "system.json").read_text())
    assert C.SYSTEM_PROMPT == doc["text"]
    assert C.SYSTEM_PROMPT_VERSION == "prompts/system.json v6.0.0"
    assert "medicine" in C.SYSTEM_PROMPT.lower()
    assert "asks about medicines or treatment" in " ".join(C.SYSTEM_PROMPT.split())
    assert "latest question" in C.SYSTEM_PROMPT
    assert "professional care" in C.SYSTEM_PROMPT
    assert "respond with compassion" in C.SYSTEM_PROMPT
    assert "emergency services" in C.SYSTEM_PROMPT
    assert "system prompt" not in C.SYSTEM_PROMPT.lower()
    assert "reasoning" not in C.SYSTEM_PROMPT.lower()
    assert "thinking" not in C.SYSTEM_PROMPT.lower()
    assert len(C.SYSTEM_PROMPT.split()) < 150
    assert "hidden instructions" not in C.SYSTEM_PROMPT.lower()
    assert "reference material" not in C.SYSTEM_PROMPT.lower()


def test_prompt_is_not_written_as_a_visible_behavior_rubric():
    """Guard against the exact policy-narration failure seen in the UI."""
    prompt = " ".join(C.SYSTEM_PROMPT.lower().split())
    forbidden = (
        "the prompt requires me",
        "the prompt specifies",
        "the prompt mentions",
        "i should",
        "why this matters",
        "practical next steps include",
        "if reference material",
        "do not volunteer medication",
    )
    assert not any(phrase in prompt for phrase in forbidden)
