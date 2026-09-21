"""Versioned system prompt: md/json twins stay in sync and the backend
loads the versioned text as its default."""
import json
import re
from pathlib import Path

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
    import src.config as C

    doc = json.loads((ROOT / "prompts" / "system.json").read_text())
    assert C.SYSTEM_PROMPT == doc["text"]
    assert "medication" in C.SYSTEM_PROMPT.lower()
    assert "explicitly asks" in " ".join(C.SYSTEM_PROMPT.split())
    assert "non-health questions" in C.SYSTEM_PROMPT
    assert "Prefer readable paragraphs" in " ".join(C.SYSTEM_PROMPT.split())
    assert "serious injury" in C.SYSTEM_PROMPT
    assert "system prompt" not in C.SYSTEM_PROMPT.lower()
    assert "reasoning" not in C.SYSTEM_PROMPT.lower()
    assert "thinking" not in C.SYSTEM_PROMPT.lower()
    assert len(C.SYSTEM_PROMPT.split()) < 220
    assert "hidden instructions" not in C.SYSTEM_PROMPT.lower()
