"""Input risk classification for Jamii Afya. Pure rule matching over the
YAML rule files (no LLM call, microseconds). Levels: LOW, ROUTINE, URGENT,
EMERGENCY, MEDICATION_HIGH_RISK. Never logs prompt text (see log_event)."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

DIR = Path(__file__).resolve().parent

URGENT_RE = re.compile(
    r"\b(high fever|fever \d|vomiting blood|coughing blood|severe pain|"
    r"cannot (urinate|swallow)|swelling.{0,10}(rapid|face)|rash.{0,10}fever|"
    r"homa kali|maumivu makali)\b", re.I)


@lru_cache(maxsize=1)
def _load_rules() -> tuple[list[dict], list[dict]]:
    import yaml

    em = yaml.safe_load((DIR / "emergency_rules.yaml").read_text())["rules"]
    md = yaml.safe_load((DIR / "medication_rules.yaml").read_text())["rules"]
    out = []
    for group in (em, md):
        compiled = []
        for rule in group:
            compiled.append({"id": rule["id"], "category": rule["category"],
                             "res": [re.compile(p, re.I) for p in rule["patterns"]]})
        out.append(compiled)
    return out[0], out[1]


def classify_input(text: str) -> dict:
    """Return {level, hits:[rule ids], categories:[...]}. No PHI retained."""
    em_rules, med_rules = _load_rules()
    text = text or ""
    em_hits = [r["id"] for r in em_rules if any(rx.search(text) for rx in r["res"])]
    med_hits = [r["id"] for r in med_rules if any(rx.search(text) for rx in r["res"])]
    if em_hits:
        level = "EMERGENCY"
    elif med_hits:
        level = "MEDICATION_HIGH_RISK"
    elif URGENT_RE.search(text):
        level = "URGENT"
    elif len(text.split()) <= 2:
        level = "LOW"
    else:
        level = "ROUTINE"
    return {"level": level, "hits": em_hits + med_hits,
            "emergency": bool(em_hits), "meds": bool(med_hits)}


ESCALATION_REMINDER = (
    "The user's message contains potential emergency red flags. Begin with "
    "the urgent action (seek immediate in-person emergency care) before any "
    "explanation, and never advise monitoring at home."
)

MEDS_REMINDER = (
    "Medication-safety question. State what patient factors dosing depends "
    "on (age, weight, pregnancy, kidney/liver, other medicines) rather than "
    "guessing; never fabricate a dose; advise pharmacist/clinician review."
)


def reminder_for(risk: dict) -> str:
    if risk["level"] == "EMERGENCY":
        return ESCALATION_REMINDER
    if risk["level"] == "MEDICATION_HIGH_RISK":
        return MEDS_REMINDER
    return ""


def log_event(risk: dict, extra: dict | None = None) -> dict:
    """Guard audit record. Contains rule ids + level ONLY — never prompt,
    response, or health content."""
    return {"guard": "jamii-safety-v1", "level": risk["level"],
            "hits": list(risk["hits"]), **(extra or {})}
