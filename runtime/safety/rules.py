"""Deterministic guidance rules engine. Loads guidance/domains/*.yaml,
matches trigger_facts conjunctions (+ trigger_any_of disjunctions) against
extracted facts, and returns the merged safety floor:

  {"risk": level, "cards": [ids], "required": [...], "prohibited": [...],
   "missing": [...], "interim": [...], "provenance": [...], "override": bool}

override=True when a priority-1 emergency card matches: the UI must show
the safety guidance prominently (never hidden in reasoning).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GUIDE = ROOT / "guidance"

ORDER = {"low": 0, "routine": 1, "urgent": 2, "emergency": 3}


@lru_cache(maxsize=1)
def load_cards() -> list[dict]:
    import yaml

    man = json.loads((GUIDE / "manifest.json").read_text())
    cards = []
    for domain, rel in sorted(man["domains"].items()):
        for c in yaml.safe_load((GUIDE / rel).read_text()):
            assert c["domain"] == domain, (c["id"], domain)
            cards.append(c)
    return cards


def match_cards(facts: set[str]) -> list[dict]:
    out = []
    for c in load_cards():
        if not set(c["trigger_facts"]) <= facts:
            continue
        any_of = c.get("trigger_any_of") or []
        if any_of and not (set(any_of) & facts):
            continue
        out.append(c)
    out.sort(key=lambda c: (c["priority"], c["id"]))
    return out


def evaluate(facts: set[str]) -> dict:
    matched = match_cards(facts)
    risk = "low"
    required, prohibited, missing, interim = [], [], [], []
    prov = []
    for c in matched:
        if ORDER[c["risk_level"]] > ORDER[risk]:
            risk = c["risk_level"]
        required += [f"[{c['id']}] {a}" for a in c["required_actions"]]
        prohibited += [f"[{c['id']}] {a}" for a in c["prohibited_actions"]]
        missing += [f"[{c['id']}] {m}"
                    for m in c["required_missing_information"]]
        interim += [f"[{c['id']}] {a}" for a in c["safe_interim_actions"]]
        s = c["source"]
        prov.append({"card": c["id"], "authority": s["authority"],
                     "title": s["title"], "url": s["canonical_url"]})
    override = any(c["priority"] == 1 and c["risk_level"] == "emergency"
                   for c in matched)
    return {"risk": risk, "cards": [c["id"] for c in matched],
            "required": required, "prohibited": prohibited,
            "missing": missing, "interim": interim, "provenance": prov,
            "override": override}


def retrieval_payload(matched_ids: list[str], max_cards: int = 3) -> list[dict]:
    """Small structured payload for the model: top cards by priority."""
    by_id = {c["id"]: c for c in load_cards()}
    cards = sorted((by_id[i] for i in matched_ids if i in by_id),
                   key=lambda c: (c["priority"], c["id"]))[:max_cards]
    return [{"id": c["id"], "risk": c["risk_level"], "scope": c["scope"],
             "required": c["required_actions"],
             "prohibited": c["prohibited_actions"],
             "missing_info": c["required_missing_information"],
             "attribution": f"{c['source']['authority']}: {c['source']['title']}"}
            for c in cards]
