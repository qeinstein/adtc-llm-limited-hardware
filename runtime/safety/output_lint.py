"""Output lint for Jamii Afya: cheap textual checks that catch obvious
dangerous failures. Returns {ok, failures:[rule ids]}. Heuristic by design:
a FAIL triggers one regen with a corrective instruction, then the safe
fallback (see webapp wiring). Never sufficient alone; the system prompt,
training, and human review carry the real safety load."""
from __future__ import annotations

import re

ESC_RE = re.compile(
    r"emergen|urgent|immediately|right away|as soon as possible|hospital|"
    r"clinic|doctor|call 112|seek care|daktari|hospitali|haraka|si salama", re.I)
DIAG_RE = re.compile(
    r"\b(you have|this is definitely|it is definitely|diagnos(is|ed) (you|with)|"
    r"i am (sure|certain) (it is|you have))\b", re.I)
WAIT_HOME_RE = re.compile(
    r"\b(just |simply )?(monitor|watch|wait).{0,20}at home\b|"
    r"\bstay home and (wait|see)\b", re.I)
DOSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|mL|units?|tablets?|drops?|puffs?|spoon)\b", re.I)
WEIGHT_RE = re.compile(r"\b(weight|weighs|kg)\b", re.I)
CHILD_RE = re.compile(r"\b(child|baby|infant|toddler|newborn|month-old|year-old|mtoto|mchanga)\b", re.I)
SUPPORT_RE = re.compile(
    r"emergency|urgent|immediately|support|someone you trust|counsel|"
    r"hotline|stay (with|safe)|haraka|msaada", re.I)

SAFE_FALLBACK = (
    "I can't safely answer that from what you've told me. Because your "
    "message may describe something urgent, please seek in-person care right "
    "away — an emergency department, urgent-care clinic, or your local "
    "emergency number — rather than waiting to see how it develops. If you "
    "can, tell the clinician exactly what happened and when it started.")


def lint_output(risk: dict, prompt: str, text: str) -> dict:
    failures: list[str] = []
    text = text or ""
    prompt = prompt or ""
    if risk.get("level") == "EMERGENCY" and not ESC_RE.search(text):
        failures.append("emergency-no-escalation")
    if WAIT_HOME_RE.search(text) and risk.get("level") in ("EMERGENCY", "URGENT"):
        failures.append("wait-at-home-despite-acuity")
    if DIAG_RE.search(text):
        failures.append("confident-diagnosis")
    if "self-harm" in risk.get("hits", []) and not SUPPORT_RE.search(text):
        failures.append("selfharm-no-support")
    if CHILD_RE.search(prompt) and DOSE_RE.search(text) \
            and not WEIGHT_RE.search(prompt + " " + text):
        failures.append("peds-dose-without-weight")
    return {"ok": not failures, "failures": failures}


CORRECTIVE = (
    "Your previous answer failed a safety check ({rules}). Rewrite it now: "
    "lead with the safe action, state uncertainty plainly, do not diagnose "
    "with certainty, do not invent doses, and escalate urgent red flags to "
    "in-person care.")


def corrective_instruction(failures: list[str]) -> str:
    return CORRECTIVE.format(rules=", ".join(failures))
