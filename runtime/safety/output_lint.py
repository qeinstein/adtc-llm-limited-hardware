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
AUTHORITY_RE = re.compile(
    r"\b(WHO|IMCI|NCDC|UNICEF)\b.{0,30}"
    r"(recommends?|protocol|says|requires?|advises?|guideline|states that)\b", re.I)
AGE_RE = re.compile(
    r"\b\d+\s*(years?|yrs?|months?|mos?|weeks?|days?)[-\s]?old\b|\bage\b.{0,10}\d+", re.I)
# Hard gates: a recipe/instruction with NO negation anywhere in the text.
# Safe answers to these prompts always contain a negation ("never", "do not").
_NEG_ANYWHERE = re.compile(r"\b(do not|don't|dont|never|not|no\b|avoid|stop|refuse)\b", re.I)
BLEACH_RECIPE_RE = re.compile(
    r"(bleach|jik).{0,60}1\s*:\s*\d+|"
    r"\b(tea|table)?spoons?\b.{0,30}(bleach|jik).{0,30}(drink|mix|add|water|swallow)|"
    r"\b(cup|cap|liter|litre|drops?|ml)\b.{0,30}(bleach|jik).{0,30}(drink|mix|add|water|swallow)|"
    r"bleach.{0,30}dilut.{0,30}(drink|mouth|glass|swallow)|"
    r"dilut.{0,30}bleach.{0,30}(drink|mouth|glass|swallow)", re.I)
AMPUTATE_RE = re.compile(
    r"(amputat.{0,50}(how|steps?|first|then|knife|saw|blade|tourniquet|tie|cut)|"
    r"cut (it|the \w+|off).{0,40}(knife|saw|blade|tourniquet|first|then))", re.I)
CITE_RE = re.compile(
    r"\bet al\.\b|doi\.org/10\.|\(doi:|guideline vol\.|pp\.\s*\d+\s*[-–]\s*\d+", re.I)

SAFE_FALLBACK = (
    "I can't safely answer that from what you've told me. Because your "
    "message may describe something urgent, please seek in-person care right "
    "away — an emergency department, urgent-care clinic, or your local "
    "emergency number — rather than waiting to see how it develops. If you "
    "can, tell the clinician exactly what happened and when it started.")


def lint_output(risk: dict, prompt: str, text: str,
                attributions: list[str] | None = None,
                thinking: str = "") -> dict:
    from .repetition import check_repetition

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
    facts = set(risk.get("facts", ()))
    if DOSE_RE.search(text) and not WEIGHT_RE.search(prompt + " " + text) \
            and (facts & {"child_under5", "infant_under2mo", "neonate",
                          "pregnant", "dosing_request"}):
        failures.append("personalized-dose-missing-variables")
    if facts & {"child_under5", "infant_under2mo", "neonate"} \
            and DOSE_RE.search(text) and not AGE_RE.search(prompt):
        failures.append("peds-dose-without-age")
    if BLEACH_RECIPE_RE.search(text) and not _NEG_ANYWHERE.search(text):
        failures.append("bleach-recipe")
    if AMPUTATE_RE.search(text) and not _NEG_ANYWHERE.search(text):
        failures.append("diy-amputation")
    if CITE_RE.search(text):
        failures.append("fabricated-citation")
    if attributions is not None and AUTHORITY_RE.search(text):
        allowed = " ".join(attributions).lower()
        claims = {m.group(1).upper() for m in AUTHORITY_RE.finditer(text)}
        if not claims or not all(
                c.lower() in allowed or
                {"WHO": "world health", "IMCI": "childhood illness",
                 "NCDC": "ncdc", "UNICEF": "unicef"}[c] in allowed
                for c in claims):
            failures.append("unsupported-authority-claim")
    rep = check_repetition(text, thinking)
    failures.extend(rep["failures"])
    return {"ok": not failures, "failures": failures}


CORRECTIVE = (
    "Your previous answer failed a safety check ({rules}). Rewrite it now: "
    "lead with the safe action, state uncertainty plainly, do not diagnose "
    "with certainty, do not invent doses, and escalate urgent red flags to "
    "in-person care.")


def corrective_instruction(failures: list[str]) -> str:
    return CORRECTIVE.format(rules=", ".join(failures))
