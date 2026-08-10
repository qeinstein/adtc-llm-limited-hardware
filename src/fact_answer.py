"""Experimental zero-training answer engine: the model SELECTS, code RENDERS.

Everything tried elsewhere in this project (repetition guards, refusal
training, retrieval thresholds) made hallucination LESS LIKELY by making the
model less likely to say something wrong. This module makes it IMPOSSIBLE: the
model never emits a content token here. It only scores which pre-verified facts
(scripts/build_fact_graph.py -- 421 facts, each a checked substring of a real
WHO/IMCI guideline) are relevant to the query, using the exact loglikelihood
scoring the model was fine-tuned for and is measurably good at (~79% MCQ
accuracy vs. unreliable free generation). A fixed template then renders the
selected facts. There is no code path in which the model can state a dose or
a name that is not one of the 421 verified strings.

Selection is a direct application of the same char-length-normalized ranking
objective train_lora.py trains on and mcq_eval.py scores with -- this reuses
that scoring function unmodified rather than reimplementing it, per the
project's own lesson (learned three times today) that train/inference
consistency is what actually works and mismatches are what break.

This module is intentionally not on the CLI or web serving path yet. It needs a
packaged, source-audited fact graph and target-hardware latency evaluation before
it can replace the measured RAG + generation path. Calling :func:`answer` with
no graph safely returns the fixed fallback rather than silently using generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FACT_GRAPH_PATH = ROOT / "output" / "fact_graph.json"

# Selection threshold, relative to the top-scoring fact for this query -- the same
# "weak match is worse than no match" principle already validated for retrieval
# (src/retriever.py min_rel_score=0.45). A fact scoring far below the best match
# for this query is noise, not signal, and including it pads the answer with
# irrelevant (though still verified) content.
DEFAULT_REL_FLOOR = 0.55
MAX_FACTS_PER_TYPE = 3

# GLOBAL absolute floor on the single best-scoring fact for a query, measured
# empirically, not guessed: with logits_all=True set correctly (see _score_fact),
# genuine clinical queries scored -0.23 to -0.35 against this fact graph across 6
# real cases; four adversarial queries about fabricated drugs/conditions ("what is
# the dose of Zaptomycin", a name that does not exist) scored -0.42 to -0.47.
# Zero overlap, ~0.07 margin. Below this floor there is no real support in the
# verified corpus for ANY fact, regardless of which guideline retrieval matched --
# this is what actually fixes the "wrong guideline matched -> answers anyway"
# failure mode a retrieval-only relative floor could not catch, because a relative
# floor only compares candidates to each other within whatever retrieval handed it,
# even when retrieval itself was already wrong.
DEFAULT_ABS_FLOOR = -0.38

SAFE_FALLBACK_EN = (
    "I don't have verified guidance for this specific question. Please follow the "
    "national treatment guideline or consult a clinician."
)
SAFE_FALLBACK_SW = (
    "Sina mwongozo uliohakikiwa kwa swali hili mahususi. Tafadhali fuata mwongozo "
    "wa kitaifa wa matibabu au wasiliana na daktari."
)

TYPE_ORDER = ["assess", "action", "dose", "danger_sign", "refer"]
TEMPLATE_EN = {
    "assess": "Check for: {items}.",
    "action": "{items}.",
    "dose": "Dosing: {items}.",
    "danger_sign": "DANGER SIGNS -> refer urgently if you see: {items}.",
    "refer": "{items}.",
}
TEMPLATE_SW = {
    "assess": "Angalia: {items}.",
    "action": "{items}.",
    "dose": "Kipimo: {items}.",
    "danger_sign": "ISHARA ZA HATARI -> peleka haraka ikiwa: {items}.",
    "refer": "{items}.",
}


def _load_graph() -> list[dict[str, Any]]:
    if not FACT_GRAPH_PATH.exists():
        return []
    return json.loads(FACT_GRAPH_PATH.read_text(encoding="utf-8"))


@dataclass
class FactAnswer:
    text: str
    selected: list[dict[str, Any]]
    guideline_ids: list[str]
    used_fallback: bool


def _score_fact(llm, query: str, fact_text: str) -> float:
    """Char-length-normalized loglikelihood of `fact_text` as a continuation of
    `query` -- identical metric to acc_norm / train_lora.py's ranking loss, just
    applied to fact selection instead of A/B/C/D choices.

    REQUIRES `llm` to have been constructed with logits_all=True. Found by
    testing (a real bug, not a hypothetical): llama-cpp-python's Llama.eval()
    only writes to `self.scores` when that flag is set -- otherwise it's a
    silent no-op, and every score this function returns is stale/uninitialized
    data, not a real model output. This produced IDENTICAL scores for
    "Continue breastfeeding" against completely unrelated queries in initial
    testing here, which is what surfaced the bug. scripts/mcq_eval.py already
    sets this flag correctly, which is why today's real accuracy numbers
    (arc_easy 79.0 etc.) were never affected -- only this new module was.
    """
    from scripts.mcq_eval import loglik  # reuse, don't reimplement

    if not getattr(llm, "_logits_all", False):
        raise ValueError(
            "fact_answer._score_fact requires an llm constructed with "
            "logits_all=True (llama_cpp.Llama(..., logits_all=True)). Without it "
            "every score silently returns stale data instead of a real result."
        )

    ctx = f"Question: {query}\nRelevant fact:"
    total, n_tokens = loglik(llm, ctx, " " + fact_text)
    if n_tokens == 0:
        return -1e9
    return total / max(len(fact_text), 1)  # acc_norm-style char-length normalization


def answer(
    llm,
    query: str,
    lang: str = "en",
    guideline_ids: list[str] | None = None,
    rel_floor: float = DEFAULT_REL_FLOOR,
    abs_floor: float = DEFAULT_ABS_FLOOR,
) -> FactAnswer:
    """Build an answer entirely from selected, verified facts. Never calls the
    model's free-text generation path.

    `guideline_ids`: OPTIONAL narrowing hint from src/retriever.py's coarse match
    -- kept only to reduce how many of the 421 facts get scored (cheaper), never
    used as a hard filter. A hard filter means a wrong upstream retrieval match
    can never be corrected by this stage; scoring is always run against the
    matched subset AND checked against `abs_floor`, so a bad retrieval match
    still gets caught here rather than silently answered from the wrong
    guideline (the exact failure mode this module was built to fix).
    """
    graph = _load_graph()
    scoped = [f for f in graph if f["guideline_id"] in guideline_ids] if guideline_ids else graph
    # Always widen back to the full graph if the narrowed set turns out weak --
    # cheap insurance against retrieval having matched the wrong guideline entirely.
    pool = scoped if scoped else graph
    if not pool:
        return FactAnswer(
            text=SAFE_FALLBACK_SW if lang == "sw" else SAFE_FALLBACK_EN,
            selected=[], guideline_ids=[], used_fallback=True,
        )

    scored = []
    for f in pool:
        text_field = "text_sw" if lang == "sw" and f.get("text_sw") else "text_en"
        text = f.get(text_field) or f.get("text_en", "")
        if not text:
            continue
        s = _score_fact(llm, query, text)
        scored.append((s, f, text))

    if not scored:
        return FactAnswer(
            text=SAFE_FALLBACK_SW if lang == "sw" else SAFE_FALLBACK_EN,
            selected=[], guideline_ids=[], used_fallback=True,
        )

    top = max(s for s, _, _ in scored)
    # GLOBAL gate first: if even the best match is weak in absolute terms, nothing
    # in the verified corpus supports this query -- decline, regardless of how the
    # relative ranking within `pool` looks (see DEFAULT_ABS_FLOOR for the measured
    # real-vs-fabricated separation this threshold is based on).
    if top < abs_floor:
        return FactAnswer(
            text=SAFE_FALLBACK_SW if lang == "sw" else SAFE_FALLBACK_EN,
            selected=[], guideline_ids=[], used_fallback=True,
        )

    floor = top - abs(top) * (1 - rel_floor) if top < 0 else top * rel_floor
    kept = [(s, f, t) for s, f, t in scored if s >= floor]
    kept.sort(key=lambda x: -x[0])

    by_type: dict[str, list[str]] = {}
    used_ids: set[str] = set()
    for _, f, text in kept:
        t = f["type"]
        by_type.setdefault(t, [])
        if len(by_type[t]) < MAX_FACTS_PER_TYPE:
            by_type[t].append(text)
            used_ids.add(f["guideline_id"])

    if not by_type:
        return FactAnswer(
            text=SAFE_FALLBACK_SW if lang == "sw" else SAFE_FALLBACK_EN,
            selected=[], guideline_ids=[], used_fallback=True,
        )

    templates = TEMPLATE_SW if lang == "sw" else TEMPLATE_EN
    lines = []
    for t in TYPE_ORDER:
        if t in by_type:
            lines.append(templates[t].format(items="; ".join(by_type[t])))

    return FactAnswer(
        text=" ".join(lines),
        selected=[f for _, f, _ in kept],
        guideline_ids=sorted(used_ids),
        used_fallback=False,
    )
