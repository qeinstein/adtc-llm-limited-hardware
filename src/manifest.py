"""Self-validation of metadata.json against the adtc-profiler schema.

Mirrors the constraints in adtc-profiler's ``schema/adtc-profiler.schema.json``
(the ``submission`` block) so we catch a malformed manifest locally, before the
profiler rejects it. ``additionalProperties`` is strict in the real schema; we
also flag unknown top-level keys (except ``_runtime``, which the profiler strips).
"""

from __future__ import annotations

from typing import Any

DOMAINS = {
    "math_scientific_reasoning",
    "healthcare_medical",
    "agriculture",
    "creative_writing",
    "coding_assistants",
    "corporate_enterprise",
    "autonomous_ai_agents",
}
PACKAGING = {"docker_image", "docker_build_from_repo", "binary_bundle"}
ALLOWED_TOP_LEVEL = {
    "team_id",
    "domain",
    "language_scope",
    "african_alpha_claim",
    "budget_laptop_claim",
    "submitter",
    "cross_disciplinary_pairing",
    "test_prompts",
    "model",
    "_runtime",  # stripped by profiler (leading underscore), allowed here
}


def _req_str(obj: dict[str, Any], key: str, errs: list[str], min_len: int = 1) -> None:
    v = obj.get(key)
    if not isinstance(v, str) or len(v) < min_len:
        errs.append(f"'{key}' must be a string of length >= {min_len}")


def validate_metadata(meta: dict[str, Any]) -> list[str]:
    """Return a list of human-readable errors ([] means valid)."""
    errs: list[str] = []

    for key in meta:
        if key not in ALLOWED_TOP_LEVEL:
            errs.append(f"unexpected top-level key '{key}' (schema is strict)")

    _req_str(meta, "team_id", errs)

    if meta.get("domain") not in DOMAINS:
        errs.append(f"'domain' must be one of {sorted(DOMAINS)}")

    langs = meta.get("language_scope")
    if not isinstance(langs, list) or not langs:
        errs.append("'language_scope' must be a non-empty array")
    elif any(not isinstance(x, str) or len(x) < 2 for x in langs):
        errs.append("each 'language_scope' item must be a string of length >= 2")

    for key in ("african_alpha_claim", "budget_laptop_claim"):
        if not isinstance(meta.get(key), bool):
            errs.append(f"'{key}' must be a boolean")

    sub = meta.get("submitter")
    if not isinstance(sub, dict):
        errs.append("'submitter' must be an object")
    else:
        _req_str(sub, "name", errs)
        _req_str(sub, "email", errs, min_len=3)
        _req_str(sub, "github_handle", errs)

    pairing = meta.get("cross_disciplinary_pairing")
    if not isinstance(pairing, dict):
        errs.append("'cross_disciplinary_pairing' must be an object")
    else:
        _req_str(pairing, "discipline", errs)
        _req_str(pairing, "description", errs)
        if not isinstance(pairing.get("load_bearing"), bool):
            errs.append("'cross_disciplinary_pairing.load_bearing' must be a boolean")

    prompts = meta.get("test_prompts")
    if not isinstance(prompts, list) or len(prompts) != 2:
        errs.append("'test_prompts' must be an array of EXACTLY 2 items")
    else:
        for i, p in enumerate(prompts):
            if not isinstance(p, dict):
                errs.append(f"test_prompts[{i}] must be an object")
                continue
            _req_str(p, "prompt_id", errs)
            _req_str(p, "prompt", errs)

    model = meta.get("model")
    if not isinstance(model, dict):
        errs.append("'model' must be an object")
    else:
        for key in ("name", "runtime", "quantization", "parameters_estimate"):
            _req_str(model, key, errs)
        if model.get("packaging") not in PACKAGING:
            errs.append(f"'model.packaging' must be one of {sorted(PACKAGING)}")
        if model.get("runtime") != "llama.cpp":
            errs.append("'model.runtime' must be 'llama.cpp' (only accepted runtime)")

    return errs
