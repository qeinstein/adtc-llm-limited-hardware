"""Validate the real metadata.json against the adtc-profiler schema constraints."""

import copy

from src.config import load_metadata, resolve_model_path
from src.manifest import validate_metadata


def test_real_metadata_is_valid():
    errs = validate_metadata(load_metadata())
    assert errs == [], f"metadata.json schema errors: {errs}"


def test_model_path_points_into_repo_model_dir():
    p = resolve_model_path()
    assert p.name.endswith(".gguf")
    assert p.parent.name == "model"


def test_domain_is_healthcare():
    assert load_metadata()["domain"] == "healthcare_medical"


def test_exactly_two_test_prompts():
    assert len(load_metadata()["test_prompts"]) == 2


def test_validator_catches_common_mistakes():
    meta = load_metadata()

    bad = copy.deepcopy(meta)
    bad["domain"] = "not_a_domain"
    assert any("domain" in e for e in validate_metadata(bad))

    bad = copy.deepcopy(meta)
    bad["test_prompts"] = bad["test_prompts"][:1]
    assert any("test_prompts" in e for e in validate_metadata(bad))

    bad = copy.deepcopy(meta)
    bad["model"]["runtime"] = "vllm"
    assert any("runtime" in e for e in validate_metadata(bad))

    bad = copy.deepcopy(meta)
    bad["surprise"] = 1
    assert any("unexpected top-level key" in e for e in validate_metadata(bad))
