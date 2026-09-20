"""JOIN4 patch-anchor integrity: the bench lazy-env compat anchor must match
the frozen pin's parser text exactly once (replace_once would raise
otherwise, failing the runtime build; this test fails faster, offline)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_apply():
    spec = importlib.util.spec_from_file_location(
        "join4_apply", ROOT / "probes/edge0_port/join4_apply.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["join4_apply"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bench_lazy_env_anchor_matches_pin_once():
    a = _load_apply()
    fixture = (ROOT / "tests/fixtures/pin_bench_lazy_defaults.txt").read_text()
    assert fixture.count(a.BENCH_LAZY_ENV_ANCHOR) == 1


def test_bench_lazy_env_patch_maps_all_modes():
    a = _load_apply()
    p = a.BENCH_LAZY_ENV_PATCH
    assert 'getenv("LLAMA_ARG_LAZY_MODE")' in p
    for mode in ("LLAMA_LAZY_MODE_ON", "LLAMA_LAZY_MODE_AUTO",
                 "LLAMA_LAZY_MODE_OFF"):
        assert mode in p
    assert "cmd_params_defaults.lazy_mode" in p  # default preserved
