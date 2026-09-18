"""JOIN4: parser regression tests (the v1 kernel died on its smoke run
because parse_perf had no summary fallback and the resident arm printed
no PROFILE line; both fixed, pinned here)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


J = _load("join4_src", "kaggle/native-sparse-edge0join4-v1/join4_src.py")
FIX = (ROOT / "tests/fixtures/join4_smoke_stdout.txt").read_text()


def test_parse_perf_summary_only():
    # Real v1 smoke stdout: summary box only, no detailed eval lines.
    p = J.parse_perf(FIX)
    assert p["perf_source"] == "summary"
    assert abs(p["tokens_per_second"] - 6.172478) < 1e-6
    assert abs(p["ms_per_token"] - 1000.0 / 6.172478) < 1e-9
    assert abs(p["prompt_tokens_per_second"] - 9.557709) < 1e-6
    assert "eval_ms" not in p  # filled from profiler (see below)


def test_parse_perf_detailed():
    t = ("prompt eval time = 123.45 ms / 60 tokens\n"
         "eval time = 800.00 ms / 80 runs (10.00 ms per token, 100.00 tokens per second)\n"
         "[ Prompt: 9.557709 t/s | Generation: 6.172478 t/s ]\n")
    p = J.parse_perf(t)
    assert p["perf_source"] == "detailed"
    assert p["eval_ms"] == 800.0 and p["eval_runs"] == 80
    assert p["ms_per_token"] == 10.0 and p["tokens_per_second"] == 100.0
    assert p["prefill_ms"] == 123.45 and p["prefill_tokens"] == 60


def test_fill_perf_from_prof():
    p = J.parse_perf(FIX)
    prof = {"dec_graphs": 80}
    J.fill_perf_from_prof(p, prof)
    assert p["eval_runs"] == 80
    assert abs(p["eval_ms"] - p["ms_per_token"] * 80) < 1e-9
    # Detailed values are never overwritten.
    q = {"eval_ms": 1.0, "eval_runs": 2}
    J.fill_perf_from_prof(q, prof)
    assert (q["eval_ms"], q["eval_runs"]) == (1.0, 2)


def test_prefill_c_ms():
    prof = {f"pre_{k}_ns": 1_000_000 for k in
            ("attn", "gdn", "moe_rest", "expert_node", "shared",
             "lmhead", "misc")}
    assert J.prefill_c_ms(prof) == 7.0
