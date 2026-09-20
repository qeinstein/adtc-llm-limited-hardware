"""JOIN4: parser regression tests (the v1 kernel died on its smoke run
because parse_perf had no summary fallback and the resident arm printed
no PROFILE line; both fixed, pinned here. The join4b v1 kernel died on
its nodelist run because a prefill-only summary prints Generation 0.00
and parse_perf divided by it; guarded, pinned here)."""
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
JB = _load("join4b_src", "kaggle/native-sparse-edge0join4b-v1/join4b_src.py")
FIX = (ROOT / "tests/fixtures/join4_smoke_stdout.txt").read_text()
FIXB_NL = (ROOT / "tests/fixtures/join4b_nodelist_stdout.txt").read_text()


def _load_sched():
    return _load("sched_src", "kaggle/native-sparse-edge0sched-v1/sched_src.py")


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


def test_parse_perf_prefill_only_no_crash():
    # Real join4b v1 nodelist stdout: prefill-only run prints
    # Generation 0.00 (dec_graphs=0); must not divide by zero.
    import math
    p = JB.parse_perf(FIXB_NL)
    assert p["perf_source"] == "summary"
    assert p["tokens_per_second"] == 0.0
    assert math.isinf(p["ms_per_token"])
    assert abs(p["prompt_tokens_per_second"] - 9.119650) < 1e-6


def test_fill_perf_from_prof_zero_graphs():
    # Prefill-only profile: eval wall is exactly 0, never NaN.
    import math
    p = JB.parse_perf(FIXB_NL)
    JB.fill_perf_from_prof(p, {"dec_graphs": 0})
    assert p["eval_ms"] == 0.0 and p["eval_runs"] == 0
    assert not math.isnan(p["eval_ms"])


def test_sched_src_parity():
    # P0 sched kernel copies the JOIN4b parsers + fixes: the copied
    # guards must behave identically on the real failed stdout, and
    # run_case must expose the scheduler flags.
    import inspect
    import math
    S = _load_sched()
    p = S.parse_perf(FIXB_NL)
    assert p["tokens_per_second"] == 0.0 and math.isinf(p["ms_per_token"])
    S.fill_perf_from_prof(p, {"dec_graphs": 0})
    assert p["eval_ms"] == 0.0 and p["eval_runs"] == 0
    sig = inspect.signature(S.run_case)
    assert sig.parameters["poll"].default == 0
    assert sig.parameters["threads"].default == 4
    assert "taskset" in sig.parameters
    assert S.EXPECTED_PID21_N80_SHA == "099f8728ef2db3b494f8a69ca9ea612331ca47342abc0e99a33e831041b7ce7e"
