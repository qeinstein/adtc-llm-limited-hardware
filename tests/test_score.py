"""Tests for the leaderboard score estimator (mirrors adtc-profiler formula)."""

from src.score import estimate_total, s_eff, s_perf_fixed, s_perf_relative


def test_s_perf_caps_at_15():
    assert s_perf_fixed(15.0) == 100.0
    assert s_perf_fixed(30.0) == 100.0  # capped
    assert abs(s_perf_fixed(7.5) - 50.0) < 1e-9


def test_s_perf_relative():
    assert s_perf_relative(10.0, 20.0) == 50.0
    assert s_perf_relative(25.0, 20.0) == 100.0  # capped at 1.0


def test_s_eff_rewards_low_ram():
    assert abs(s_eff(1.3) - (7 - 1.3) / 7 * 100) < 1e-6
    assert s_eff(7.0) == 0.0
    assert s_eff(8.0) == 0.0  # clamped, never negative


def test_estimate_total_math():
    # S_acc=50, tps=15 -> S_perf=100, peak=1.3GB -> S_eff~81.4
    b = estimate_total(s_acc=50.0, tps=15.0, peak_rss_gb=1.3)
    expected = 0.5 * 50 + 0.3 * 100 + 0.2 * s_eff(1.3)
    assert abs(b.s_total - round(expected, 2)) < 0.05
    assert b.s_perf == 100.0


def test_thermal_penalty_applied():
    hot = estimate_total(50.0, 15.0, 1.3, throttled=True)
    cool = estimate_total(50.0, 15.0, 1.3, throttled=False)
    assert abs((cool.s_total - hot.s_total) - 10.0) < 1e-6
