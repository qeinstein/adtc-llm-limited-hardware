"""ADTC 2026 leaderboard score estimation.

Mirrors the formula documented in the adtc-profiler README so our own harnesses
can print an estimated S_total from measured (TPS, peak RAM) and an accuracy
number. This is an ESTIMATE — the official score is computed by the organizers on
their standard VM (scalar llama.cpp build). See REPORT.md.

    S_total = 0.50*S_acc + 0.30*S_perf + 0.20*S_eff - P_thermal
    S_perf  = min(TPS / 15.0, 1.0) * 100        # profiler code (fixed reference)
    S_eff   = max(0, (7.0 - peak_rss_gb) / 7.0) * 100
    P_thermal = 10 if throttled or core_temp > 85C

Note: the Foundation site describes S_perf as "relative to the fastest submission"
while the profiler code caps at a fixed 15 tps. We expose both so strategy is
robust to either interpretation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

TPS_REFERENCE = 15.0
RAM_LIMIT_GB = 7.0
THERMAL_PENALTY = 10.0


def s_perf_fixed(tps: float) -> float:
    return min(tps / TPS_REFERENCE, 1.0) * 100.0


def s_perf_relative(tps: float, tps_max: float) -> float:
    if tps_max <= 0:
        return 0.0
    return min(tps / tps_max, 1.0) * 100.0


def s_eff(peak_rss_gb: float) -> float:
    return max(0.0, (RAM_LIMIT_GB - peak_rss_gb) / RAM_LIMIT_GB) * 100.0


@dataclass
class ScoreBreakdown:
    s_acc: float
    s_perf: float
    s_eff: float
    p_thermal: float
    s_total: float

    def pretty(self) -> str:
        return (
            f"  S_acc   (50%): {self.s_acc:6.2f}  -> {0.5 * self.s_acc:6.2f}\n"
            f"  S_perf  (30%): {self.s_perf:6.2f}  -> {0.3 * self.s_perf:6.2f}\n"
            f"  S_eff   (20%): {self.s_eff:6.2f}  -> {0.2 * self.s_eff:6.2f}\n"
            f"  P_thermal    : {-self.p_thermal:6.2f}\n"
            f"  {'-' * 30}\n"
            f"  S_total      : {self.s_total:6.2f}"
        )


def estimate_total(
    s_acc: float,
    tps: float,
    peak_rss_gb: float,
    throttled: bool = False,
    tps_max: float | None = None,
) -> ScoreBreakdown:
    """Estimate the leaderboard score. Pass ``tps_max`` to use relative S_perf."""
    sp = s_perf_relative(tps, tps_max) if tps_max else s_perf_fixed(tps)
    se = s_eff(peak_rss_gb)
    pt = THERMAL_PENALTY if throttled else 0.0
    total = 0.5 * s_acc + 0.3 * sp + 0.2 * se - pt
    return ScoreBreakdown(round(s_acc, 2), round(sp, 2), round(se, 2), round(pt, 2), round(total, 2))


def as_dict(b: ScoreBreakdown) -> dict[str, float]:
    return asdict(b)
