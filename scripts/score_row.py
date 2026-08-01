#!/usr/bin/env python3
"""Print `S_perf S_eff S_total` for one measured (tps, peak_rss_mb, accuracy) row.

Mirrors the adtc-profiler README formula (same math as src/score.py), used by the
quant-sweep workflow to rank candidates. Reads from env so the caller doesn't have
to worry about shell quoting:

    TG=15.6 PEAK=598 ACC=55.0 ACCN=58.5 python scripts/score_row.py
    -> "100.0 91.7 88.4"

Note: uses measured MCQ accuracy as a stand-in for S_acc. The real S_acc is largely
judge-assessed, so the S_total here is a RELATIVE ranking signal between candidates,
not an absolute score prediction.
"""

from __future__ import annotations

import os


def _f(name: str) -> float:
    try:
        return float(os.environ.get(name, "") or 0)
    except ValueError:
        return 0.0


def main() -> None:
    tps = _f("TG")
    peak_gb = _f("PEAK") / 1024.0
    acc = _f("ACCN") or _f("ACC")  # prefer acc_norm, as the profiler does

    s_perf = min(tps / 15.0, 1.0) * 100.0
    s_eff = max(0.0, (7.0 - peak_gb) / 7.0) * 100.0
    s_total = 0.5 * acc + 0.3 * s_perf + 0.2 * s_eff

    print(f"{s_perf:.1f} {s_eff:.1f} {s_total:.1f}")


if __name__ == "__main__":
    main()
