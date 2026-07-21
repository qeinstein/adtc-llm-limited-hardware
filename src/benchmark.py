"""Honest performance benchmarking for the ADTC 2026 submission.

Two modes:

1. --profiler-parity  (the number that actually scores)
   Runs ``llama-bench -m <model> -p 512 -n 128 --output json`` — byte-for-byte the
   command the adtc-profiler uses — while sampling the process-tree RSS at 100 ms
   like the profiler's memory.py. Reports generation TPS + peak RSS + an estimated
   S_perf/S_eff/S_total. Run it against a SCALAR (no-SIMD) llama.cpp build
   (scripts/build_llamacpp_scalar.sh) to reproduce the Gate-2 audit environment and
   avoid the ±25% throughput / ±15% memory variance-fail trap.

2. (default) app benchmark
   Times the real Jamii Afya engine over clinical prompts (end-to-end product
   experience: streaming latency, decode TPS, peak RSS).

Never fabricates: if the model or llama-bench is absent it says exactly what to run
and exits. Real numbers only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from src.config import resolve_model_path
from src.score import estimate_total


# --------------------------------------------------------------------------- #
# Process-tree RSS sampler (mirrors adtc-profiler/memory.py)
# --------------------------------------------------------------------------- #
class RSSSampler(threading.Thread):
    def __init__(self, pid: int, interval: float = 0.1):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        try:
            import psutil
        except Exception:
            return
        try:
            root = psutil.Process(self.pid)
        except Exception:
            return
        while not self._stop.is_set():
            try:
                family = [root] + root.children(recursive=True)
                total = sum(p.memory_info().rss for p in family if p.is_running())
                self.peak_mb = max(self.peak_mb, total / (1024 * 1024))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> float:
        self._stop.set()
        self.join(timeout=1.0)
        return self.peak_mb


def _find_llama_bench(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    for name in ("llama-bench", "llama.cpp-llama-bench"):
        found = shutil.which(name)
        if found:
            return found
    return None


def profiler_parity(model_path: Path, llama_bench: str, n_prompt: int, n_gen: int,
                    threads: int | None, s_acc: float | None) -> int:
    cmd = [llama_bench, "-m", str(model_path), "-p", str(n_prompt),
           "-n", str(n_gen), "--output", "json"]
    if threads:
        cmd += ["-t", str(threads)]
    print(f"[parity] $ {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sampler = RSSSampler(proc.pid)
    sampler.start()
    out, err = proc.communicate()
    peak_mb = sampler.stop()

    if proc.returncode != 0:
        print(f"[parity] llama-bench failed:\n{err[-800:]}")
        return 1

    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        print(f"[parity] could not parse llama-bench JSON:\n{out[-800:]}")
        return 1

    tg = next((r for r in rows if int(r.get("n_gen", 0)) > 0), None)
    pp = next((r for r in rows if int(r.get("n_prompt", 0)) > 0), None)
    tg_tps = float(tg.get("avg_ts", 0.0)) if tg else 0.0
    pp_tps = float(pp.get("avg_ts", 0.0)) if pp else 0.0
    peak_gb = peak_mb / 1024.0

    print("\n=== Profiler-parity results (matches adtc-profiler measurement) ===")
    print(f"  Generation throughput : {tg_tps:8.2f} tok/s   (n_gen={n_gen})")
    print(f"  Prompt processing     : {pp_tps:8.2f} tok/s   (n_prompt={n_prompt})")
    print(f"  Peak process-tree RSS : {peak_mb:8.1f} MB  ({peak_gb:.3f} GB)")

    acc = s_acc if s_acc is not None else 0.0
    b = estimate_total(s_acc=acc, tps=tg_tps, peak_rss_gb=peak_gb)
    label = "" if s_acc is not None else "  (S_acc placeholder = 0; supply --s-acc to complete)"
    print(f"\n  Estimated leaderboard breakdown{label}:")
    print(b.pretty())

    out_path = Path("submission_bench.json")
    out_path.write_text(json.dumps({
        "model": model_path.name,
        "tokens_per_second_generation": round(tg_tps, 2),
        "tokens_per_second_prompt": round(pp_tps, 2),
        "peak_rss_mb": round(peak_mb, 1),
        "n_prompt": n_prompt, "n_gen": n_gen,
        "score_estimate": {"s_perf": b.s_perf, "s_eff": b.s_eff, "s_total_with_given_sacc": b.s_total},
    }, indent=2))
    print(f"\n  Wrote {out_path} (git-ignored).")
    return 0


def app_benchmark(n_gen: int) -> int:
    from src.engine import MedicalLLMEngine

    prompts = [
        "What are the danger signs of severe malaria in a child?",
        "Mtoto ana kuharisha maji maji na anaonekana amechoka. Nifanye nini?",
        "How do I manage a snakebite before referral in a rural clinic?",
    ]
    engine = MedicalLLMEngine()
    print("\n=== App engine benchmark (interactive product) ===")
    tps_all = []
    for p in prompts:
        r = engine.generate(p, max_tokens=n_gen)
        t = r["telemetry"]
        tps_all.append(t["throughput_tps"])
        print(f"  {t['throughput_tps']:6.2f} tok/s | {t['completion_tokens']:3d} tok | "
              f"{t['elapsed_sec']:5.2f}s | peak {t['peak_rss_mb']:.0f} MB | {p[:40]}")
    if tps_all:
        print(f"\n  Mean decode: {sum(tps_all) / len(tps_all):.2f} tok/s")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ADTC benchmark (honest; profiler-parity)")
    ap.add_argument("--profiler-parity", action="store_true",
                    help="Run llama-bench exactly like the profiler")
    ap.add_argument("--llama-bench", type=str, default=None, help="Path to llama-bench binary")
    ap.add_argument("--n-prompt", type=int, default=512)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--s-acc", type=float, default=None, help="Known S_acc to complete the estimate")
    args = ap.parse_args(argv)

    model_path = resolve_model_path()
    if not model_path.exists():
        print(f"[bench] Model not found at {model_path}. Run ./download_model.sh first.")
        return 2

    if args.profiler_parity:
        lb = _find_llama_bench(args.llama_bench)
        if not lb:
            print("[bench] llama-bench not found. Build a SCALAR (audit-parity) build:")
            print("        bash scripts/build_llamacpp_scalar.sh")
            print("        then re-run with --llama-bench ./llama.cpp/build/bin/llama-bench")
            return 2
        return profiler_parity(model_path, lb, args.n_prompt, args.n_gen, args.threads, args.s_acc)

    try:
        return app_benchmark(args.n_gen)
    except Exception as e:
        print(f"[bench] engine benchmark unavailable ({e}). Is llama-cpp-python installed?")
        return 2


if __name__ == "__main__":
    sys.exit(main())
