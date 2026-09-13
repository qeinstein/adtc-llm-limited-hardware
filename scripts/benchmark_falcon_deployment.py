#!/usr/bin/env python3
"""Run repeated profiler-shaped Falcon GGUF deployment measurements.

The script intentionally measures every repetition instead of keeping the best
one.  It samples the complete process tree while llama-bench runs, captures
the exact scalar command, records page faults from ``/usr/bin/time -v``, and
checks deterministic CLI output hashes separately from throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_rss_mb(pid: int) -> float:
    try:
        import psutil

        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        return sum(process.memory_info().rss for process in processes if process.is_running()) / 1024**2
    except (ImportError, OSError, RuntimeError):
        return 0.0


def parse_time_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    values: dict[str, Any] = {}
    patterns = {
        "max_rss_kb": r"Maximum resident set size \(kbytes\):\s*(\d+)",
        "minor_page_faults": r"Minor \(reclaiming a frame\) page faults:\s*(\d+)",
        "major_page_faults": r"Major \(page faults\):\s*(\d+)",
        "elapsed": r"Elapsed \(wall clock\) time \(seconds\):\s*([^\n]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            values[key] = int(match.group(1)) if key.endswith(("kb", "faults")) else match.group(1).strip()
    return values


def bench_once(command: list[str], time_report: Path, sample_seconds: float) -> dict[str, Any]:
    time_report.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Kaggle images do not consistently ship the external /usr/bin/time
    # utility.  Keep this measurement self-contained: resource.getrusage gives
    # child faults/RSS and the sampler below gives the complete process-tree
    # peak, which is the deployment metric we actually need.
    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    samples: list[float] = []
    next_heartbeat = started + 30.0
    while process.poll() is None:
        samples.append(tree_rss_mb(process.pid))
        if time.monotonic() >= next_heartbeat:
            print(json.dumps({
                "timestamp_utc": stamp(),
                "event": "benchmark_heartbeat",
                "pid": process.pid,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "sampled_peak_rss_mb": round(max(samples or [0.0]), 1),
            }), flush=True)
            next_heartbeat += 30.0
        time.sleep(sample_seconds)
    stdout, stderr = process.communicate()
    samples.append(tree_rss_mb(process.pid))
    elapsed = time.monotonic() - started
    if process.returncode:
        raise RuntimeError(f"llama-bench failed ({process.returncode}): {stderr[-2000:]}\n{stdout[-1000:]}")
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"llama-bench did not emit JSON: {stdout[-2000:]}") from exc
    generation = next(row for row in rows if row.get("n_gen", 0) > 0)
    prompt = next(row for row in rows if row.get("n_gen", 0) == 0 and row.get("n_prompt", 0) > 0)
    steady_samples = samples[max(0, len(samples) // 2):]
    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report_text = (
        f"Maximum resident set size (kbytes): {int(after_usage.ru_maxrss)}\n"
        f"Minor (reclaiming a frame) page faults: {max(0, int(after_usage.ru_minflt - before_usage.ru_minflt))}\n"
        f"Major (page faults): {max(0, int(after_usage.ru_majflt - before_usage.ru_majflt))}\n"
        f"Elapsed (wall clock) time (seconds): {elapsed:.3f}\n"
    )
    time_report.write_text(report_text, encoding="utf-8")
    report = parse_time_report(time_report)
    return {
        "started_utc": stamp(),
        "elapsed_seconds_wall": round(elapsed, 3),
        "prompt_tps": float(prompt["avg_ts"]),
        "decode_tps": float(generation["avg_ts"]),
        "decode_ms_per_token": round(1000 / max(float(generation["avg_ts"]), 1e-12), 3),
        "peak_tree_rss_mb_sampled": round(max(samples or [0.0]), 3),
        "steady_tree_rss_mb_mean_last_half": round(statistics.mean(steady_samples or [0.0]), 3),
        "rss_sample_count": len(samples),
        "returncode": process.returncode,
        "time_report": report,
        "command": command,
    }


def deterministic_hashes(cli: str, model: Path, threads: int, repetitions: int, output_dir: Path) -> dict[str, Any]:
    prompt = "A child has chest indrawing and fast breathing. State the immediate disposition in one sentence."
    command = [cli, "-m", str(model), "-ngl", "0", "-t", str(threads), "-c", "2048", "-p", prompt, "-n", "32", "--temp", "0", "--seed", "42", "--single-turn", "--no-display-prompt"]
    hashes: list[str] = []
    outputs: list[str] = []
    for index in range(repetitions):
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode:
            raise RuntimeError(f"llama-cli failed ({result.returncode}): {result.stderr[-2000:]}")
        raw = result.stdout
        hashes.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
        outputs.append(raw)
        (output_dir / f"deterministic-output-{index + 1}.txt").write_text(raw, encoding="utf-8")
    return {"command": command, "hashes": hashes, "deterministic": len(set(hashes)) == 1, "output_lengths": [len(value) for value in outputs]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--llama-bench", required=True)
    ap.add_argument("--llama-cli", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--sample-seconds", type=float, default=0.25)
    args = ap.parse_args(argv)
    if args.repetitions < 3:
        raise ValueError("deployment baseline requires at least three repetitions")
    model = args.model.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bench_command = [str(Path(args.llama_bench).resolve()), "-m", str(model), "-p", "512", "-n", "128", "-ngl", "0", "-t", str(args.threads), "--output", "json"]
    repetitions = []
    for index in range(args.repetitions):
        print(json.dumps({"timestamp_utc": stamp(), "event": "benchmark_start", "tag": args.tag, "repetition": index + 1, "command": bench_command}), flush=True)
        repetitions.append(bench_once(bench_command, out / f"time-{index + 1}.log", args.sample_seconds))
        print(json.dumps({"timestamp_utc": stamp(), "event": "benchmark_complete", "tag": args.tag, "repetition": index + 1, "decode_tps": repetitions[-1]["decode_tps"], "peak_rss_mb": repetitions[-1]["peak_tree_rss_mb_sampled"]}), flush=True)
    summary = {
        "schema_version": "1.0.0",
        "experiment_id": "falcon-deployment-baseline",
        "tag": args.tag,
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "model_sha256": sha256(model),
        "threads": args.threads,
        "repetitions": args.repetitions,
        "profiler_shaped_command": bench_command,
        "repetitions_detail": repetitions,
        "aggregate": {
            key: {"mean": round(statistics.mean(values), 4), "stdev": round(statistics.stdev(values), 4) if len(values) >= 2 else 0.0, "min": round(min(values), 4), "max": round(max(values), 4)}
            for key, values in {
                "prompt_tps": [item["prompt_tps"] for item in repetitions],
                "decode_tps": [item["decode_tps"] for item in repetitions],
                "decode_ms_per_token": [item["decode_ms_per_token"] for item in repetitions],
                "peak_tree_rss_mb_sampled": [item["peak_tree_rss_mb_sampled"] for item in repetitions],
                "steady_tree_rss_mb_mean_last_half": [item["steady_tree_rss_mb_mean_last_half"] for item in repetitions],
                "wall_seconds": [item["elapsed_seconds_wall"] for item in repetitions],
            }.items()
        },
        "deterministic_generation": deterministic_hashes(str(Path(args.llama_cli).resolve()), model, args.threads, args.repetitions, out),
        "completed_utc": stamp(),
    }
    (out / "deployment_baseline.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"timestamp_utc": stamp(), "event": "deployment_baseline_complete", "tag": args.tag, "model_bytes": summary["model_bytes"], "model_sha256": summary["model_sha256"], "decode_tps_mean": summary["aggregate"]["decode_tps"]["mean"], "decode_tps_stdev": summary["aggregate"]["decode_tps"]["stdev"], "peak_rss_mean_mb": summary["aggregate"]["peak_tree_rss_mb_sampled"]["mean"], "deterministic": summary["deterministic_generation"]["deterministic"]}, ensure_ascii=False), flush=True)
    return 0 if summary["deterministic_generation"]["deterministic"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
