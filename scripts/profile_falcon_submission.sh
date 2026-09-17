#!/usr/bin/env bash
# Run the official ADTC profiler-compatible llama-bench path three times on
# the exact active Falcon GGUF, including prompt/decode speed and RSS samples.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-$ROOT/model/Falcon-H1-1.5B-Deep-JamiiAfya-Q4_K_M.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-llama-bench}"
OUT_DIR="${2:-$ROOT/experiments/falcon-submission-sft-v1/profiler}"
mkdir -p "$OUT_DIR"

[ "$(basename "$MODEL")" = "Falcon-H1-1.5B-Deep-JamiiAfya-Q4_K_M.gguf" ] || {
    echo "profiler requires the exact Q4_K_M submission filename" >&2
    exit 1
}
[ -f "$MODEL" ] || { echo "missing final GGUF: $MODEL" >&2; exit 1; }
command -v "$LLAMA_BENCH" >/dev/null 2>&1 || { echo "llama-bench not found: $LLAMA_BENCH" >&2; exit 1; }

python3 - "$MODEL" "$LLAMA_BENCH" "$OUT_DIR" <<'PY'
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError as exc:
    raise SystemExit("psutil is required to capture peak/steady RSS") from exc

model = Path(sys.argv[1]).resolve()
bench = sys.argv[2]
out = Path(sys.argv[3]).resolve()
command = [bench, "-m", str(model), "-p", "512", "-n", "128", "-ngl", "0", "--output", "json"]

def tree_rss_mb(pid):
    try:
        root = psutil.Process(pid)
        return sum(item.memory_info().rss for item in [root, *root.children(recursive=True)] if item.is_running()) / 1024**2
    except (psutil.Error, OSError):
        return 0.0

repetitions = []
for index in range(1, 4):
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples = []
    while process.poll() is None:
        samples.append(tree_rss_mb(process.pid))
        time.sleep(0.25)
    stdout, stderr = process.communicate()
    if process.returncode:
        raise SystemExit(f"llama-bench repetition {index} failed: {stderr[-2000:]}")
    rows = json.loads(stdout)
    generation = next(row for row in rows if row.get("n_gen", 0) > 0)
    prompt = next(row for row in rows if row.get("n_gen", 0) == 0 and row.get("n_prompt", 0) > 0)
    steady = samples[max(0, len(samples) // 2):] or [0.0]
    report = {
        "repetition": index,
        "command": command,
        "prompt_tok_s": float(prompt["avg_ts"]),
        "generation_tok_s": float(generation["avg_ts"]),
        "peak_rss_mb": max(samples or [0.0]),
        "steady_rss_mb": sum(steady) / len(steady),
        "rss_sample_count": len(samples),
        "elapsed_seconds": time.monotonic() - started,
        "llama_bench": rows,
    }
    (out / f"repetition-{index}.json").write_text(json.dumps(report, indent=2) + "\n")
    repetitions.append(report)

h = hashlib.sha256()
with model.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(block)
summary = {
    "schema": "falcon-official-profiler-v1",
    "model": str(model),
    "model_size_bytes": model.stat().st_size,
    "model_sha256": h.hexdigest(),
    "command": command,
    "repetitions": repetitions,
}
(out / "profiler_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({
    "model_size_bytes": summary["model_size_bytes"],
    "model_sha256": summary["model_sha256"],
    "prompt_tok_s": [round(x["prompt_tok_s"], 3) for x in repetitions],
    "generation_tok_s": [round(x["generation_tok_s"], 3) for x in repetitions],
    "peak_rss_mb": [round(x["peak_rss_mb"], 3) for x in repetitions],
    "steady_rss_mb": [round(x["steady_rss_mb"], 3) for x in repetitions],
}, indent=2))
PY
