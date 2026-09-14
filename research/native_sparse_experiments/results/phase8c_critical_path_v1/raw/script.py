"""Phase 8 critical-path measurement for the exact bounded executor.

Reuse the committed v10 implementation verbatim, then compare cold and
page-cache-warm executions.  The warm arm is a measurement-only I/O oracle:
it preserves exact model math and process RSS, but intentionally relies on
host page cache and is not a deployable <= 4 GiB system point.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-critical-path-v1")
OUT = WORK / "native-sparse-critical-path-v1-results"
BASE = SCRATCH / "bounded_executor_v10.py"
RESEARCH_COMMIT = "cae43e5fb930ec562bf5749c7df5cb0d8827a911"
BASE_URL = (
    "https://raw.githubusercontent.com/qeinstein/adtc-llm-limited-hardware/"
    f"{RESEARCH_COMMIT}/research/native_sparse_experiments/results/"
    "phase7b_staged_pipeline_v1/raw/script.py"
)
N_REPS = 3

SCRATCH.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def fetch_base() -> str:
    urllib.request.urlretrieve(BASE_URL, BASE)
    digest = hashlib.sha256(BASE.read_bytes()).hexdigest()
    print(f"v10 source sha256={digest}", flush=True)
    return digest


def load_base():
    spec = importlib.util.spec_from_file_location("bounded_v10", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load v10 module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def means(rows: list[dict]) -> dict:
    return {
        "process_elapsed_sec_mean": statistics.mean(x["elapsed_sec"] for x in rows),
        "decode_tok_s_mean": statistics.mean(x["decode_perf"]["tokens_per_second"] for x in rows),
        "decode_ms_per_token_mean": statistics.mean(x["decode_perf"]["ms_per_token"] for x in rows),
        "prompt_tok_s_mean": statistics.mean(x["decode_perf"]["prompt_tokens_per_second"] for x in rows),
        "physical_read_bytes_mean": statistics.mean(x["read_bytes_max"] for x in rows),
        "logical_rchar_bytes_mean": statistics.mean(x["rchar_max"] for x in rows),
        "peak_rss_mib_max": max(x["peak_rss_mib"] for x in rows),
    }


def main() -> None:
    started = time.time()
    base_sha = fetch_base()
    base = load_base()
    base.SCRATCH = SCRATCH
    base.OUT = OUT
    base.LLAMA = SCRATCH / "llama.cpp"
    base.BUILD = base.LLAMA / "build-native"
    base.CLI = base.BUILD / "bin" / "llama-cli"
    base.MODEL = SCRATCH / base.MODEL_FILE
    base.N_REPS = N_REPS

    original_clone = base.clone_and_patch

    def clone_patch_precise() -> dict:
        runtime = original_clone()
        cli_context = base.LLAMA / "tools/cli/cli-context.cpp"
        base.replace_once(
            cli_context,
            "[ Prompt: %.1f t/s | Generation: %.1f t/s ]",
            "[ Prompt: %.6f t/s | Generation: %.6f t/s ]",
        )
        runtime["precise_cli_timings"] = True
        return runtime

    base.clone_and_patch = clone_patch_precise
    runtime = base.clone_and_patch()
    base.build()
    model = base.fetch_model()
    hardware = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "threads_used": base.N_THREADS,
        "lscpu": subprocess.run(["lscpu"], text=True, capture_output=True).stdout,
    }

    original_drop = base.drop_file_cache
    cache_mode = "cold"

    def controlled_drop(path: Path) -> dict:
        if cache_mode == "cold":
            result = original_drop(path)
            result["mode"] = "cold_fadvise_dontneed"
            return result
        return {"available": hasattr(os, "posix_fadvise"), "called": False,
                "mode": "warm_measurement_oracle"}

    base.drop_file_cache = controlled_drop
    arms: list[dict] = []
    for rep in range(1, N_REPS + 1):
        cache_mode = "cold"
        arms.append(base.run_arm("serial_cold", True, False, rep))
        cache_mode = "warm"
        arms.append(base.run_arm("serial_warm_oracle", True, False, rep))
        cache_mode = "cold"
        arms.append(base.run_arm("staged_cold", True, True, rep))
        cache_mode = "warm"
        arms.append(base.run_arm("staged_warm_oracle", True, True, rep))

    groups = {name: [x for x in arms if x["name"] == name] for name in (
        "serial_cold", "serial_warm_oracle", "staged_cold", "staged_warm_oracle")}
    summary = {name: means(rows) for name, rows in groups.items()}
    serial = summary["serial_cold"]
    staged = summary["staged_cold"]
    oracle = summary["serial_warm_oracle"]
    summary["derived"] = {
        "staged_decode_delta_fraction": staged["decode_tok_s_mean"] / serial["decode_tok_s_mean"] - 1,
        "warm_oracle_decode_delta_fraction": oracle["decode_tok_s_mean"] / serial["decode_tok_s_mean"] - 1,
        "cold_minus_warm_decode_ms_per_token": serial["decode_ms_per_token_mean"] - oracle["decode_ms_per_token_mean"],
        "warm_oracle_physical_bytes_per_token": oracle["physical_read_bytes_mean"] / base.N_GEN,
        "cold_physical_bytes_per_token": serial["physical_read_bytes_mean"] / base.N_GEN,
        "logical_expert_bytes_per_token": 127_030_400,
    }
    exactness = {
        "all_response_equal": len({x["response_sha256"] for x in arms}) == 1,
        "response_hashes": sorted({x["response_sha256"] for x in arms}),
        "native_k": 8,
        "weights_unchanged": True,
    }
    result = {
        "schema": "native-sparse-critical-path/v1",
        "status": "complete",
        "hypothesis": "v10 improves prompt batching, while exact single-token decode has little same-layer overlap; a warm-page-cache oracle measures removable unhidden storage wall time.",
        "research_source_commit": RESEARCH_COMMIT,
        "base_source_url": BASE_URL,
        "base_source_sha256": base_sha,
        "runtime": runtime,
        "model": model,
        "hardware": hardware,
        "n_gen": base.N_GEN,
        "repetitions": N_REPS,
        "cache_capacity_bytes": base.CACHE_BYTES,
        "arms": arms,
        "summary": summary,
        "exactness": exactness,
        "quality": "deterministic exact smoke only",
        "wall_sec": time.time() - started,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(__file__, OUT / Path(__file__).name)
    print(json.dumps({"summary": summary, "exactness": exactness}, indent=2), flush=True)


if __name__ == "__main__":
    main()
