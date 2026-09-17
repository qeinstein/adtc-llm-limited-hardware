"""Moonshot C probe v2: MTP N=1/N=2 with v1 method fixes (phase10j).

v1 FAILED methodologically (base CV 27% from mid-run contention, 1-decimal
quantization, no acceptance telemetry) but showed within-probe n1>n2>n3>n4
and exactness everywhere. v2: precise %.6f timing patch, INTERLEAVED base/n1
x4 pairs + n2 x2, base-CV method gate (<10% else INCONCLUSIVE), trailing
--verbose telemetry arm (excluded from means).

Vanilla upstream llama.cpp @3057bb66 (MTP is upstream; no fork needed).
Single model file: unsloth MTP-GGUF IQ2_XXS (base weights + MTP heads,
11.8 GB). Baseline runs on the SAME file without spec flags (heads never
faulted via mmap) => exact apples-to-apples on identical base weights.

Invocation verified against source @3057bb66 (NOT the ledger draft):
  -m MTPFILE --spec-type draft-mtp --spec-draft-n-min N --spec-draft-n-max N
  NO -md: without a draft path the MTP draft context is created against
  the target model (single load, mem-shared, no N clamp). With -md this
  commit loads params.model.path AS the draft (speculative.cpp ~L2562,
  -md path ignored) => same-file -md would double-load 2x12 GB.
  --draft/--draft-max are REMOVED at this commit (use --spec-draft-n-max).

Metrics: EFFECTIVE tok/s (Generation t/s) is primary; RAW verify passes
from n_decode when the perf dump prints it (best-effort). Greedy (temp 0)
speculative decoding is EXACT: every MTP arm response hash MUST equal the
baseline hash, else REJECT (no likelihood gate needed on hash match).

RAM guard: MTP arms only if MemTotal >= 12.5 GB (11.8 GB file + MTP ctx);
baseline always runs (proven ~10.6 GB footprint class).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path


WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-mtp-probe-v2")
OUT = WORK / "native-sparse-mtp-probe-v2-results"
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
CLI = BUILD / "bin" / "llama-cli"

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MTP_REPO_COMMIT = "63af8373893a7a73c6dfcb84cb63d815981da5e0"
MTP_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MTP_SIZE = 11_819_120_800
MTP_SHA256 = "3914ae7b550f0b3279ee925875ca9fa6b7dc927a7554a25ae59b10675c4448b9"
MTP_URL = ("https://huggingface.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF/resolve/"
           f"{MTP_REPO_COMMIT}/{MTP_FILE}")
MODEL = WORK / "Qwen3.5-35B-A3B-MTP-UD-IQ2_XXS.gguf"

N_THREADS = 4
N_GEN = 64
N_REPS = 2
N_DRAFTS = (1, 2, 3, 4)
SETTLE_SEC = 10
PROMPT = "Give one concise reason oral rehydration solution helps a child with watery diarrhoea."

OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], cwd: Path | None = None, log: Path | None = None) -> str:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    print(p.stdout[-3000:], flush=True)
    if log:
        log.write_text(p.stdout, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"command exited {p.returncode}: {cmd}")
    return p.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build() -> dict:
    if not (LLAMA / ".git").exists():
        run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA)])
    run(["git", "-C", str(LLAMA), "fetch", "--depth", "1", "origin", LLAMA_COMMIT])
    run(["git", "-C", str(LLAMA), "checkout", LLAMA_COMMIT])
    cli_context = LLAMA / "tools" / "cli" / "cli-context.cpp"
    text = cli_context.read_text()
    old = "[ Prompt: %.1f t/s | Generation: %.1f t/s ]"
    assert text.count(old) == 1, "precise-timing anchor not found"
    cli_context.write_text(text.replace(old, "[ Prompt: %.6f t/s | Generation: %.6f t/s ]"))
    run(["cmake", "-S", str(LLAMA), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
         "-DGGML_NATIVE=ON", "-DLLAMA_CURL=OFF"])
    run(["cmake", "--build", str(BUILD), "--config", "Release", f"-j{N_THREADS}",
         "--target", "llama-cli"], log=OUT / "build.log")
    help_text = run([str(CLI), "--help"])
    for flag in ("--spec-type", "--spec-draft-n-max", "--spec-draft-n-min", "draft-mtp"):
        if flag not in help_text:
            raise RuntimeError(f"CLI missing expected flag {flag}")
    return {"llama_commit": LLAMA_COMMIT, "cli": str(CLI), "spec_flags_verified": True}


def fetch_model() -> dict:
    # Resilient resume loop: v2-version-1 died on transient curl-56 at 11.8GB.
    for attempt in range(1, 6):
        if MODEL.exists() and MODEL.stat().st_size == MTP_SIZE:
            break
        print(f"model download attempt {attempt}/5 (resume)", flush=True)
        p = subprocess.run(["curl", "-L", "--retry", "5", "--retry-delay", "10",
                            "--retry-all-errors", "-C", "-", "-o", str(MODEL), MTP_URL],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(p.stdout[-2000:], flush=True)
        (OUT / f"model-download-attempt{attempt}.log").write_text(p.stdout, encoding="utf-8")
        if p.returncode:
            print(f"attempt {attempt} rc={p.returncode}, retrying", flush=True)
            time.sleep(15)
    size = MODEL.stat().st_size
    digest = sha256(MODEL)
    if size != MTP_SIZE or digest != MTP_SHA256:
        raise RuntimeError(f"model mismatch: size={size} sha={digest[:16]}")
    with MODEL.open("rb") as f:
        head = f.read(64 << 20)
    n_nextn = head.count(b"nextn")
    if b"nextn.eh_proj.weight" not in head:
        raise RuntimeError("MTP tensors NOT found in file header (no nextn.eh_proj.weight)")
    return {"file": MTP_FILE, "size_bytes": size, "sha256": digest,
            "nextn_mentions_in_header": n_nextn}


def drop_file_cache(path: Path) -> dict:
    os.sync()
    if not hasattr(os, "posix_fadvise"):
        return {"available": False}
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return {"available": True, "called": True}
    except OSError as exc:
        return {"available": True, "called": False, "error": repr(exc)}
    finally:
        os.close(fd)


PERF_RE = re.compile(r"eval time\s*=\s*([0-9.]+) ms /\s*([0-9]+) runs\s*\(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\s*\)")
SUMMARY_RE = re.compile(r"\[\s*Prompt:\s*([0-9.]+) t/s\s*\|\s*Generation:\s*([0-9.]+) t/s\s*\]")
NDECODE_RE = re.compile(r"n_decode\s*=\s*([0-9]+)")


def parse_perf(text: str) -> dict:
    detailed = PERF_RE.findall(text)
    if detailed:
        elapsed, runs, ms, tps = detailed[-1]
        out = {"ms_per_token": float(ms), "tokens_per_second": float(tps),
               "eval_ms": float(elapsed), "eval_runs": int(runs)}
    else:
        summary = SUMMARY_RE.findall(text)
        if not summary:
            raise RuntimeError("no parseable llama performance line")
        prompt, generation = summary[-1]
        out = {"prompt_tokens_per_second": float(prompt),
               "tokens_per_second": float(generation),
               "ms_per_token": 1000.0 / float(generation)}
    ndec = NDECODE_RE.findall(text)
    if ndec:
        out["n_decode"] = int(ndec[-1])
    return out


def response_payload(stdout: str) -> str:
    marker = stdout.find("[Start thinking]")
    if marker < 0:
        marker = stdout.rfind(PROMPT) + len(PROMPT)
    tail = stdout[marker:]
    perf = SUMMARY_RE.search(tail)
    if perf:
        tail = tail[:perf.start()]
    if tail.rstrip().endswith("Exiting..."):
        tail = tail.rstrip()[:-len("Exiting...")]
    return tail.strip()


def proc_peak(pid: int, samples: list) -> None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                samples.append(int(line.split()[1]))
                return
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass


def run_arm(name: str, rep: int, n_draft: int | None, verbose: bool = False) -> dict:
    import threading
    prefix = f"{name}_rep{rep}"
    cmd = [str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(N_THREADS),
           "-c", "512", "-n", str(N_GEN), "--temp", "0", "--seed", "1234",
           "--single-turn", "--no-display-prompt", "--no-warmup", "--perf",
           "-p", PROMPT]
    if verbose:
        cmd += ["--verbose"]
    if n_draft is not None:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-min", str(n_draft),
                "--spec-draft-n-max", str(n_draft)]
    print(f"\n===== {prefix} n_draft={n_draft} =====", flush=True)
    cache_drop = drop_file_cache(MODEL)
    os.sync()
    time.sleep(SETTLE_SEC)
    start = time.monotonic_ns()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    samples: list[int] = []
    stop = threading.Event()

    def sample_loop() -> None:
        while not stop.is_set():
            proc_peak(proc.pid, samples)
            stop.wait(0.05)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    out, err = proc.communicate()
    stop.set()
    sampler.join(timeout=2)
    proc_peak(proc.pid, samples)
    elapsed = (time.monotonic_ns() - start) / 1e9
    (OUT / f"{prefix}.stdout.txt").write_text(out, encoding="utf-8")
    (OUT / f"{prefix}.stderr.txt").write_text(err, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{prefix} exited {proc.returncode}: {err[-3000:]}")
    payload = response_payload(out)
    return {"name": name, "rep": rep, "n_draft": n_draft, "command": cmd,
            "elapsed_sec": elapsed, "cache_drop": cache_drop,
            "peak_rss_mib": (max(samples) / 1024) if samples else 0,
            "decode_perf": parse_perf(out + "\n" + err),
            "stdout_sha256": hashlib.sha256(out.encode()).hexdigest(),
            "response_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "response": payload}


def main() -> None:
    started = time.time()
    build_info = build()
    model_info = fetch_model()
    mem_kb = int([x for x in Path("/proc/meminfo").read_text().splitlines()
                  if x.startswith("MemTotal:")][0].split()[1])
    mtp_allowed = mem_kb >= 12_500_000
    print(f"RAM guard: memtotal_kib={mem_kb} mtp_allowed={mtp_allowed}", flush=True)
    arms = []
    skipped = []
    if mtp_allowed:
        # INTERLEAVED pairs: adjacent-in-time base/n1 reps defeat shared-box
        # contention drift (v1 method failure: base CV 27%).
        for rep in range(1, 5):
            arms.append(run_arm("base", rep, None))
            arms.append(run_arm("mtp_n1", rep, 1))
        for rep in range(1, 3):
            arms.append(run_arm("mtp_n2", rep, 2))
        # Trailing --verbose telemetry arm (EXCLUDED from means): captures
        # acceptance stats if the verbose log prints them.
        arms.append(run_arm("mtp_n1_verbose", 1, 1, verbose=True))
    else:
        skipped = ["mtp_n1", "mtp_n2"]
        print("SKIP MTP arms: box RAM below 12.5GB guard", flush=True)
    import statistics as st
    g = lambda n: [x for x in arms if x["name"] == n]
    base = g("base")
    base_reps = [x["decode_perf"]["tokens_per_second"] for x in base]
    base_tps = st.mean(base_reps)
    base_cv = (st.pstdev(base_reps) / base_tps) if len(base_reps) > 1 else 0.0
    base_hashes = {x["response_sha256"] for x in base}
    method_sound = mtp_allowed and len(base) == 4 and base_cv < 0.10
    means, decision = {}, {"rule": ">10% EXPLOIT; 5-10% refine-once; <5% KILL; <3% hard KILL",
                           "method_gate": {"base_cv": base_cv, "required_cv_below": 0.10,
                                           "sound": method_sound}}
    for n in (1, 2):
        rows = g(f"mtp_n{n}")
        if not rows:
            continue
        tps = st.mean(x["decode_perf"]["tokens_per_second"] for x in rows)
        gain = tps / base_tps - 1.0
        exact = all(x["response_sha256"] in base_hashes for x in rows)
        means[f"mtp_n{n}"] = {
            "decode_tok_s_mean": tps,
            "decode_tok_s_reps": [x["decode_perf"]["tokens_per_second"] for x in rows],
            "peak_rss_mib_max": max(x["peak_rss_mib"] for x in rows),
            "responses_exact_vs_baseline": exact,
        }
        verdict = ("EXPLOIT" if gain > 0.10 else "REFINE_ONCE" if gain >= 0.05
                   else "KILL" if gain >= 0.03 else "KILL_HARD")
        if not exact:
            verdict = "REJECT_OUTPUT_MISMATCH"
        if not method_sound:
            verdict = "INCONCLUSIVE_BASE_UNSTABLE"
        decision[f"mtp_n{n}_tok_s_gain_fraction"] = gain
        decision[f"verdict_mtp_n{n}"] = verdict
    means["base"] = {"decode_tok_s_mean": base_tps, "decode_tok_s_reps": base_reps,
                     "base_cv": base_cv}
    tel = g("mtp_n1_verbose")
    if tel:
        means["mtp_n1_verbose_telemetry_only"] = {
            "decode_tok_s": tel[0]["decode_perf"]["tokens_per_second"]}
    result = {"schema_version": 2, "status": "complete",
              "hypothesis": "MTP N=1 raises EFFECTIVE tok/s on Kaggle CPU (v1 method "
                            "fix: precise timing, interleaved pairs, base-CV method gate).",
              "runtime": build_info, "model": model_info,
              "ram_guard": {"memtotal_kib": mem_kb, "mtp_allowed": mtp_allowed,
                            "skipped": skipped},
              "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
              "prompt": PROMPT, "n_gen": N_GEN, "repetitions": N_REPS,
              "arms": arms, "means": means, "decision": decision,
              "metric_note": "tok/s here is EFFECTIVE (accepted tokens incl. bonus). RAW verify "
                             "passes in decode_perf.n_decode when printed. Keep RAW vs EFFECTIVE separate.",
              "wall_sec": time.time() - started}
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(__file__, OUT / Path(__file__).name)
    print(json.dumps({"means": means, "decision": decision,
                      "wall_sec": result["wall_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
