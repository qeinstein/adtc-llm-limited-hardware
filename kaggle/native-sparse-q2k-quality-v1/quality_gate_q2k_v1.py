"""Likelihood quality gate for Q2_K challengers (experts + all).

Self-contained vanilla build (quality needs no fork): clone pinned
llama.cpp, build llama-quantize + llama-perplexity, download the IQ2_XXS
model (sha-pinned), transcode experts-only and all-Q2_K (same anchored
override logic as the combo kernel), then matched MMLU likelihood evals
(control / q2k-experts / q2k-all, v3 OOM-repaired flags, 100 tasks).

Decision rule (v3-identical): REJECT if matched score drops > 2.0 pp.
Push ONLY if the combo kernel wins on speed (phase10h plan).
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
import urllib.request
from pathlib import Path


WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-q2k-quality-v1")
OUT = WORK / "native-sparse-q2k-quality-v1-results"
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
QUANTIZE = BUILD / "bin" / "llama-quantize"
PERPLEXITY = BUILD / "bin" / "llama-perplexity"

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_SHA256 = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
MODEL_URL = ("https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
             f"{MODEL_REPO_COMMIT}/{MODEL_FILE}")
DATA_REVISION = "37884b81b4957f1950a53b6ff48d77c8dd5e430c"
DATA_URL = ("https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/"
            f"resolve/{DATA_REVISION}/mmlu-test.bin")
N_TASKS = 100
N_THREADS = 4

MODEL_IQ2 = WORK / MODEL_FILE
MODEL_Q2K = SCRATCH / "Qwen3.5-35B-A3B-Q2K-experts.gguf"
MODEL_Q2KALL = SCRATCH / "Qwen3.5-35B-A3B-Q2K-all.gguf"

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
    run(["cmake", "-S", str(LLAMA), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
         "-DGGML_NATIVE=ON", "-DLLAMA_CURL=OFF"])
    run(["cmake", "--build", str(BUILD), "--config", "Release", f"-j{N_THREADS}",
         "--target", "llama-quantize", "llama-perplexity"],
        log=OUT / "build.log")
    return {"llama_commit": LLAMA_COMMIT,
            "quantize": str(QUANTIZE), "perplexity": str(PERPLEXITY)}


def fetch_model() -> dict:
    if not MODEL_IQ2.exists() or MODEL_IQ2.stat().st_size != MODEL_SIZE:
        run(["curl", "-L", "--retry", "3", "-C", "-", "-o", str(MODEL_IQ2), MODEL_URL],
            log=OUT / "model-download.log")
    size = MODEL_IQ2.stat().st_size
    digest = sha256(MODEL_IQ2)
    if size != MODEL_SIZE or digest != MODEL_SHA256:
        raise RuntimeError(f"model mismatch: size={size} sha={digest[:16]}")
    return {"file": MODEL_FILE, "size_bytes": size, "sha256": digest}


GGML_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 9: "q8_1", 10: "q2_k", 11: "q3_k", 12: "q4_k",
    13: "q5_k", 14: "q6_k", 15: "q8_k", 16: "iq2_xxs", 17: "iq2_xs",
    18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s",
    23: "iq4_xs", 24: "i8", 25: "i16", 26: "i32", 27: "i64", 28: "f64",
    29: "iq1_m", 30: "bf16", 34: "tq1_0", 35: "tq2_0", 36: "mxfp4"}


def gguf_tensor_table(path: Path) -> list[tuple[str, str]]:
    import struct
    with path.open("rb") as f:
        head = f.read(64 << 20)
    if head[0:4] != b"GGUF":
        raise RuntimeError("not a GGUF file")
    _ver, n_tensors, n_kv = struct.unpack_from("<IQQ", head, 4)
    off = 24

    def read_str(o: int) -> tuple[str, int]:
        (n,) = struct.unpack_from("<Q", head, o)
        o += 8
        return head[o:o + n].decode("utf-8", "replace"), o + n

    def skip_value(o: int, typ: int) -> int:
        if typ in (4, 5, 6):
            return o + 4
        if typ in (10, 11, 12):
            return o + 8
        if typ in (0, 1, 7):
            return o + 1
        if typ in (2, 3):
            return o + 2
        if typ == 8:
            _, o = read_str(o)
            return o
        if typ == 9:
            (at,) = struct.unpack_from("<I", head, o)
            o += 4
            (n,) = struct.unpack_from("<Q", head, o)
            o += 8
            if at == 8:
                for _ in range(n):
                    _, o = read_str(o)
                return o
            return o + {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                        10: 8, 11: 8, 12: 8}[at] * n
        raise RuntimeError(f"unknown GGUF KV type {typ}")

    for _ in range(n_kv):
        _, off = read_str(off)
        (typ,) = struct.unpack_from("<I", head, off)
        off = skip_value(off + 4, typ)
    out = []
    for _ in range(n_tensors):
        name, off = read_str(off)
        (nd,) = struct.unpack_from("<I", head, off)
        off += 4 + 8 * nd
        (typ,) = struct.unpack_from("<I", head, off)
        off += 4 + 8
        out.append((name, GGML_TYPE_NAMES.get(typ, f"T{typ}")))
    return out


def write_overrides(path: Path, table: list[tuple[str, str]], mode: str) -> dict:
    import re as _re
    lines = []
    n_q2k = 0
    for name, typ in table:
        if "_exps" in name:
            target = "q2_k"
        elif mode == "all" and typ not in ("f32",) and (
                "attn" in name or "mlp" in name or "ffn" in name or
                "output.weight" in name or "token_embd" in name or
                "delta" in name or "shexp" in name or "router" in name or
                "ssm" in name):
            target = "q2_k"
        else:
            target = typ
        if target == "q2_k":
            n_q2k += 1
        lines.append(f"^{_re.escape(name)}$={target}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"tensors": len(lines), "to_q2k": n_q2k}


def transcode(name: str, out_path: Path, mode: str) -> dict:
    table = gguf_tensor_table(MODEL_IQ2)
    ov = write_overrides(OUT / f"overrides_{name}.txt", table, mode)
    print(f"transcode {name}: {ov['to_q2k']}/{ov['tensors']} tensors -> q2_k", flush=True)
    t0 = time.time()
    run([str(QUANTIZE), "--allow-requantize", "--tensor-type-file",
         str(OUT / f"overrides_{name}.txt"),
         str(MODEL_IQ2), str(out_path), "Q8_0", str(N_THREADS)],
        log=OUT / f"transcode_{name}.log")
    dt = time.time() - t0
    check = gguf_tensor_table(out_path)
    n_exp = sum(1 for n, t in check if "_exps" in n and t == "q2_k")
    if n_exp != 120:
        raise RuntimeError(f"{name}: expert q2_k count {n_exp} != 120")
    return {"mode": mode, "size_bytes": out_path.stat().st_size,
            "sha256": sha256(out_path), "sec": dt,
            "q2k_tensors": sum(1 for _, t in check if t == "q2_k")}


def run_eval(model: Path, dataset: Path, label: str) -> dict:
    cmd = [str(PERPLEXITY), "-m", str(model), "-f", str(dataset), "-ngl", "0",
           "-t", str(N_THREADS), "-c", "512", "-b", "256", "-ub", "64", "-np", "2",
           "--multiple-choice", "--multiple-choice-tasks", str(N_TASKS),
           "--no-repack", "--no-warmup"]
    print("+", " ".join(cmd), flush=True)
    started = time.monotonic()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=False)
    elapsed = time.monotonic() - started
    (OUT / f"{label}.stdout.txt").write_text(p.stdout, encoding="utf-8")
    (OUT / f"{label}.stderr.txt").write_text(p.stderr, encoding="utf-8")
    text = p.stdout + "\n" + p.stderr
    finals = re.findall(r"Final result:\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", text)
    rows = re.findall(r"^\s*(\d+)\s+([0-9.]+)\s*$", text, flags=re.MULTILINE)
    if p.returncode or not finals:
        raise RuntimeError(f"{label} eval failed rc={p.returncode}: {text[-3000:]}")
    score, sigma = finals[-1]
    return {"label": label, "elapsed_sec": elapsed,
            "score_percent": float(score), "sigma_percent": float(sigma),
            "running_accuracy": [[int(n), float(v)] for n, v in rows],
            "stdout_sha256": hashlib.sha256(p.stdout.encode()).hexdigest()}


def main() -> None:
    started = time.time()
    build_info = build()
    model_info = fetch_model()
    transc = {"experts": transcode("experts", MODEL_Q2K, "experts")}
    urllib.request.urlretrieve(DATA_URL, SCRATCH / "mmlu-test.bin")
    dataset = SCRATCH / "mmlu-test.bin"
    dataset_info = {"url": DATA_URL, "revision": DATA_REVISION,
                    "size_bytes": dataset.stat().st_size,
                    "sha256": sha256(dataset)}
    control = run_eval(MODEL_IQ2, dataset, "control")
    q2k = run_eval(MODEL_Q2K, dataset, "q2k_experts")
    MODEL_Q2K.unlink(missing_ok=True)
    transc["all"] = transcode("all", MODEL_Q2KALL, "all")
    q2kall = run_eval(MODEL_Q2KALL, dataset, "q2k_all")
    MODEL_Q2KALL.unlink(missing_ok=True)
    d_exp = q2k["score_percent"] - control["score_percent"]
    d_all = q2kall["score_percent"] - control["score_percent"]
    result = {
        "schema": "native-sparse-q2k-quality/v1",
        "status": "complete",
        "hypothesis": "Q2_K experts (and dense) preserve matched MMLU likelihood accuracy within 2 pp.",
        "runtime": {"llama_commit": LLAMA_COMMIT, "threads": N_THREADS,
                    "context": 512, "batch": 256, "ubatch": 64, "parallel": 2},
        "build": build_info, "model": model_info, "transcodes": transc,
        "dataset": dataset_info,
        "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "tasks": N_TASKS, "control": control, "q2k_experts": q2k, "q2k_all": q2kall,
        "delta_experts_pp": d_exp, "delta_all_pp": d_all,
        "decision_experts": "KEEP" if d_exp >= -2.0 else "REJECT_QUALITY_REGRESSION",
        "decision_all": "KEEP" if d_all >= -2.0 else "REJECT_QUALITY_REGRESSION",
        "decision_rule": "reject if matched score drops by more than 2.0 percentage points",
        "wall_sec": time.time() - started,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(__file__, OUT / Path(__file__).name)
    print(json.dumps({"control": control["score_percent"],
                      "q2k_experts": q2k["score_percent"],
                      "q2k_all": q2kall["score_percent"],
                      "delta_exp_pp": d_exp, "delta_all_pp": d_all}, indent=2), flush=True)


if __name__ == "__main__":
    main()
