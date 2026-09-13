"""Phase 5H: inventory the exact non-routed resident model floor.

The model and runtime are unchanged mathematically.  A loader hook records
every model tensor's logical size and lazy flag, while the already-established
measurement-only routed-MoE skip arm measures the process RSS floor.  This is
an inventory experiment, not a valid generation path.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path


SCRATCH = Path("/tmp/native-sparse-floor-decomp")
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
CLI = BUILD / "bin" / "llama-cli"
OUT = Path("/kaggle/working/native-sparse-floor-decomp-v1-results")
LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
MODEL_REPO_COMMIT = "bc014a17be43adabd7066b7a86075ff935c6a4e2"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
MODEL_SIZE = 10_656_955_008
MODEL_SHA256 = "2a809de317cfd49ac9130b95619ee6ac039a855e467015b3e996eb61af84718b"
MODEL = SCRATCH / MODEL_FILE
MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/"
    f"{MODEL_REPO_COMMIT}/{MODEL_FILE}"
)
PROMPT = "Give one concise reason oral rehydration solution helps a child with watery diarrhoea."
THREADS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(cmd: list[str], *, cwd: Path | None = None, log: Path | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    process = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if log is not None:
        log.write_text(process.stdout + "\n--- STDERR ---\n" + process.stderr, encoding="utf-8")
    if process.returncode:
        print(process.stdout[-2000:], flush=True)
        print(process.stderr[-2000:], flush=True)
        raise RuntimeError(f"command exited {process.returncode}")
    return process


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: patch anchor count mismatch ({text.count(old)})")
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_runtime() -> dict:
    qwen = LLAMA / "src/models/qwen35moe.cpp"
    cpu = LLAMA / "ggml/src/ggml-cpu/ggml-cpu.c"
    loader = LLAMA / "src/llama-model-loader.cpp"

    # Keep all non-expert tensors on the normal resident path.  Only the
    # three routed expert tensors per layer are lazy, matching the Phase 0
    # exact control's storage intervention.
    replace_once(qwen,
        "        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, \"weight\", il), { n_ff_exp, n_embd, n_expert }, flags);\n"
        "        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);",
        "        const int expert_flags = flags | TENSOR_READ_LAZY;\n"
        "        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, \"weight\", il), { n_ff_exp, n_embd, n_expert }, expert_flags);\n"
        "        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, expert_flags);")

    # This arm retains the graph and every non-MoE allocation but replaces
    # selected-expert outputs before the lazy expert mapping is touched.
    skip_hook = r'''
    const char * phase5_skip_moe = getenv("GGML_PHASE5_SKIP_MOE");
    if (tensor->op == GGML_OP_MUL_MAT_ID && phase5_skip_moe != NULL &&
            atoi(phase5_skip_moe) != 0) {
        if (params->ith == 0) {
            memset(tensor->data, 0, ggml_nbytes(tensor));
        }
        ggml_barrier(params->threadpool);
        return;
    }

'''
    replace_once(cpu,
        "    // extra_buffer op?\n",
        skip_hook + "    // extra_buffer op?\n")

    # Logical tensor inventory emitted by the model-loader after tensor
    # creation.  Names are GGUF-safe and the fields are sufficient to split
    # the non-MoE floor into model categories without reading tensor data.
    replace_once(loader,
        "#include <cstdint>\n",
        "#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n")
    inventory = r'''
    if (const char * inventory_path = std::getenv("GGML_FLOOR_INVENTORY")) {
        static FILE * inventory_file = nullptr;
        if (inventory_file == nullptr) {
            inventory_file = std::fopen(inventory_path, "a");
            if (inventory_file != nullptr) {
                std::setvbuf(inventory_file, nullptr, _IOLBF, 0);
            }
        }
        if (inventory_file != nullptr) {
            std::fprintf(inventory_file,
                "{\"name\":\"%s\",\"nbytes\":%zu,\"type\":%d,\"lazy\":%d,\"flags\":%d}\n",
                tensor->name, ggml_nbytes(tensor), (int) tensor->type,
                is_lazy ? 1 : 0, flags);
        }
    }

'''
    replace_once(loader,
        "    return tensor;\n}\n\nvoid llama_model_loader::done_getting_tensors",
        inventory + "    return tensor;\n}\n\nvoid llama_model_loader::done_getting_tensors")

    return {"llama_commit": LLAMA_COMMIT,
            "qwen_patch": "routed expert tensors marked TENSOR_READ_LAZY",
            "skip_patch": "GGML_PHASE5_SKIP_MOE measurement-only graph arm",
            "inventory_patch": "GGML_FLOOR_INVENTORY logical tensor records"}


def setup() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not LLAMA.exists():
        run_checked(["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run_checked(["git", "fetch", "--depth", "1", "origin", LLAMA_COMMIT], cwd=LLAMA)
    run_checked(["git", "checkout", "--detach", LLAMA_COMMIT], cwd=LLAMA)
    patch = patch_runtime()
    configure = ["cmake", "-S", str(LLAMA), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
                 "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF",
                 "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF"]
    run_checked(configure, log=OUT / "cmake-configure.log")
    build = ["cmake", "--build", str(BUILD), "--config", "Release", "-j4", "--target", "llama-cli"]
    run_checked(build, log=OUT / "cmake-build.log")
    (OUT / "runtime.patch").write_text(
        run_checked(["git", "diff", "--", "src/models/qwen35moe.cpp", "ggml/src/ggml-cpu/ggml-cpu.c",
                     "src/llama-model-loader.cpp"], cwd=LLAMA).stdout, encoding="utf-8")
    (OUT / "llama-version.txt").write_text(run_checked([str(CLI), "--version"]).stdout, encoding="utf-8")
    return {"runtime": patch, "configure": configure, "build": build,
            "compiler": run_checked(["c++", "--version"]).stdout.splitlines()[0]}


def fetch_model() -> dict:
    if not MODEL.exists() or MODEL.stat().st_size != MODEL_SIZE:
        run_checked(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5", "-C", "-",
                     "-o", str(MODEL), MODEL_URL], log=OUT / "model-download.log")
    size, digest = MODEL.stat().st_size, sha256(MODEL)
    if size != MODEL_SIZE or digest != MODEL_SHA256:
        raise RuntimeError(f"model identity mismatch: {size} {digest}")
    return {"repo": "unsloth/Qwen3.5-35B-A3B-GGUF", "revision": MODEL_REPO_COMMIT,
            "file": MODEL_FILE, "size_bytes": size, "sha256": digest}


def proc_sample(pid: int) -> dict:
    sample = {"mono_ns": time.monotonic_ns(), "rss_kib": 0, "anon_kib": 0,
              "file_kib": 0, "minor_faults": 0, "major_faults": 0}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"): sample["rss_kib"] = int(line.split()[1])
            elif line.startswith("RssAnon:"): sample["anon_kib"] = int(line.split()[1])
            elif line.startswith("RssFile:"): sample["file_kib"] = int(line.split()[1])
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        sample["minor_faults"], sample["major_faults"] = int(tail[7]), int(tail[9])
        sample["valid"] = True
    except (OSError, ValueError):
        sample["valid"] = False
    return sample


def run_floor() -> dict:
    inventory = OUT / "tensor_inventory.jsonl"
    inventory.unlink(missing_ok=True)
    cmd = [str(CLI), "-m", str(MODEL), "-ngl", "0", "-t", str(THREADS), "-c", "512", "-n", "64",
           "--temp", "0", "--seed", "1234", "--single-turn", "--no-display-prompt", "--no-warmup",
           "--perf", "-lm", "mmap", "-lzm", "on", "--poll", "0", "-p", PROMPT]
    env = dict(os.environ, GGML_PHASE5_SKIP_MOE="1", GGML_FLOOR_INVENTORY=str(inventory))
    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    samples = []
    while process.poll() is None:
        samples.append(proc_sample(process.pid))
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    (OUT / "floor.stdout.txt").write_text(stdout, encoding="utf-8")
    (OUT / "floor.stderr.txt").write_text(stderr, encoding="utf-8")
    valid = [sample for sample in samples if sample.get("valid")]
    return {"command": cmd, "returncode": process.returncode,
            "peak_rss_mib": max((x["rss_kib"] for x in valid), default=0) / 1024,
            "peak_anon_mib": max((x["anon_kib"] for x in valid), default=0) / 1024,
            "peak_file_mib": max((x["file_kib"] for x in valid), default=0) / 1024,
            "minor_faults": max((x["minor_faults"] for x in valid), default=0),
            "major_faults": max((x["major_faults"] for x in valid), default=0),
            "samples": len(valid), "inventory_records": sum(1 for _ in inventory.open()) if inventory.exists() else 0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"schema": "native-sparse-floor-decomp/v1", "status": "failed",
              "hypothesis": "The 1.61 GiB non-MoE operational floor should be decomposed into logical tensor categories and actual anonymous/file-backed RSS.",
              "research_source": {"base_commit": "d1df128", "script_sha256": sha256(Path(__file__))},
              "model": {}, "hardware": {"platform": platform.platform(), "python": platform.python_version(),
              "cpu_count": os.cpu_count(), "lscpu": run_checked(["lscpu"]).stdout}, "runs": []}
    try:
        result["build"] = setup()
        result["model"] = fetch_model()
        result["runs"] = [run_floor()]
        result["status"] = "ok" if result["runs"][0]["returncode"] == 0 else "invalid_result"
    except Exception as exc:
        result["error"] = repr(exc)
        print(f"ERROR: {exc}", flush=True)
    result["finished_unix"] = time.time()
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "runs": result.get("runs"), "error": result.get("error")}), flush=True)
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
