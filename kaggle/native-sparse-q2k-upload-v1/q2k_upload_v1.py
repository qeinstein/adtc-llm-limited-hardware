"""Re-transcode the frozen Q2_K-experts artifact and upload it to Hugging Face.

Determinism is PROVEN (edge0base v5 re-transcoded byte-identical), so this
kernel reproduces sha 0f3698ae... exactly, asserts it, then uploads:
  - Qwen3.6-35B-A3B-UD-Q2K-experts.gguf (12,262,341,600 bytes)
  - .sha256 sidecar + model card README.md

Requires an HF write token as Kaggle Secret env var HF_TOKEN (Add-ons ->
Secrets -> attach to this notebook). Fails LOUD without it. The token
never appears in logs or outputs.

v2: trigger run with secret attached (no code change).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-q2k-upload-v1")
OUT = WORK / "native-sparse-q2k-upload-v1-results"
LLAMA = SCRATCH / "llama.cpp"
BUILD = LLAMA / "build-native"
QUANTIZE = BUILD / "bin" / "llama-quantize"

LLAMA_COMMIT = "3057bb66c86c46d5781e50e85462a760ba7d1feb"
BASE = {"repo": "a483e9e6cbd595906af30beda3187c2663a1118c",
        "file": "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
        "size": 10_756_586_464,
        "sha": "2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef",
        "hf": "unsloth/Qwen3.6-35B-A3B-GGUF"}
Q2K_NAME = "Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"
Q2K_SIZE = 12_262_341_600
Q2K_SHA = "0f3698ae92f91db2eb10a3650bdb6693ff8cfaf7060cbc5f256845c700c7603b"
HF_REPO = "Fluxx08/jamii-afya-qwen36-35b-q2k"
THREADS = 4

GGML_TYPE_NAMES = {0: "f32", 1: "f16", 8: "q8_0", 10: "q2_k", 12: "q4_k",
                   13: "q5_k", 14: "q6_k", 16: "iq2_xxs", 21: "iq3_s"}

CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
tags:
- gguf
- moe
- medical
- swahili
- llama.cpp
---

# Jamii Afya — Qwen3.6-35B-A3B routed-expert Q2_K GGUF

Deployment artifact for the Jamii Afya ADTC 2026 submission (offline clinical
advisor for African frontline health workers, English + Kiswahili).

## File

- `{q2k}` — {size} bytes
- SHA256: `{sha}`

## Provenance

- Base: `unsloth/Qwen3.6-35B-A3B-GGUF` (`{base_file}` @ `{base_commit}`)
  - SHA256: `{base_sha}`
- Transform: `llama-quantize --allow-requantize` @ llama.cpp `{llama_commit}`
  with per-tensor overrides: the 120 routed-expert tensors (`*_exps`) are
  requantized to **Q2_K**; every other tensor keeps its base type. No
  weight-level fine-tuning was performed (no LoRA/QLoRA/full finetune).
- Runtime: K4/16 sparse expert execution + bounded expert staging
  (see the submission repo). Intended for CPU-only, offline inference.

## Reproduce / use

```bash
git clone https://github.com/qeinstein/adtc-llm-limited-hardware.git
cd adtc-llm-limited-hardware
make model    # downloads + verifies this file
make webui    # builds the runtime and launches the chat UI
```

## Limitations

- The shipping weights are NOT fine-tuned and NOT clinically validated.
- Safety comes from the system prompt, deterministic safety rules, structured
  clinical guidance retrieval, and output linting in the submission repo —
  not from the weights alone. Always keep a clinician in the loop.
""".format(q2k=Q2K_NAME, size=Q2K_SIZE, sha=Q2K_SHA,
           base_file=BASE["file"], base_commit=BASE["repo"],
           base_sha=BASE["sha"], llama_commit=LLAMA_COMMIT)


def run_checked(cmd, cwd=None, log=None):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    p = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       check=False)
    print(p.stdout[-2000:], flush=True)
    if log:
        Path(log).write_text(p.stdout, encoding="utf-8")
    if p.returncode:
        raise RuntimeError(f"command exited {p.returncode}: {cmd}")
    return p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(8 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def gguf_tensor_table(path):
    head = open(path, "rb").read(64 << 20)
    assert head[:4] == b"GGUF"
    n_tensors, n_kv = struct.unpack_from("<QQ", head, 8)
    off = 24

    def read_str(o):
        (n,) = struct.unpack_from("<Q", head, o)
        return head[o + 8:o + 8 + n].decode(), o + 8 + n

    _SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
               10: 8, 11: 8, 12: 8}

    def skip_value(o, typ):
        if typ in _SCALAR:
            return o + _SCALAR[typ]
        if typ == 8:
            _, o2 = read_str(o)
            return o2
        if typ == 9:
            (at,) = struct.unpack_from("<I", head, o)
            (n,) = struct.unpack_from("<Q", head, o + 4)
            o += 12
            if at == 8:
                for _ in range(n):
                    _, o = read_str(o)
                return o
            return o + _SCALAR[at] * n
        raise RuntimeError(f"kv type {typ}")

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


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "").strip()
    via = "env"
    if not token:
        # Attached private token-drop dataset (created via API, deleted
        # minutes after this read; never logged, never in outputs).
        cands = sorted(Path("/kaggle/input").rglob("token.txt"))
        if len(cands) == 1:
            token = cands[0].read_text().strip()
            via = "attached dataset"
    if not token:
        # Kaggle Secrets attached to the notebook (UI: Add-ons -> Secrets).
        try:
            from kaggle_secrets import UserSecretsClient
            token = (UserSecretsClient().get_secret("HF_TOKEN") or "").strip()
            via = "UserSecretsClient"
        except Exception as exc:
            print(f"HF_TOKEN: UserSecretsClient failed ({exc!r})", flush=True)
    if not token:
        raise SystemExit(
            "HF_TOKEN is empty: attach a Hugging Face WRITE token as a "
            "Kaggle Secret named HF_TOKEN (notebook Add-ons -> Secrets), "
            "then re-run. Nothing was uploaded.")
    print(f"HF_TOKEN acquired via {via} (value never logged)", flush=True)
    for p in (WORK, SCRATCH):
        u = shutil.disk_usage(p)
        print(f"disk {p}: free={u.free/1e9:.1f}GB", flush=True)
    # 1. vanilla pinned build (quantize path untouched by edge0 patches).
    if not (LLAMA / ".git").exists():
        run_checked(["git", "clone", "--filter=blob:none",
                     "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run_checked(["git", "-C", str(LLAMA), "fetch", "--depth", "1", "origin",
                 LLAMA_COMMIT])
    run_checked(["git", "-C", str(LLAMA), "checkout", "--detach", LLAMA_COMMIT])
    head = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"],
                          check=True, text=True,
                          capture_output=True).stdout.strip()
    assert head == LLAMA_COMMIT, head
    run_checked(["cmake", "-S", str(LLAMA), "-B", str(BUILD),
                 "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON",
                 "-DLLAMA_CURL=ON"], log=OUT / "cmake-configure.log")
    run_checked(["cmake", "--build", str(BUILD), "--config", "Release",
                 "-j4", "--target", "llama-quantize"],
                log=OUT / "cmake-build.log")
    # 2. base download + identity.
    url = f"https://huggingface.co/{BASE['hf']}/resolve/{BASE['repo']}/{BASE['file']}"
    dest = SCRATCH / BASE["file"]
    if not dest.exists() or dest.stat().st_size != BASE["size"]:
        run_checked(["curl", "-L", "--fail", "--http1.1", "--retry", "5",
                     "--retry-all-errors", "--retry-delay", "5", "-C", "-",
                     "-o", str(dest), url], log=OUT / "model-download.log")
    d = sha256(dest)
    if dest.stat().st_size != BASE["size"] or d != BASE["sha"]:
        raise RuntimeError("base model identity mismatch")
    print(f"base ok: {d[:16]}...", flush=True)
    # 3. overrides (same rule as JOIN4b: *_exps -> q2_k, else keep).
    table = gguf_tensor_table(dest)
    lines, n = [], 0
    for name, typ in table:
        tgt = "q2_k" if "_exps" in name else typ
        n += tgt == "q2_k"
        lines.append(f"^{re.escape(name)}$={tgt}")
    (OUT / "overrides_q2k.txt").write_text("\n".join(lines) + "\n")
    print(f"overrides: {n}/{len(lines)} -> q2_k", flush=True)
    assert n == 120, n
    # 4. transcode + byte-identity assert (determinism proven by v5).
    q2k_path = SCRATCH / Q2K_NAME
    t0 = time.time()
    run_checked([str(QUANTIZE), "--allow-requantize", "--tensor-type-file",
                 str(OUT / "overrides_q2k.txt"), str(dest), str(q2k_path),
                 "Q8_0", str(THREADS)], log=OUT / "transcode_q2k.log")
    print(f"transcode took {time.time()-t0:.0f}s", flush=True)
    size = q2k_path.stat().st_size
    digest = sha256(q2k_path)
    print(f"size={size} sha={digest[:16]}...", flush=True)
    if size != Q2K_SIZE or digest != Q2K_SHA:
        raise RuntimeError(
            f"BYTE IDENTITY FAIL: size={size} (want {Q2K_SIZE}) "
            f"sha={digest[:16]}... (want {Q2K_SHA[:16]}...). NOT uploading.")
    chk = gguf_tensor_table(q2k_path)
    nexp = sum(1 for x, t in chk if "_exps" in x and t == "q2_k")
    assert nexp == 120, nexp
    print("byte-identical to frozen artifact. Uploading.", flush=True)
    # 5. HF upload (repo + gguf + sidecar + card).
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(HF_REPO, exist_ok=True, private=False)
    (SCRATCH / "README.md").write_text(CARD, encoding="utf-8")
    (SCRATCH / (Q2K_NAME + ".sha256")).write_text(
        f"{digest}  {Q2K_NAME}\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(q2k_path), path_in_repo=Q2K_NAME,
                    repo_id=HF_REPO)
    api.upload_file(path_or_fileobj=str(SCRATCH / "README.md"),
                    path_in_repo="README.md", repo_id=HF_REPO)
    api.upload_file(path_or_fileobj=str(SCRATCH / (Q2K_NAME + ".sha256")),
                    path_in_repo=Q2K_NAME + ".sha256", repo_id=HF_REPO)
    # 6. remote verify (size via HEAD; never trust the upload call alone).
    import urllib.request
    rev = api.repo_info(HF_REPO).sha
    for fname, want in ((Q2K_NAME, Q2K_SIZE), ("README.md", None),
                        (Q2K_NAME + ".sha256", None)):
        req = urllib.request.Request(
            f"https://huggingface.co/{HF_REPO}/resolve/{rev}/{fname}",
            method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            got = int(r.headers.get("Content-Length", -1))
        print(f"remote {fname}: {got} bytes", flush=True)
        if want is not None and got != want:
            raise RuntimeError(f"remote size mismatch for {fname}")
    (OUT / "result.json").write_text(json.dumps(
        {"repo": HF_REPO, "rev": rev, "file": Q2K_NAME, "size": Q2K_SIZE,
         "sha256": Q2K_SHA}, indent=1))
    print(f"UPLOAD COMPLETE: {HF_REPO}@{rev[:12]}", flush=True)


if __name__ == "__main__":
    main()
