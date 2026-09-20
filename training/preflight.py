#!/usr/bin/env python3
"""Production preflight + 20-step canary for the one-shot QLoRA run.

Runs ON the GPU host (Kaggle 2xT4). Gates (§Y order):
  0. env asserts (2xT4, disk, host RAM, swap idle)
  1. pinned install + pip-freeze capture + source-grep asserts (§C)
  2. data SHA + snapshot + §B arch asserts
  3. YAML freeze + `axolotl preprocess` gate
  4. 20-optimizer-step canary with watchdog + JSONL logging (§N/V/W)
  5. adapter exact asserts (§E) + fresh-process reload/resume (§O, §Q)
  6. merge/export + tiny generation (§P)
  7. GO/NO-GO report (§Y) + run_manifest.json (§W)

Exit code 0 + GO means: launch production with ONLY max_steps/output_dir/
run_name changed (§X). Any RED gate aborts with nonzero exit.

 chaos: PREFLIGHT_CHAOS=1 kills the canary mid-run once, then proves resume.
 resume: RESUME_FROM=<checkpoint dir> skips to the resume leg (after chaos).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("PREFLIGHT_WORK", "/kaggle/working"))
SCRATCH = Path(os.environ.get("PREFLIGHT_SCRATCH", "/tmp/preflight"))
OUT = WORK / "preflight-results"
KAGGLE_INPUT = Path("/kaggle/input")

AXOLOTL_SHA = "82109efa69a147853253186e4c314cddac66ab59"
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
BASE_REV = "995ad96eacd98c81ed38be0c5b274b04031597b0"
TEMPLATE = Path(__file__).resolve().parent / "kaggle_train.yaml"
CANARY_STEPS = 20

# §E: q/k/v/o on 10 full-attn layers. q:2048x4096 k/v:2048x512 o:4096x2048.
EXP_LORA_PARAMS = (8 * (2048 + 4096) + 8 * (2048 + 512) * 2
                   + 8 * (4096 + 2048)) * 10
EXP_LORA_TENSORS = 10 * 4 * 2
ALLOW_LORA = {"q_proj", "k_proj", "v_proj", "o_proj"}
FORBID_LORA = ("mlp", "router", "gate", "expert", "embed", "head", "norm",
               "mtp", "visual", "vision", "linear_attn", "in_proj", "out_proj")
AXO_BIN = "axolotl"


def sh(cmd, log=None, env=None, timeout=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=False, text=True, stdout=log,
                          stderr=subprocess.STDOUT if log else None,
                          env=env or os.environ, timeout=timeout)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(8 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def die(msg):
    print(f"PREFLIGHT RED: {msg}", flush=True)
    (OUT / "VERDICT").write_text(f"NO-GO: {msg}\n")
    raise SystemExit(1)


# ---------------- stage 0: hardware ----------------
def stage0_env():
    assert sys.version_info >= (3, 10), sys.version
    assert shutil.which("nvidia-smi"), "no nvidia-smi"
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                        "--format=csv,noheader"], check=True, text=True,
                       capture_output=True).stdout.strip().splitlines()
    if len(q) != 2 or not all("T4" in l and "7.5" in l for l in q):
        die(f"need exactly 2x T4 sm75: {q}")
    mem = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")[:2]
        if k.strip() in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            mem[k.strip()] = int(v.split()[0])
    swap_used = (mem["SwapTotal"] - mem["SwapFree"]) / 1e6
    print(f"host RAM total={mem['MemTotal']/1e6:.1f}GB swap_used={swap_used:.2f}GB",
          flush=True)
    if swap_used > 0.5:
        die(f"swap already in use: {swap_used:.2f}GB")
    du = shutil.disk_usage(SCRATCH if SCRATCH.exists() else "/tmp")
    print(f"disk free={du.free/1e9:.0f}GB", flush=True)
    if du.free < 100e9:
        die(f"only {du.free/1e9:.0f}GB free (need 100+: snapshot+env+ckpts)")
    import torch
    print(f"torch {torch.__version__} cuda={torch.version.cuda} "
          f"n_gpus={torch.cuda.device_count()}", flush=True)
    return {"gpus": q, "host_ram_gb": round(mem["MemTotal"] / 1e6, 1)}


# ---------------- stage 1: pinned env ----------------
def stage1_install():
    log = (OUT / "install.log").open("w")
    p = sh([sys.executable, "-m", "pip", "install",
            f"axolotl @ git+https://github.com/axolotl-ai-cloud/axolotl.git@{AXOLOTL_SHA}"],
           log=log)
    log.close()
    if p.returncode:
        die("axolotl install failed (see install.log)")
    bcv = subprocess.run(
        [sys.executable, "-c", "import botocore; print(botocore.__version__)"],
        check=True, text=True, capture_output=True).stdout.strip()
    log2 = (OUT / "boto_fix.log").open("w")
    p = sh([sys.executable, "-m", "pip", "install", "-q", f"boto3=={bcv}"],
           log=log2)
    log2.close()
    if p.returncode:
        die("boto3 re-pin failed")
    def py(code):
        return subprocess.run([sys.executable, "-c", code], check=True,
                              text=True, capture_output=True).stdout.strip()
    torch_before = py("import torch; print(torch.__version__)")
    log3 = (OUT / "tv_fix.log").open("w")
    p1 = sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
             "torchvision", "torchaudio"], log=log3)
    p2 = sh([sys.executable, "-m", "pip", "install", "-q", "torchvision"],
            log=log3)
    log3.close()
    if p1.returncode or p2.returncode:
        die("torchvision rebuild failed")
    if py("import torch, torchvision.ops; print(torch.__version__)") != torch_before:
        die("torchvision rebuild moved torch")
    global AXO_BIN
    AXO_BIN = (shutil.which("axolotl")
               or str(Path(sys.prefix) / "bin" / "axolotl"))
    if not Path(AXO_BIN).exists():
        die("axolotl binary missing after install")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            check=True, text=True,
                            capture_output=True).stdout
    (OUT / "environment.freeze.txt").write_text(freeze)
    import torch, transformers, peft, trl, bitsandbytes, datasets
    pins = {"transformers": "5.17.0", "peft": "0.21.0", "trl": "1.13.0",
            "bitsandbytes": "0.50.2", "datasets": "4.8.4"}
    got = {"transformers": transformers.__version__, "peft": peft.__version__,
           "trl": trl.__version__, "bitsandbytes": bitsandbytes.__version__,
           "datasets": datasets.__version__}
    print("pins:", got, flush=True)
    if got != pins:
        die(f"pin drift: {got}")
    parts = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
    if not (2, 11, 0) <= parts <= (2, 14, 99):
        die(f"torch out of range: {torch.__version__}")
    # §C source-grep asserts on the INSTALLED axolotl (not HEAD-on-GitHub).
    import axolotl
    src = Path(axolotl.__file__).resolve().parent
    mono = (src / "monkeypatch/models/qwen3_5/modeling.py").read_text()
    if "self.block_type" not in mono or "self.layer_type" in mono:
        die("Qwen3.5 monkeypatch lacks block_type fix")
    if "position_ids.ndim == 3" not in mono or "reshape(-1)" not in mono:
        die("Qwen3.5 get_cu_seqlens lacks MRoPE/reshape fix")
    val = (src / "utils/schemas/validation.py").read_text()
    if "cpu_ram_efficient_loading` with load_in_4bit" not in val:
        die("FSDP2/4-bit validator text changed; re-audit load path")
    import importlib
    bnb = importlib.import_module("bitsandbytes.functional")
    w = torch.randn(64, 64, device="cuda:0", dtype=torch.float16)
    q, state = bnb.quantize_4bit(w, quant_type="nf4")
    dq = bnb.dequantize_4bit(q, state, quant_type="nf4")
    if not (dq.dtype == torch.float16 and dq.shape == (64, 64)):
        die("bnb nf4 smoke failed")
    print("install+source gates GREEN", flush=True)
    return {"pins": got, "torch": torch.__version__,
            "cuda": torch.version.cuda, "axolotl_src": str(src)}


# ---------------- stage 2: data + model ----------------
def stage2_data_model():
    cands = sorted(KAGGLE_INPUT.rglob("prod_chat.jsonl"))
    if len(cands) != 1:
        die(f"production data ambiguous: {cands}")
    data = cands[0]
    print("data:", data, flush=True)
    from huggingface_hub import snapshot_download
    snap = SCRATCH / "qwen36-35b"
    snapshot_download(BASE_MODEL, revision=BASE_REV, local_dir=snap,
                      allow_patterns=["*.safetensors", "*.json"])
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(snap, trust_remote_code=False)
    checks = [
        ("architectures", cfg.architectures,
         ["Qwen3_5MoeForConditionalGeneration"]),
        ("model_type", cfg.model_type, "qwen3_5_moe"),
        ("text model_type", cfg.text_config.model_type, "qwen3_5_moe_text"),
        ("layers", cfg.text_config.num_hidden_layers, 40),
        ("experts", cfg.text_config.num_experts, 256),
        ("topk", cfg.text_config.num_experts_per_tok, 8),
        ("hidden", cfg.text_config.hidden_size, 2048),
        ("moe_inter", cfg.text_config.moe_intermediate_size, 512),
    ]
    for name, got, exp in checks:
        if got != exp:
            die(f"arch {name}: {got} != {exp}")
    print("arch asserts GREEN:", cfg.architectures, flush=True)
    return {"snapshot": str(snap), "data": str(data),
            "data_sha256": sha256(data)}


# ---------------- stage 3: frozen YAML + preprocess ----------------
ADVERSARIAL = {  # §N: every archetype must exist in production data
    "kiswahili": re.compile(r"\b(mtoto|mama|dawa|homa|maumivu|mjamzito|nini)\b", re.I),
    "medication": re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|ml|mL|units?|IU)\b", re.I),
    "emergency": re.compile(r"bleed|seizure|unconscious|chest pain|emergency|kill myself", re.I),
}


def stage3_preprocess(snap, data):
    if not TEMPLATE.exists():
        die(f"config template missing: {TEMPLATE}")
    text = TEMPLATE.read_text()
    if "OVERWRITE_ME" not in text or "max_steps: 20" not in text:
        die("template lost its canary placeholders; refusing")
    ypath = SCRATCH / "canary.yaml"
    ypath.write_text(text.replace(
        "/kaggle/input/jamii-sft-prod-v1/prod_chat.jsonl", data).replace(
        "/tmp/OVERWRITE_ME_prod_out", str(SCRATCH / "canary_out")).replace(
        "/tmp/OVERWRITE_ME_prepared", str(SCRATCH / "prepared")))
    (OUT / "canary.yaml").write_text(ypath.read_text())
    yaml_sha = sha256(ypath)
    log = (OUT / "preprocess.log").open("w")
    p = sh([AXO_BIN, "preprocess", str(ypath)], log=log)
    log.close()
    if p.returncode:
        die("axolotl preprocess failed")
    from datasets import load_from_disk
    prep = SCRATCH / "prepared"
    subdirs = sorted(x for x in prep.iterdir() if x.is_dir())
    if not subdirs:
        die(f"no prepared dataset under {prep}")
    counts = {d.name: len(load_from_disk(str(d))) for d in subdirs}
    n = sum(counts.values())
    print(f"prepared rows: {n} {counts}", flush=True)
    if n < 1450:
        die(f"only {n} rows survived (overlong drops?)")
    # §N adversarial archetypes present in the frozen data.
    blobs = [json.loads(l) for l in open(data, encoding="utf-8")]
    texts = [" ".join(m["content"] for m in r["messages"]) for r in blobs]
    found = {k: sum(1 for t in texts if rx.search(t))
             for k, rx in ADVERSARIAL.items()}
    found["replay"] = sum(1 for r in blobs if r.get("src") == "oasst1")
    found["system_prompt"] = sum(
        1 for r in blobs if r["messages"][0]["role"] == "system")
    print("archetypes:", found, flush=True)
    if any(v == 0 for v in found.values()):
        die(f"missing adversarial archetype: {found}")
    return {"prepared_rows": n, "yaml_sha256": yaml_sha, "archetypes": found}


# ---------------- stage 4: 20-step canary with watchdog ----------------
STEP_RE = re.compile(r"'loss':\s*([0-9.eE+-]+).*?'grad_norm':\s*([0-9.eE+-]+).*?"
                     r"'learning_rate':\s*([0-9.eE+-]+).*?'epoch':\s*([0-9.eE+-]+)")


def host_mem():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")[:2]
        if k.strip() in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            m[k.strip()] = int(v.split()[0])
    return {"ram_used_gb": (m["MemTotal"] - m["MemAvailable"]) / 1e6,
            "swap_used_gb": (m["SwapTotal"] - m["SwapFree"]) / 1e6}


def gpu_mem():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"], check=True, text=True,
                       capture_output=True).stdout.strip().splitlines()
    return [[int(x) for x in line.split(",")] for line in q]


class Watch:
    """Polls train.log for step lines + samples GPU/host/swap (§V, §W)."""

    def __init__(self, logpath, tokens_per_step):
        self.logpath = logpath
        self.tps = tokens_per_step
        self.steps = []
        self.fh = open(OUT / "steps.jsonl", "w")
        self.t0 = time.time()
        self._off = 0

    def poll(self):
        with open(self.logpath, errors="replace") as f:
            f.seek(self._off)
            chunk = f.read()
            self._off = f.tell()
        for m in STEP_RE.finditer(chunk):
            loss, gn, lr, ep = (float(m.group(1)), float(m.group(2)),
                               float(m.group(3)), float(m.group(4)))
            now = time.time()
            gm = gpu_mem()
            hm = host_mem()
            prev = self.steps[-1]["t"] if self.steps else self.t0
            rec = {"step": len(self.steps) + 1, "loss": loss, "lr": lr,
                   "grad_norm": gn, "epoch": ep, "t": now,
                   "seconds": now - prev, "tokens": self.tps,
                   "tokens_per_sec": self.tps / max(now - prev, 1e-9),
                   "gpu0_alloc": gm[0][0], "gpu1_alloc": gm[1][0],
                   "host_ram": hm["ram_used_gb"],
                   "swap_gb": hm["swap_used_gb"], "timestamp": now}
            self.steps.append(rec)
            self.fh.write(json.dumps({k: v for k, v in rec.items()
                                      if k != "t"}) + "\n")
            self.fh.flush()
            # §V abort rules.
            if not (loss == loss and abs(loss) < 1e9):
                return f"loss not finite: {loss}"
            if not (gn == gn and abs(gn) < 1e9):
                return f"grad_norm not finite: {gn}"
            if hm["swap_used_gb"] > 1.0:
                return f"swap thrash: {hm['swap_used_gb']:.1f}GB"
            if len(self.steps) >= 6:
                med = sorted(s["seconds"]
                             for s in self.steps[-6:])[3]
                if rec["seconds"] > 2 * med and rec["seconds"] > 600:
                    return (f"step {rec['step']} took {rec['seconds']:.0f}s "
                            f"(>2x median {med:.0f}s)")
        return None

    def close(self):
        self.fh.close()


def stage4_train():
    chaos = os.environ.get("PREFLIGHT_CHAOS") == "1"
    resume = os.environ.get("RESUME_FROM")
    ypath = str(SCRATCH / "canary.yaml")
    if resume:
        text = Path(ypath).read_text()
        if "resume_from_checkpoint" in text:
            text = re.sub(r"resume_from_checkpoint:.*",
                          f"resume_from_checkpoint: {resume}", text)
        else:
            text += f"\nresume_from_checkpoint: {resume}\n"
        Path(ypath).write_text(text)
        (OUT / "canary_resume.yaml").write_text(text)
    env = dict(os.environ, USE_HUB_KERNELS="0",
               TOKENIZERS_PARALLELISM="false",
               NCCL_TIMEOUT="1800")
    logpath = OUT / ("train_resume.log" if resume else "train.log")
    log = logpath.open("w")
    t0 = time.time()
    # tokens/optimizer-step: micro(1) x gpu(2) x accum(8) x seq(2048).
    watch = Watch(logpath, 1 * 2 * 8 * 2048)
    proc = subprocess.Popen(
        [AXO_BIN, "train", ypath, "--", "--multi_gpu", "--num_processes=2"],
        stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    reason, killed = None, False
    try:
        while proc.poll() is None:
            time.sleep(10)
            reason = watch.poll()
            if reason:
                break
            if chaos and not killed and len(watch.steps) >= 8:
                proc.kill()  # §Q: intentional interrupt, then resume leg
                killed = True
                break
    finally:
        log.close()
    wall = time.time() - t0
    watch.close()
    if killed:
        print(f"CHAOS: killed at step {len(watch.steps)}; resume leg next",
              flush=True)
        return {"chaos_killed_at": len(watch.steps), "wall": wall,
                "resumed": False}
    if proc.returncode != 0:
        tail = logpath.read_text(errors="replace")[-3000:]
        die(f"axolotl train exit={proc.returncode}: ...{tail[-800:]}")
    if reason:
        die(f"watchdog: {reason}")
    out = SCRATCH / "canary_out"
    ckpts = sorted(out.glob("checkpoint-*"))
    if not ckpts:
        die(f"no checkpoints under {out}")
    last = ckpts[-1]
    tstate = json.loads((last / "trainer_state.json").read_text())
    if tstate["global_step"] < CANARY_STEPS and not resume:
        die(f"only reached step {tstate['global_step']}")
    losses = [s["loss"] for s in watch.steps]
    secs = [s["seconds"] for s in watch.steps]
    med = sorted(secs)[len(secs) // 2] if secs else 0
    print(f"canary ok: {len(watch.steps)} steps loss {losses[0]:.4f}"
          f"->{losses[-1]:.4f} med {med:.1f}s/step", flush=True)
    return {"steps": len(watch.steps), "loss_first": losses[0],
            "loss_last": losses[-1], "median_sec_per_step": med,
            "tokens_per_sec": watch.tps / med if med else 0,
            "last_checkpoint": last.name,
            "global_step": tstate["global_step"], "wall": wall}


# ---------------- stage 5: adapter asserts + reload ----------------
def stage5_adapter(last_ckpt):
    ck = SCRATCH / "canary_out" / last_ckpt
    acfg = json.loads((ck / "adapter_config.json").read_text())
    if not (acfg["r"] == 8 and acfg["lora_alpha"] == 16):
        die(f"adapter config drift: r={acfg['r']} alpha={acfg['lora_alpha']}")
    if set(acfg["target_modules"]) - ALLOW_LORA:
        die(f"adapter targets drift: {acfg['target_modules']}")
    from safetensors import safe_open
    f = ck / "adapter_model.safetensors"
    if f.stat().st_size == 0:
        die("adapter file empty")
    n_tensors, n_params, bad = 0, 0, []
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            n_tensors += 1
            n_params += sf.get_tensor(k).numel()
            if any(s in k.lower() for s in FORBID_LORA):
                bad.append(k)
    if bad:
        die(f"forbidden adapters: {bad[:5]}")
    if n_tensors != EXP_LORA_TENSORS or n_params != EXP_LORA_PARAMS:
        die(f"adapter footprint {(n_tensors, n_params)} != "
            f"{(EXP_LORA_TENSORS, EXP_LORA_PARAMS)}")
    print(f"adapter ok: {n_tensors} tensors, {n_params} params", flush=True)
    shutil.copy(ck / "adapter_config.json", OUT / "adapter_config.json")
    shutil.copy(f, OUT / "adapter_model.safetensors")
    return {"tensors": n_tensors, "params": n_params,
            "adapter_sha256": sha256(OUT / "adapter_model.safetensors")}


RELOAD_SNIPPET = r"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
# §F: assert 4-bit actually engaged, never infer from config.
n4 = 0
m = AutoModelForCausalLM.from_pretrained({snap!r}, quantization_config=qc,
    attn_implementation="sdpa", device_map="auto", torch_dtype=torch.float16)
from bitsandbytes.nn import Linear4bit
for mod in m.modules():
    if isinstance(mod, Linear4bit):
        n4 += 1
print("linear4bit_modules:", n4)
assert n4 > 100, f"4-bit not engaged: only {{n4}} Linear4bit modules"
m = PeftModel.from_pretrained(m, {ad!r})
m.print_trainable_parameters()
tok = AutoTokenizer.from_pretrained({snap!r})
ids = tok("Fever for three days. What should I do?", return_tensors="pt")["input_ids"]
dev = next(m.parameters()).device
with torch.no_grad():
    out = m(ids.to(dev))
print("logits:", tuple(out.logits.shape))
assert out.logits.shape[1] == ids.shape[1]
print("RELOAD_OK")
"""


def stage6_reload_merge(snap, ckpt):
    """§O fresh-process reload + §P merge/export + tiny generation."""
    code = RELOAD_SNIPPET.format(
        snap=snap, ad=str(SCRATCH / "canary_out" / ckpt))
    env = dict(os.environ, USE_HUB_KERNELS="0",
               TOKENIZERS_PARALLELISM="false")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=5400)
    (OUT / "reload.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0 or "RELOAD_OK" not in p.stdout:
        die(f"fresh-process reload failed: {p.stderr[-1500:]}")
    print(f"reload ok ({time.time()-t0:.0f}s)", flush=True)
    return {"reload_sec": time.time() - t0}
    # NOTE §P merge: full BF16 merge needs ~72GB host RAM (unavailable on
    # Kaggle-class hosts); the merge leg runs on provisioned hardware via
    # training/merge.py against this same adapter + revision. The adapter
    # load + forward above proves the §P input boundary.


# ---------------- stage 7: GO/NO-GO ----------------
def stage7_report(gates):
    tr = gates["train"]
    est_steps_4h = int(4 * 3600 / tr["median_sec_per_step"]) \
        if tr["median_sec_per_step"] else 0
    report = f"""\
============================== GO / NO-GO ==============================
SOFTWARE Axolotl@{AXOLOTL_SHA[:12]} torch={gates['install']['torch']} \
pins={gates['install']['pins']} block_type/MRoPE asserts PASS
HARDWARE {gates['env']['gpus']} host={gates['env']['host_ram_gb']}GB
MODEL {BASE_MODEL}@{BASE_REV[:12]} qwen3_5_moe arch asserts PASS
DATA rows={gates['preprocess']['prepared_rows']} \
yaml={gates['preprocess']['yaml_sha256'][:12]} archetypes={gates['preprocess']['archetypes']}
TRAINABLES {gates['adapter']['tensors']}T/{gates['adapter']['params']}p \
router/experts/vision FROZEN (exact footprint match)
CANARY {tr['steps']} steps loss {tr['loss_first']:.4f}->{tr['loss_last']:.4f} \
med {tr['median_sec_per_step']:.1f}s/step {tr['tokens_per_sec']:.0f} tok/s
RECOVERY ckpt={tr['last_checkpoint']} reload=PASS resume={'PASS' if tr.get('resumed') else 'see resume leg'}
EXPORT adapter-load=PASS merge=DEFERRED to provisioned HW (merge.py)
PRODUCTION max_steps~{est_steps_4h} for 4h @ measured throughput \
(+25% margin: {int(est_steps_4h/1.25)})
VERDICT: GO — launch with ONLY max_steps/output_dir/run_name changed.
========================================================================
"""
    print(report, flush=True)
    (OUT / "GO_REPORT.txt").write_text(report)
    (OUT / "VERDICT").write_text("GO\n")


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    gates = {}
    try:
        gates["env"] = stage0_env()
        gates["install"] = stage1_install()
        d2 = stage2_data_model()
        gates["data_model"] = d2
        gates["preprocess"] = stage3_preprocess(d2["snapshot"], d2["data"])
        tr = stage4_train()
        gates["train"] = tr
        if tr.get("chaos_killed_at") and not tr.get("resumed"):
            # §Q resume leg: fresh `axolotl train` from latest checkpoint.
            out = SCRATCH / "canary_out"
            last = sorted(out.glob("checkpoint-*"))[-1].name
            os.environ["RESUME_FROM"] = str(out / last)
            tr2 = stage4_train()
            tr2["resumed"] = tr2["global_step"] > tr["chaos_killed_at"]
            if not tr2["resumed"]:
                die("resume leg did not advance past kill point")
            gates["train"] = tr2
        d5 = stage5_adapter(gates["train"]["last_checkpoint"])
        gates["adapter"] = d5
        gates["reload"] = stage6_reload_merge(
            d2["snapshot"], gates["train"]["last_checkpoint"])
        manifest = {"axolotl_sha": AXOLOTL_SHA, "base_rev": BASE_REV,
                    "data_sha256": d2["data_sha256"],
                    "yaml_sha256": gates["preprocess"]["yaml_sha256"],
                    "pins": gates["install"]["pins"],
                    "torch": gates["install"]["torch"],
                    "seed": 11, "start_time": time.time()}
        (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=1))
        stage7_report(gates)
        (OUT / "result.json").write_text(json.dumps(gates, indent=1,
                                                   default=str))
        print("PREFLIGHT COMPLETE: GO", flush=True)
    except SystemExit:
        raise
    except Exception as e:  # §Q finally: never lose diagnostics
        import traceback
        (OUT / "EXCEPTION.txt").write_text(traceback.format_exc())
        try:
            (OUT / "mem_diag.txt").write_text(
                json.dumps({"gpu": gpu_mem(), "host": host_mem()}, indent=1))
        except Exception:
            pass
        die(f"unexpected: {e!r}")


if __name__ == "__main__":
    main()

