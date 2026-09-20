"""QLoRA feasibility CANARY on 2x Kaggle T4 (free tier). STOPS after 10 steps.

Gates, in order (fail-loud, no silent fallback):
  0. env asserts: python, torch 2.11-2.14, CUDA, exactly 2x T4 sm75
  1. install axolotl @ pinned commit; verify dep pins; bnb CUDA smoke test
  2. data sha check + base snapshot @ pinned revision + arch assert
  3. write canary YAML (strict) + `axolotl preprocess` gate (>=1450 rows kept)
  4. `axolotl train` max_steps=10 with nvidia-smi sampler
  5. adapter verify: EXACT 80 tensors / 1,392,640 params, q/k/v/o only
  6. reload test in a FRESH process: base 4-bit + adapter, one forward
  7. report: peaks, sec/step, pilot estimate

Writes OUT/result.json + canary.yaml + adapter copy. Never launches the pilot.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/native-sparse-qlora-canary-v1")
OUT = WORK / "native-sparse-qlora-canary-v1-results"
KAGGLE_INPUT = Path("/kaggle/input")

AXOLOTL_SHA = "82109efa69a147853253186e4c314cddac66ab59"
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
BASE_REV = "995ad96eacd98c81ed38be0c5b274b04031597b0"
PILOT_SHA = "062c1d27a2fff45222c731901b3e4e9aadc63e697c586c1f7222092e54904895"

# Expected LoRA footprint: q/k/v/o on the 10 full-attention layers only.
# q: 8*(2048+4096), k: 8*(2048+512), v: 8*(2048+512), o: 8*(4096+2048)
EXP_LORA_PARAMS = (8 * (2048 + 4096) + 8 * (2048 + 512) * 2
                   + 8 * (4096 + 2048)) * 10
EXP_LORA_TENSORS = 10 * 4 * 2
AXO_BIN = "axolotl"  # resolved to an absolute path in stage1


def sh(cmd, log=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    if log is None:
        return subprocess.run(cmd, check=True, text=True)
    return subprocess.run(cmd, check=True, text=True, stdout=log,
                          stderr=subprocess.STDOUT)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(8 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def stage0_env():
    # No torch import here: pip may replace torch in stage1, and this process
    # must import it exactly once, after the install.
    assert sys.version_info >= (3, 10), sys.version
    assert shutil.which("nvidia-smi"), "no nvidia-smi"
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap",
                        "--format=csv,noheader"], check=True, text=True,
                       capture_output=True).stdout.strip().splitlines()
    assert len(q) == 2, f"need exactly 2 GPUs: {q}"
    print("gpus:", q, flush=True)
    assert all("T4" in line for line in q), q
    assert all("7.5" in line for line in q), q
    du = shutil.disk_usage("/tmp")
    print(f"disk /tmp: free={du.free / 1e9:.0f}GB total={du.total / 1e9:.0f}GB",
          flush=True)
    # 72GB base snapshot + ~10GB env + prepared/checkpoints + slack.
    assert du.free >= 90e9, f"only {du.free / 1e9:.0f}GB free on /tmp"
    return {"gpus": q, "tmp_free_gb": round(du.free / 1e9, 1)}


def stage1_install():
    log = (OUT / "install.log").open("w")
    sh([sys.executable, "-m", "pip", "install",
        f"axolotl @ git+https://github.com/axolotl-ai-cloud/axolotl.git@{AXOLOTL_SHA}"],
       log=log)
    log.close()
    # Kaggle image ships boto3/botocore; axolotl's deps downgrade botocore,
    # leaving boto3 unimportable — and accelerate imports boto3 (via its
    # sagemaker-config module) whenever transformers loads. Pin boto3 to the
    # installed botocore (they share version numbers; neither is used here).
    bcv = subprocess.run(
        [sys.executable, "-c", "import botocore; print(botocore.__version__)"],
        check=True, text=True, capture_output=True).stdout.strip()
    log2 = (OUT / "boto_fix.log").open("w")
    sh([sys.executable, "-m", "pip", "install", "-q", f"boto3=={bcv}"],
       log=log2)
    log2.close()
    subprocess.run([sys.executable, "-c", "import boto3"], check=True)
    print(f"boto3 pinned to botocore {bcv}", flush=True)
    # Kaggle image ships torchvision/torchaudio built for its old torch;
    # axolotl's newer torch leaves their C++ ops unloadable, which poisons
    # `import transformers`. But `accelerate launch` needs torchvision
    # (via timm), so reinstall a matching build instead of removing it;
    # torchaudio is unused and stays out.
    def py(code):
        return subprocess.run(
            [sys.executable, "-c", code], check=True, text=True,
            capture_output=True).stdout.strip()
    torch_before = py("import torch; print(torch.__version__)")
    log3 = (OUT / "tv_fix.log").open("w")
    sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision",
        "torchaudio"], log=log3)
    sh([sys.executable, "-m", "pip", "install", "-q", "torchvision"],
       log=log3)
    log3.close()
    torch_after = py("import torch, torchvision, torchvision.ops;"
                     " print(torch.__version__)")
    assert torch_after == torch_before, (torch_before, torch_after)
    print(f"torchvision rebuilt for torch {torch_after}", flush=True)
    global AXO_BIN
    AXO_BIN = (shutil.which("axolotl")
               or str(Path(sys.prefix) / "bin" / "axolotl"))
    assert Path(AXO_BIN).exists(), f"axolotl binary missing: {AXO_BIN}"
    print("axolotl:", AXO_BIN, flush=True)
    import torch  # first import in this process: post-install version
    parts = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
    assert (2, 11, 0) <= parts <= (2, 14, 99), f"torch {torch.__version__}"
    assert torch.cuda.is_available(), "no CUDA after install"
    assert torch.cuda.device_count() == 2, torch.cuda.device_count()
    import transformers, peft, trl, bitsandbytes, datasets
    pins = {"transformers": "5.17.0", "peft": "0.21.0", "trl": "1.13.0",
            "bitsandbytes": "0.50.2", "datasets": "4.8.4"}
    got = {"transformers": transformers.__version__, "peft": peft.__version__,
           "trl": trl.__version__, "bitsandbytes": bitsandbytes.__version__,
           "datasets": datasets.__version__}
    print("pins:", got, flush=True)
    assert got == pins, f"pin drift: {got}"
    # bnb 4-bit CUDA smoke on a T4 (fail here, not 40 min into loading).
    import importlib
    bnb = importlib.import_module("bitsandbytes.functional")
    import torch
    w = torch.randn(64, 64, device="cuda:0", dtype=torch.float16)
    q, state = bnb.quantize_4bit(w, quant_type="nf4")
    dq = bnb.dequantize_4bit(q, state, quant_type="nf4")
    assert dq.dtype == torch.float16 and dq.shape == (64, 64)
    print("bnb nf4 smoke ok", flush=True)
    got["torch"] = torch.__version__
    return got


def stage2_data_model():
    # Mount layout varies (/kaggle/input/<slug> vs /kaggle/input/datasets/...):
    # discover by filename, then verify by SHA.
    cands = sorted(KAGGLE_INPUT.rglob("pilot_chat.jsonl"))
    assert len(cands) == 1, f"pilot file ambiguous: {cands}"
    f = cands[0]
    print("data:", f, flush=True)
    digest = sha256(f)
    assert digest == PILOT_SHA, f"data drift: {digest}"
    from huggingface_hub import snapshot_download
    snap = SCRATCH / "qwen36-35b"
    snapshot_download(BASE_MODEL, revision=BASE_REV, local_dir=snap,
                      allow_patterns=["*.safetensors", "*.json"])
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(snap, trust_remote_code=False)
    assert cfg.model_type == "qwen3_5_moe", cfg.model_type
    assert cfg.text_config.num_hidden_layers == 40
    assert cfg.text_config.vocab_size == 248320
    print("arch ok:", cfg.architectures, flush=True)
    return {"snapshot": str(snap), "data": str(f)}


CANARY_YAML = """\
base_model: {snap}
chat_template: qwen3_5
strict: true

datasets:
  - path: {data}
    type: chat_template
    field_messages: messages
val_set_size: 0.0
output_dir: {out}
dataset_prepared_path: {prep}

sequence_len: 2048
sample_packing: false
pad_to_sequence_len: false

load_in_4bit: true
quantize_moe_experts: true
adapter: qlora
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
lora_mlp_kernel: false
lora_qkv_kernel: false
lora_o_kernel: false

gradient_accumulation_steps: 1
micro_batch_size: 1
max_steps: 10
num_epochs: 1
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 0.0002
warmup_ratio: 0.1
weight_decay: 0.0
train_on_inputs: false

bf16: false
fp16: true
tf32: false
attn_implementation: sdpa

gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

logging_steps: 1
save_steps: 10

fsdp_config:
  fsdp_version: 2
  offload_params: true
  cpu_ram_efficient_loading: false
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Qwen3_5MoeDecoderLayer
  state_dict_type: FULL_STATE_DICT
  reshard_after_forward: true
  activation_checkpointing: true
"""


def stage3_preprocess(snap, data):
    ypath = SCRATCH / "canary.yaml"
    ypath.write_text(CANARY_YAML.format(
        snap=snap, data=data,
        out=SCRATCH / "canary_out", prep=SCRATCH / "prepared"))
    (OUT / "canary.yaml").write_text(ypath.read_text())
    log = (OUT / "preprocess.log").open("w")
    t0 = time.time()
    sh([AXO_BIN, "preprocess", str(ypath)], log=log)
    log.close()
    # Prepared data lands in dataset_prepared_path/<config-hash>/ (one subdir
    # per split), not directly in dataset_prepared_path.
    from datasets import load_from_disk
    prep = SCRATCH / "prepared"
    subdirs = sorted(p for p in prep.iterdir() if p.is_dir())
    assert subdirs, f"no prepared dataset under {prep}: {sorted(prep.iterdir())}"
    counts = {}
    for d in subdirs:
        counts[d.name] = len(load_from_disk(str(d)))
    n = sum(counts.values())
    print(f"prepared rows: {n} {counts} ({time.time() - t0:.0f}s)", flush=True)
    assert n >= 1450, f"only {n} rows survived (overlong drops?)"
    return {"prepared_rows": n, "prepared_splits": counts,
            "yaml_sha256": sha256(ypath)}


class Sampler:
    def __init__(self):
        self.proc = None
        self.log = OUT / "nvidia_smi.log"

    def start(self):
        fh = open(self.log, "w")
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,"
             "utilization.gpu", "--format=csv,noheader,nounits", "-l", "5"],
            stdout=fh, stderr=subprocess.DEVNULL, text=True)
        self.mem0 = self.host_gb()

    def host_gb(self):
        m = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":")[:2]
            if k in ("MemTotal", "MemAvailable"):
                m[k] = int(v.split()[0])
        return {"total": m["MemTotal"] / 1e6,
                "used": (m["MemTotal"] - m["MemAvailable"]) / 1e6}

    def stop(self):
        self.proc.terminate()
        self.proc.wait()
        peaks = {}
        for line in open(self.log):
            try:
                idx, used, total, util = [x.strip() for x in line.split(",")]
                peaks[idx] = max(peaks.get(idx, 0), int(used))
            except ValueError:
                pass
        host = self.host_gb()
        return {"gpu_peak_mb": peaks, "host_gb": host, "host0_gb": self.mem0}


def stage4_train():
    env = dict(os.environ, USE_HUB_KERNELS="0",
               TOKENIZERS_PARALLELISM="false")
    smp = Sampler()
    smp.start()
    log = (OUT / "train.log").open("w")
    t0 = time.time()
    try:
        # Explicit 2-worker launch (args after -- reach accelerate verbatim).
        p = subprocess.run([AXO_BIN, "train", str(SCRATCH / "canary.yaml"),
                            "--", "--multi_gpu", "--num_processes=2"],
                           stdout=log, stderr=subprocess.STDOUT, text=True,
                           env=env)
    finally:
        log.close()
    wall = time.time() - t0
    mem = smp.stop()
    assert p.returncode == 0, f"axolotl train exit={p.returncode} (see train.log)"
    out = SCRATCH / "canary_out"
    res = json.loads((out / "all_results.json").read_text())
    ck10 = out / "checkpoint-10"
    assert ck10.is_dir(), f"no checkpoint-10: {sorted(out.iterdir())}"
    tstate = json.loads((ck10 / "trainer_state.json").read_text())
    assert tstate["global_step"] == 10, f"steps != 10: {tstate['global_step']}"
    loss = res.get("train_loss")
    assert loss is not None and loss == loss and loss < 1e4, f"bad loss {loss}"
    runtime = res.get("train_runtime", wall)
    print(f"train ok: loss={loss:.4f} runtime={runtime:.0f}s "
          f"({runtime / 10:.1f}s/step)", flush=True)
    return {"loss": loss, "runtime": runtime, "sec_per_step": runtime / 10,
            "wall": wall, **mem, "results": res}


def stage5_adapter():
    out = SCRATCH / "canary_out"
    ck = out / "checkpoint-10"
    assert ck.is_dir(), f"no checkpoint-10: {sorted(out.iterdir())}"
    acfg = json.loads((ck / "adapter_config.json").read_text())
    assert acfg["r"] == 8 and acfg["lora_alpha"] == 16, acfg
    tgts = acfg["target_modules"]
    assert set(tgts) <= {"q_proj", "k_proj", "v_proj", "o_proj"}, tgts
    from safetensors import safe_open
    f = ck / "adapter_model.safetensors"
    n_tensors, n_params = 0, 0
    bad = []
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            n_tensors += 1
            t = sf.get_tensor(k)
            n_params += t.numel()
            kl = k.lower()
            if any(s in kl for s in ("mlp", "router", "gate", "expert",
                                     "embed", "head", "norm", "mtp")):
                bad.append(k)
    assert not bad, f"forbidden adapters: {bad[:5]}"
    assert n_tensors == EXP_LORA_TENSORS, (n_tensors, EXP_LORA_TENSORS)
    assert n_params == EXP_LORA_PARAMS, (n_params, EXP_LORA_PARAMS)
    print(f"adapter ok: {n_tensors} tensors, {n_params} params", flush=True)
    shutil.copy(ck / "adapter_config.json", OUT / "adapter_config.json")
    shutil.copy(f, OUT / "adapter_model.safetensors")
    return {"checkpoint": ck.name, "tensors": n_tensors, "params": n_params,
            "adapter_sha256": sha256(OUT / "adapter_model.safetensors")}


RELOAD_SNIPPET = r"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
m = AutoModelForCausalLM.from_pretrained({snap!r}, quantization_config=qc,
    attn_implementation="sdpa", device_map="auto", torch_dtype=torch.float16)
m = PeftModel.from_pretrained(m, {ad!r})
tok = AutoTokenizer.from_pretrained({snap!r})
ids = tok("Fever for three days. What should I do?", return_tensors="pt")["input_ids"]
dev = next(m.parameters()).device
with torch.no_grad():
    out = m(ids.to(dev))
print("logits:", tuple(out.logits.shape))
assert out.logits.shape[1] == ids.shape[1] and out.logits.shape[2] == 248320
print("RELOAD_OK")
"""


def stage6_reload(snap, ckpt):
    code = RELOAD_SNIPPET.format(snap=snap,
                                ad=str(SCRATCH / "canary_out" / ckpt))
    env = dict(os.environ, USE_HUB_KERNELS="0",
               TOKENIZERS_PARALLELISM="false")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=3600)
    (OUT / "reload.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    assert p.returncode == 0 and "RELOAD_OK" in p.stdout, \
        f"reload failed (see reload.log): {p.stderr[-2000:]}"
    print(f"reload ok ({time.time() - t0:.0f}s)", flush=True)
    return {"reload_sec": time.time() - t0}


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    gates = {}
    gates["env"] = stage0_env()
    gates["install"] = stage1_install()
    d2 = stage2_data_model()
    gates["data_model"] = d2
    gates["preprocess"] = stage3_preprocess(d2["snapshot"], d2["data"])
    gates["train"] = stage4_train()
    d5 = stage5_adapter()
    gates["adapter"] = d5
    gates["reload"] = stage6_reload(d2["snapshot"], d5["checkpoint"])
    sps = gates["train"]["sec_per_step"]
    # pilot estimate: 1465 rows, eff batch 16 (1*2gpu*8accum) => ~92 steps
    gates["pilot_estimate"] = {
        "steps_per_epoch_92": 92 * sps,
        "note": "92 steps * measured s/step + load/save overhead, est only"}
    (OUT / "result.json").write_text(json.dumps(gates, indent=1))
    print("CANARY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
