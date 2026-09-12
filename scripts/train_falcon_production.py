#!/usr/bin/env python3
"""Observable, resumable staged Falcon-H1 LoRA training.

This is deliberately separate from the historical ``train_lora.py`` probe.
It consumes the normalized JSONL files made by ``build_falcon_dataset.py`` and
fails closed on sequence/label errors.  Prompt labels are always -100; only
assistant target tokens (including EOS) contribute to SFT loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import selectors
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def event(path: Path, name: str, **fields: Any) -> None:
    record = {"timestamp_utc": now(), "event": name, **fields}
    print("[{}] {} {}".format(record["timestamp_utc"], name, json.dumps(fields, sort_keys=True)), flush=True)
    append_jsonl(path, record)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(row)
    return rows


def render_sft(tokenizer: Any, row: dict[str, Any], system: str) -> tuple[list[int], list[int]]:
    instruction = str(row.get("instruction") or "").strip()
    answer = str(row.get("output") or "").strip()
    if not instruction or not answer:
        raise ValueError(f"{row.get('example_id', '?')}: empty SFT instruction/answer")
    user = instruction
    if str(row.get("input") or "").strip():
        user += "\n\n" + str(row["input"]).strip()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    target_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    target_ids.append(int(tokenizer.eos_token_id))
    return prompt_ids, target_ids


class FalconDataset:
    """Pre-tokenized fail-closed mixed SFT/MCQA dataset."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_len: int, system: str):
        self.items: list[dict[str, Any]] = []
        self.rejected: list[dict[str, str]] = []
        for row in rows:
            try:
                fmt = row.get("format")
                if fmt == "sft":
                    prompt, target = render_sft(tokenizer, row, system)
                    if len(target) > max_len:
                        raise ValueError(f"target length {len(target)} exceeds max_len={max_len}")
                    prompt = prompt[-max(0, max_len - len(target)):]
                    ids = prompt + target
                    if len(ids) > max_len or not target:
                        raise ValueError("invalid prompt/target packing")
                    self.items.append({
                        "kind": "sft", "input_ids": ids,
                        "labels": [-100] * len(prompt) + target,
                        # ``tokens`` is the number of tokens that contribute to
                        # the objective.  Prompt tokens are intentionally not
                        # used for token-share sampling.
                        "tokens": len(target), "sample_tokens": int(row.get("loss_tokens", len(target))), "example_id": row["example_id"],
                        "source": row.get("source", ""),
                    })
                elif fmt == "mcqa":
                    context_ids = list(tokenizer(str(row["context"]), add_special_tokens=False)["input_ids"])
                    choice_ids = [list(tokenizer(" " + str(x), add_special_tokens=False)["input_ids"]) for x in row["choices"]]
                    if any(len(context_ids) + len(choice) > max_len for choice in choice_ids):
                        raise ValueError("MCQA choice would be truncated")
                    self.items.append({
                        "kind": "mcqa", "context_ids": context_ids,
                        "choice_ids": choice_ids, "choices": row["choices"],
                        "gold": int(row["gold"]),
                        "tokens": sum(len(x) for x in choice_ids), "sample_tokens": int(row.get("loss_tokens", sum(len(x) for x in choice_ids))),
                        "example_id": row["example_id"], "source": row.get("source", ""),
                    })
                else:
                    raise ValueError(f"unsupported format {fmt!r}")
            except (KeyError, TypeError, ValueError) as exc:
                self.rejected.append({"example_id": str(row.get("example_id", "?")), "reason": str(exc)})
        if self.rejected:
            raise ValueError(f"{len(self.rejected)} rows failed training tokenization; rebuild data and inspect rejection report")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def identity_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


class Progress:
    def __init__(self, path: Path, interval: int):
        self.path = path
        self.interval = max(10, interval)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.started = time.monotonic()
        self.last_step = 0
        self.microstep = 0
        self.last_loss = None
        self.last_tokens = 0
        self.seen_tokens = 0
        self.seen_examples = 0
        self.total_steps: int | None = None
        self.thread = threading.Thread(target=self._loop, name="heartbeat", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)

    def update(self, *, microstep: int | None = None, step: int | None = None, loss: float | None = None, tokens: int | None = None, examples: int | None = None) -> None:
        with self.lock:
            if microstep is not None:
                self.microstep = microstep
            if step is not None:
                self.last_step = step
            if loss is not None:
                self.last_loss = loss
            if tokens is not None:
                self.last_tokens = tokens
                self.seen_tokens += tokens
            if examples is not None:
                self.seen_examples += examples

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval):
            with self.lock:
                record = {
                    "timestamp_utc": now(), "event": "HEARTBEAT", "stage": "train",
                    "elapsed_seconds": round(time.monotonic() - self.started, 1),
                    "optimizer_step": self.last_step, "microstep": self.microstep,
                    "last_loss": self.last_loss, "last_batch_tokens": self.last_tokens,
                    "seen_tokens": self.seen_tokens, "seen_examples": self.seen_examples,
                }
                elapsed = max(1e-6, time.monotonic() - self.started)
                record["tokens_per_second"] = round(self.seen_tokens / elapsed, 3)
                record["examples_per_second"] = round(self.seen_examples / elapsed, 3)
                if self.total_steps and self.last_step:
                    record["eta_seconds"] = round(max(0, self.total_steps - self.last_step) * elapsed / self.last_step, 1)
                try:
                    import torch
                    if torch.cuda.is_available():
                        record["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
                        record["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
                except Exception:  # noqa: BLE001 - heartbeat must never kill training
                    pass
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        text=True, capture_output=True, check=False, timeout=3,
                    )
                    value = result.stdout.strip().splitlines()[0]
                    record["gpu_utilization_percent"] = float(value)
                except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
                    pass
            print("[{}] HEARTBEAT {}".format(record["timestamp_utc"], json.dumps({k: v for k, v in record.items() if k not in {"timestamp_utc", "event"}}, sort_keys=True)), flush=True)
            append_jsonl(self.path, record)


def checkpoint_is_complete(path: Path, *, require_scaler: bool = False) -> bool:
    """Return true only for a checkpoint that can actually be resumed.

    ``trainer_state.json`` alone is not enough: an interrupted save can leave
    that file behind before optimizer/scheduler/RNG state is durable.  Keep the
    resume selector conservative and fail closed.
    """
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
    if not path.is_dir() or not any(candidate.is_file() for candidate in path.glob("adapter_model.*")):
        return False
    if not all((path / name).is_file() for name in required):
        return False
    try:
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(state.get("global_step"), int):
        return False
    manifest_path = path / "checkpoint_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            return False
        if manifest.get("global_step") != state["global_step"]:
            return False
        require_scaler = require_scaler or manifest.get("scaler_required") is True
    return not require_scaler or (path / "scaler.pt").is_file()


def latest_checkpoint(directory: Path) -> Path | None:
    candidates = []
    for path in directory.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        try:
            complete = checkpoint_is_complete(path)
        except (OSError, ValueError, json.JSONDecodeError):
            complete = False
        if complete:
            candidates.append((step, path))
    return max(candidates, default=(0, None))[1]


def token_share_sampling_weights(items: list[dict[str, Any]], shares: dict[str, float]) -> list[float]:
    """Return row weights whose expected *loss-token* exposure matches shares.

    For objective ``k``, each row receives ``share[k] / total_loss_tokens[k]``.
    The sum of ``weight * loss_tokens`` is therefore exactly the requested
    share before sampling variance.  Using ``1 / row_tokens`` here would
    accidentally make the result depend on row count and favor short rows.
    """
    totals: Counter[str] = Counter()
    for item in items:
        totals[item["kind"]] += max(1, int(item.get("tokens", 1)))
    return [
        float(shares.get(item["kind"], 0.0)) / max(1, totals[item["kind"]])
        for item in items
    ]


def run_streamed(command: list[str], log_path: Path) -> int:
    """Run a potentially slow child process with line-buffered timestamped output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    merged["PYTHONUNBUFFERED"] = "1"
    print("STREAM " + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=merged,
        )
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        started = time.monotonic()
        try:
            while True:
                ready = selector.select(timeout=30)
                if ready:
                    line = proc.stdout.readline()
                    if line:
                        rendered = f"[{now()}] {line}"
                        print(rendered, end="", flush=True)
                        handle.write(rendered)
                        handle.flush()
                    elif proc.poll() is not None:
                        break
                else:
                    heartbeat = f"[{now()}] HEARTBEAT child=persistence elapsed={time.monotonic() - started:.1f}s\n"
                    print(heartbeat, end="", flush=True)
                    handle.write(heartbeat)
                    handle.flush()
                if proc.poll() is not None:
                    for line in proc.stdout:
                        rendered = f"[{now()}] {line}"
                        print(rendered, end="", flush=True)
                        handle.write(rendered)
                        handle.flush()
                    break
        finally:
            selector.close()
        code = proc.wait()
        handle.write(f"[{now()}] EXIT={code}\n")
        handle.flush()
    print(f"[{now()}] EXIT={code}", flush=True)
    return code


def package_versions() -> dict[str, str]:
    names = ["torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes"]
    result: dict[str, str] = {}
    for name in names:
        try:
            module = __import__(name)
            result[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001 - environment report must continue
            result[name] = f"unavailable:{type(exc).__name__}"
    return result


def git_revision(directory: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() or f"unavailable:{result.returncode}"
    except OSError as exc:
        return f"unavailable:{type(exc).__name__}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/falcon-production-v1.json"))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", required=True, choices=["stage_a_capability_preserving", "stage_b_clinical_safety", "stage_c_mcqa_replay"])
    ap.add_argument("--resume-from-checkpoint", default=None, help="Path or 'latest' under this stage checkpoint directory")
    ap.add_argument("--init-adapter", default=None, help="Optional adapter from the preceding stage; unlike resume, this starts a new stage with fresh optimizer/scheduler state")
    ap.add_argument("--max-steps", type=int, default=0, help="Override config for smoke/resume tests")
    ap.add_argument("--save-steps", type=int, default=0, help="Override checkpoint interval; used by the tiny resume test")
    ap.add_argument("--quantize", choices=("auto", "4bit", "none"), default="auto")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-ephemeral", action="store_true", help="Only for local/tiny resume tests; bypass required Kaggle checkpoint persistence")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stage_cfg = config["stages"][args.stage]
    data_dir = Path(args.data_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    stage_dir = run_dir / args.stage
    checkpoint_dir = stage_dir / "checkpoints"
    stage_dir.mkdir(parents=True, exist_ok=True)
    event_path = stage_dir / "events.jsonl"
    metrics_path = stage_dir / "training_metrics.jsonl"
    heartbeat_path = stage_dir / "heartbeat.jsonl"
    seed = int(args.seed if args.seed is not None else config["training"]["seed"])
    persistence_dataset = os.environ.get(config["persistence"]["dataset_slug_env"])
    if args.stage != "stage_a_capability_preserving" and not args.init_adapter and not args.resume_from_checkpoint and not args.dry_run:
        raise RuntimeError(f"{args.stage} must initialize from the preceding stage adapter; pass --init-adapter (resume is only for an interrupted same-stage run)")
    if config["persistence"].get("required_before_long_run", True) and not persistence_dataset and not args.allow_ephemeral and not args.dry_run:
        raise RuntimeError(
            f"{config['persistence']['dataset_slug_env']} is unset; refusing a long run without durable checkpoint persistence"
        )
    set_seed(seed)

    train_rows = load_jsonl(data_dir / "train.jsonl")
    dev_rows = load_jsonl(data_dir / "dev.jsonl")
    event(event_path, "startup", experiment_id=config["experiment_id"], stage=args.stage, train_rows=len(train_rows), dev_rows=len(dev_rows), seed=seed)
    if args.dry_run:
        print(json.dumps({"train_rows": len(train_rows), "dev_rows": len(dev_rows), "stage": stage_cfg}, indent=2))
        return 0

    import torch
    from scripts.train_lora import get_cuda_capability, patch_peft_transformers_compat, resolve_precision

    # PEFT 0.15 imports a legacy Transformers symbol at module import time;
    # patch before importing PEFT, not after it.
    patch_peft_transformers_compat()
    from transformers import AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1ForCausalLM
    model_id = config["model"]["id"]
    revision = config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=config["model"].get("tokenizer_revision", revision), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_len = int(config["data"]["max_length"])
    train_data = FalconDataset(train_rows, tokenizer, max_len, config["data"]["system_prompt"])
    dev_data = FalconDataset(dev_rows, tokenizer, max_len, config["data"]["system_prompt"])
    objective_counts = Counter(item["kind"] for item in train_data.items)
    total_tokens = sum(item["tokens"] for item in train_data.items)
    event(event_path, "dataset_ready", train_items=len(train_data), dev_items=len(dev_data), objective_counts=dict(objective_counts), train_tokens=total_tokens, max_len=max_len)

    use_cuda = torch.cuda.is_available()
    cap = get_cuda_capability(torch) if use_cuda else None
    requested_quant = args.quantize
    if requested_quant == "auto":
        requested_quant = "4bit" if use_cuda and cap and cap >= (7, 0) else "none"
    load_mode, dtype_name = resolve_precision(use_cuda=use_cuda, cuda_capability=cap, mps=False, quantize=requested_quant, compute_dtype="auto")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    event(event_path, "environment", python=platform.python_version(), platform=platform.platform(), packages=package_versions(), cuda=use_cuda, cuda_capability=cap, dtype=dtype_name, load_mode=load_mode)
    atomic_json(stage_dir / "environment.json", {"timestamp_utc": now(), "python": platform.python_version(), "platform": platform.platform(), "packages": package_versions(), "cuda": use_cuda, "cuda_capability": cap, "dtype": dtype_name, "load_mode": load_mode})
    data_manifest_path = data_dir / "data_manifest.json"
    atomic_json(stage_dir / "run_manifest.json", {
        "experiment_id": config["experiment_id"], "stage": args.stage,
        "git_sha": git_revision(ROOT),
        "command": sys.argv,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model": config["model"], "seed": seed, "stage_config": stage_cfg,
        "data_manifest": str(data_manifest_path),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "init_adapter": args.init_adapter,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "quantize": args.quantize, "started_utc": now(),
    })

    event(event_path, "model_load_start", model=model_id, revision=revision)
    load_kwargs = {"revision": revision, "trust_remote_code": True, "torch_dtype": dtype}
    if load_mode == "bnb4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        load_kwargs["device_map"] = "auto"
    model = FalconH1ForCausalLM.from_pretrained(model_id, **load_kwargs)
    if load_mode == "plain":
        model = model.to("cuda" if use_cuda else "cpu")
    model.config.use_cache = False
    if load_mode == "bnb4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    event(event_path, "model_load_complete", model=model_id, revision=revision, load_mode=load_mode, dtype=dtype_name)

    targets = list(config["training"]["lora_target_modules"])
    linear_suffixes = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    missing_targets = [name for name in targets if name not in linear_suffixes]
    if missing_targets:
        raise RuntimeError(f"Configured Falcon LoRA targets not present as Linear modules: {missing_targets}; available={sorted(linear_suffixes)}")
    lora_cfg = LoraConfig(r=int(stage_cfg["lora_r"]), lora_alpha=int(stage_cfg["lora_alpha"]), lora_dropout=float(stage_cfg["lora_dropout"]), bias="none", task_type="CAUSAL_LM", target_modules=targets)
    if args.init_adapter:
        init_adapter = Path(args.init_adapter).resolve()
        if not init_adapter.is_dir():
            raise FileNotFoundError(f"initial adapter does not exist: {init_adapter}")
        event(event_path, "init_adapter_load_start", adapter=str(init_adapter))
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
        event(event_path, "init_adapter_load_complete", adapter=str(init_adapter))
    else:
        model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    event(event_path, "model_ready", trainable_parameters=trainable, total_parameters=total, trainable_percent=round(100 * trainable / max(1, total), 6), lora_targets=targets)
    model.print_trainable_parameters()

    progress = Progress(heartbeat_path, int(config["training"]["heartbeat_seconds"]))

    class ProductionTrainer(Trainer):
        def __init__(self, *trainer_args, objective_weights: dict[str, float], sampling_token_share: dict[str, float], **trainer_kwargs):
            self.objective_weights = objective_weights
            self.sampling_token_share = sampling_token_share
            self._component_sum = Counter()
            self._component_count = Counter()
            self._microstep = 0
            self._seen_tokens = 0
            self._seen_examples = 0
            self._training_started = time.monotonic()
            super().__init__(*trainer_args, **trainer_kwargs)

        def _pad(self, sequences: list[list[int]], value: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
            width = max(len(x) for x in sequences)
            ids = torch.full((len(sequences), width), value, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(sequences):
                ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                mask[i, :len(seq)] = 1
            return ids, mask

        def _sft_loss(self, model_arg, batch: list[dict[str, Any]]) -> tuple[torch.Tensor, int]:
            ids, mask = self._pad([x["input_ids"] for x in batch], int(tokenizer.pad_token_id), model_arg.device)
            labels, _ = self._pad([x["labels"] for x in batch], -100, model_arg.device)
            out = model_arg(input_ids=ids, attention_mask=mask, labels=labels)
            return out.loss, sum(x["tokens"] for x in batch)

        def _mcqa_loss(self, model_arg, batch: list[dict[str, Any]]) -> tuple[torch.Tensor, int]:
            sequences: list[list[int]] = []
            spans: list[tuple[int, int, int, list[int]]] = []
            for item in batch:
                start = len(sequences)
                for choice in item["choice_ids"]:
                    sequences.append(item["context_ids"] + choice)
                spans.append((start, len(sequences), len(item["context_ids"]), item["choice_ids"]))
            ids, mask = self._pad(sequences, int(tokenizer.pad_token_id), model_arg.device)
            out = model_arg(input_ids=ids, attention_mask=mask)
            rows: list[int] = []
            positions: list[int] = []
            targets: list[int] = []
            choices_per_item: list[list[tuple[int, int]]] = []
            for start, end, ctx_len, choices in spans:
                item_spans = []
                for row, choice in zip(range(start, end), choices):
                    begin = len(rows)
                    for offset, token in enumerate(choice):
                        rows.append(row); positions.append(ctx_len + offset - 1); targets.append(token)
                    item_spans.append((begin, len(rows)))
                choices_per_item.append(item_spans)
            if any(position < 0 for position in positions):
                raise RuntimeError("MCQA context must contain at least one token")
            selected = out.logits[torch.tensor(rows, device=model_arg.device), torch.tensor(positions, device=model_arg.device), :].float()
            selected = selected - selected.max(dim=-1, keepdim=True).values
            token_logp = selected.log_softmax(dim=-1).gather(1, torch.tensor(targets, device=model_arg.device).unsqueeze(1)).squeeze(1)
            losses = []
            offset = 0
            for item, item_spans in zip(batch, choices_per_item):
                sums = [token_logp[a:b].sum() for a, b in item_spans]
                norms = [value / max(1, len(str(choice))) for value, choice in zip(sums, item["choices"])]
                ranking = -torch.stack(norms).log_softmax(dim=0)[item["gold"]]
                gold_a, gold_b = item_spans[item["gold"]]
                aux = -(token_logp[gold_a:gold_b].mean())
                losses.append(ranking + 0.2 * aux)
                offset += sum(b - a for a, b in item_spans)
            return torch.stack(losses).mean(), sum(x["tokens"] for x in batch)

        def compute_loss(self, model_arg, inputs, return_outputs=False, num_items_in_batch=None):
            del num_items_in_batch
            by_kind = {"sft": [], "mcqa": []}
            for item in inputs:
                by_kind[item["kind"]].append(item)
            losses = {}
            token_count = 0
            if by_kind["sft"]:
                losses["sft"] , tokens = self._sft_loss(model_arg, by_kind["sft"])
                token_count += tokens
            if by_kind["mcqa"]:
                losses["mcqa"], tokens = self._mcqa_loss(model_arg, by_kind["mcqa"])
                token_count += tokens
            active = [(name, loss, float(self.objective_weights.get(name, 0.0))) for name, loss in losses.items() if self.objective_weights.get(name, 0.0) > 0]
            if not active:
                raise RuntimeError(f"No active objective in batch; weights={self.objective_weights}, kinds={list(losses)}")
            denom = sum(weight for _, _, weight in active)
            loss = sum(weight * value for _, value, weight in active) / denom
            # Trainer calls compute_loss for both training and evaluation.  Only
            # training calls may update the live training counters; otherwise an
            # eval pass would silently contaminate the next training log record.
            if model_arg.training:
                self._microstep += 1
                self._component_sum["total"] += float(loss.detach().cpu())
                self._component_count["total"] += 1
                for name, value in losses.items():
                    self._component_sum[name] += float(value.detach().cpu())
                    self._component_count[name] += 1
                self._seen_tokens += token_count
                self._seen_examples += len(inputs)
                progress.update(microstep=self._microstep, loss=float(loss.detach().cpu()), tokens=token_count, examples=len(inputs))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at microstep {self._microstep}")
            return (loss, None) if return_outputs else loss

        def prediction_step(self, model_arg, inputs, prediction_loss_only, ignore_keys=None):
            """Evaluate this list-based dataset without asking Trainer to collate labels/logits."""
            del prediction_loss_only, ignore_keys
            model_arg.eval()
            with torch.no_grad():
                loss = self.compute_loss(model_arg, inputs)
            return loss.detach(), None, None

        def log(self, logs: dict[str, float], *log_args, **log_kwargs):
            enriched = dict(logs)
            for name in ("sft", "mcqa", "total"):
                if self._component_count[name]:
                    enriched[f"loss_{name}"] = self._component_sum[name] / self._component_count[name]
            # HF Trainer's state/log history expects numeric metric values.  Keep
            # the timestamp in our JSONL artifact, but never pass it to Trainer
            # as a pseudo-metric (which breaks some older Transformers versions).
            timestamp = now()
            elapsed = max(1e-6, time.monotonic() - self._training_started)
            enriched["elapsed_seconds"] = round(elapsed, 3)
            enriched["tokens_per_second"] = round(self._seen_tokens / elapsed, 3)
            enriched["examples_per_second"] = round(self._seen_examples / elapsed, 3)
            if estimated_steps and self.state.global_step:
                enriched["eta_seconds"] = round(max(0, estimated_steps - self.state.global_step) * elapsed / self.state.global_step, 1)
            if torch.cuda.is_available():
                enriched["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
                enriched["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
            # Keep both the human log and Trainer's log history JSON-safe.
            enriched = {
                key: (float(value) if hasattr(value, "item") else value)
                for key, value in enriched.items()
            }
            print(f"[{timestamp}] METRICS {json.dumps(enriched, sort_keys=True)}", flush=True)
            append_jsonl(metrics_path, {"event": "train_log", "timestamp_utc": timestamp, "global_step": int(self.state.global_step), **enriched})
            super().log(enriched, *log_args, **log_kwargs)
            self._component_sum.clear(); self._component_count.clear()

        def _get_train_sampler(self):
            from torch.utils.data import WeightedRandomSampler
            # Objective weights describe desired *loss-token* exposure, not
            # row frequency.  The helper gives every objective a total
            # expected token mass equal to its configured share.
            weights = token_share_sampling_weights(self.train_dataset.items, self.sampling_token_share)
            if not any(weight > 0 for weight in weights):
                raise RuntimeError("all sampler weights are zero")
            return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)

    from transformers import TrainerCallback

    class ProductionCallback(TrainerCallback):
        persisted_steps: set[int] = set()

        def _persist(self, path: Path, step: int, args_):
            upload_every = int(config["persistence"].get("checkpoint_upload_every_steps", 0))
            if not persistence_dataset or not upload_every or step in self.persisted_steps:
                return
            if step % upload_every != 0:
                return
            command = [sys.executable, str(ROOT / "scripts" / "persist_checkpoint.py"), "--checkpoint", str(path), "--dataset", persistence_dataset, "--message", f"{config['experiment_id']} {args.stage} step {step}"]
            event(event_path, "checkpoint_persist_start", checkpoint=str(path), dataset=persistence_dataset)
            result = run_streamed(command, event_path.parent / "persistence.log")
            if result:
                raise RuntimeError(f"checkpoint persistence failed with exit {result}")
            self.persisted_steps.add(step)
            event(event_path, "checkpoint_persist_complete", checkpoint=str(path), dataset=persistence_dataset)

        def on_step_end(self, args_, state, control, **kwargs):
            del args_, control, kwargs
            progress.update(step=int(state.global_step))
            return None

        def on_save(self, args_, state, control, **kwargs):
            del control, kwargs
            path = Path(args_.output_dir) / f"checkpoint-{state.global_step}"
            if path.exists():
                if not checkpoint_is_complete(path, require_scaler=bool(args_.fp16)):
                    raise RuntimeError(f"Trainer produced an incomplete checkpoint: {path}")
                files = [str(x.relative_to(path)) for x in path.rglob("*") if x.is_file()]
                atomic_json(path / "checkpoint_manifest.json", {"timestamp_utc": now(), "global_step": int(state.global_step), "files": files, "complete": True, "scaler_required": bool(args_.fp16), "scaler_present": (path / "scaler.pt").is_file()})
                event(event_path, "checkpoint_saved", path=str(path), global_step=int(state.global_step), file_count=len(files))
                self._persist(path, int(state.global_step), args_)

        def on_train_end(self, args_, state, control, **kwargs):
            del control, kwargs
            # A final stage is often shorter than the periodic upload interval.
            # Persist its latest complete Trainer checkpoint as well, so the
            # durable store always contains a resumable endpoint.
            path = latest_checkpoint(Path(args_.output_dir))
            if path is not None and persistence_dataset:
                step = int(path.name.split("-")[-1])
                if step not in self.persisted_steps:
                    original = config["persistence"].get("checkpoint_upload_every_steps", 0)
                    config["persistence"]["checkpoint_upload_every_steps"] = 1
                    try:
                        self._persist(path, step, args_)
                    finally:
                        config["persistence"]["checkpoint_upload_every_steps"] = original

    training = config["training"]
    counts = Counter(item["kind"] for item in train_data.items)
    missing_objectives = [name for name, weight in stage_cfg["objective_weights"].items() if float(weight) > 0 and counts.get(name, 0) == 0]
    if missing_objectives:
        raise RuntimeError(f"stage {args.stage} requires objective data that is absent: {missing_objectives}; rebuild the production dataset before training")
    per_device = int(training["per_device_batch_size"])
    grad_accum = int(training["gradient_accumulation_steps"])
    steps_per_epoch = max(1, math.ceil(len(train_data) / max(1, per_device * grad_accum)))
    estimated_steps = int(args.max_steps) if args.max_steps > 0 else math.ceil(steps_per_epoch * float(stage_cfg["epochs"]))
    progress.total_steps = estimated_steps
    kwargs = dict(
        output_dir=str(checkpoint_dir), num_train_epochs=float(stage_cfg["epochs"]),
        per_device_train_batch_size=int(training["per_device_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(stage_cfg["learning_rate"]), lr_scheduler_type=training["scheduler"],
        warmup_ratio=float(training["warmup_ratio"]), logging_steps=int(training["logging_steps"]),
        save_strategy="steps", save_steps=int(args.save_steps or training["save_steps"]), save_total_limit=int(training["save_total_limit"]),
        max_grad_norm=float(training["max_grad_norm"]), gradient_checkpointing=bool(training["gradient_checkpointing"]),
        report_to="none", remove_unused_columns=False, dataloader_num_workers=0, seed=seed,
        bf16=(use_cuda and dtype_name == "bf16"), fp16=(use_cuda and dtype_name == "fp16"),
        max_steps=int(args.max_steps) if args.max_steps > 0 else -1,
    )
    import inspect
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = int(training["eval_steps"])
    else:
        kwargs["evaluation_strategy"] = "steps"
        kwargs["eval_steps"] = int(training["eval_steps"])
    if "optim" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["optim"] = training["optimizer"]
    train_args = TrainingArguments(**kwargs)
    production_callback = ProductionCallback()
    trainer = ProductionTrainer(model=model, args=train_args, train_dataset=train_data, eval_dataset=dev_data, data_collator=identity_collate, objective_weights=stage_cfg["objective_weights"], sampling_token_share=stage_cfg.get("sampling_token_share", stage_cfg["objective_weights"]), callbacks=[production_callback])

    def save_terminal_checkpoint() -> Path:
        """Create a complete resumable checkpoint even off the save interval."""
        step = int(trainer.state.global_step)
        path = checkpoint_dir / f"checkpoint-{step}"
        if path.exists() and not checkpoint_is_complete(path):
            raise RuntimeError(f"terminal checkpoint path exists but is incomplete: {path}")
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(path))
            trainer.state.save_to_json(str(path / "trainer_state.json"))
            if trainer.optimizer is not None:
                torch.save(trainer.optimizer.state_dict(), path / "optimizer.pt")
            if trainer.lr_scheduler is not None:
                torch.save(trainer.lr_scheduler.state_dict(), path / "scheduler.pt")
            scaler = getattr(getattr(trainer, "accelerator", None), "scaler", None)
            if scaler is not None:
                torch.save(scaler.state_dict(), path / "scaler.pt")
            rng = {"python": random.getstate(), "torch": torch.get_rng_state()}
            try:
                import numpy as np
                rng["numpy"] = np.random.get_state()
            except ImportError:
                pass
            if torch.cuda.is_available():
                rng["cuda"] = torch.cuda.get_rng_state_all()
            torch.save(rng, path / "rng_state.pth")
        files = [str(x.relative_to(path)) for x in path.rglob("*") if x.is_file()]
        atomic_json(path / "checkpoint_manifest.json", {"timestamp_utc": now(), "global_step": step, "files": files, "complete": True, "terminal": True, "scaler_required": bool(train_args.fp16), "scaler_present": (path / "scaler.pt").is_file()})
        if not checkpoint_is_complete(path, require_scaler=bool(train_args.fp16)):
            raise RuntimeError(f"terminal checkpoint failed completeness validation: {path}")
        return path
    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume_path = latest_checkpoint(checkpoint_dir)
        resume = str(resume_path) if resume_path else None
    if resume:
        if not Path(resume).exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume}")
        event(event_path, "resume_start", checkpoint=resume)
    trainer._training_started = time.monotonic()
    progress.started = trainer._training_started
    progress.start()
    event(event_path, "train_start", optimizer_steps_estimate=estimated_steps, steps_per_epoch=steps_per_epoch, effective_batch_size=per_device * grad_accum, objective_weights=stage_cfg["objective_weights"], sampling_token_share=stage_cfg.get("sampling_token_share", stage_cfg["objective_weights"]), resume=resume, init_adapter=args.init_adapter)
    try:
        trainer.train(resume_from_checkpoint=resume)
        terminal_checkpoint = save_terminal_checkpoint()
        event(event_path, "terminal_checkpoint_saved", path=str(terminal_checkpoint), global_step=int(trainer.state.global_step))
        if persistence_dataset and int(trainer.state.global_step) not in production_callback.persisted_steps:
            event(event_path, "checkpoint_persist_start", checkpoint=str(terminal_checkpoint), dataset=persistence_dataset, terminal=True)
            result = run_streamed([sys.executable, str(ROOT / "scripts" / "persist_checkpoint.py"), "--checkpoint", str(terminal_checkpoint), "--dataset", persistence_dataset, "--message", f"{config['experiment_id']} {args.stage} terminal step {trainer.state.global_step}"], event_path.parent / "persistence-terminal.log")
            if result:
                raise RuntimeError(f"terminal checkpoint persistence failed with exit {result}")
            event(event_path, "checkpoint_persist_complete", checkpoint=str(terminal_checkpoint), dataset=persistence_dataset, terminal=True)
        trainer.save_model(str(checkpoint_dir / "final-adapter"))
        tokenizer.save_pretrained(str(checkpoint_dir / "final-adapter"))
        event(event_path, "train_complete", global_step=int(trainer.state.global_step), final_adapter=str(checkpoint_dir / "final-adapter"))
        atomic_json(stage_dir / "final_summary.json", {"status": "complete", "stage": args.stage, "global_step": int(trainer.state.global_step), "completed_utc": now(), "trainable_parameters": trainable, "total_parameters": total})
    except Exception as exc:  # noqa: BLE001 - preserve a machine-readable failure
        event(event_path, "train_failed", error_type=type(exc).__name__, error=str(exc))
        atomic_json(stage_dir / "final_summary.json", {"status": "failed", "stage": args.stage, "failed_utc": now(), "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        progress.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
