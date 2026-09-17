#!/usr/bin/env python3
"""Run the single fixed Falcon-H1 submission SFT trajectory.

This is intentionally a small, ordinary causal-LM loop.  The complete JSONL
corpus is rendered and tokenized once on the CPU, examples are packed by mix
bucket into 512-token sequences, and every GPU forward receives only batched
``input_ids``, ``attention_mask`` and assistant-only ``labels`` tensors.

The script is not a research harness: there are no strategy, learning-rate,
target-module, quantization, or step overrides.  It loads the pinned base once,
creates one full-model LoRA adapter, and runs the three configured stages on
that same adapter trajectory.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scripts.build_falcon_submission_sft import SYSTEM_PROMPT
from scripts.falcon_format import generation_stop_ids, render_completion


MODEL_ID = "tiiuae/Falcon-H1-1.5B-Deep-Instruct"
MODEL_REVISION = "b6648636ddc906688974282de6e7a243395f5423"
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "gate_proj", "up_proj", "down_proj",
)
FORBIDDEN_MODULES = ("out_proj", "conv1d")
STAGES = (
    "stage1_domain_lock_in",
    "stage2_safety_polish",
    "stage3_capability_replay",
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    return result.stdout.strip() or "unknown"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit(path: Path, event: str, **fields: Any) -> None:
    record = {"timestamp_utc": now(), "event": event, **fields}
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def validate_config(config: dict[str, Any]) -> None:
    """Refuse silent drift from the research-lead decision."""
    model = config.get("model", {})
    if model.get("id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise ValueError("submission trainer is pinned to the approved Falcon-H1 model revision")
    data = config.get("data", {})
    if data.get("system_prompt") != SYSTEM_PROMPT:
        raise ValueError("config system prompt is not the exact compact Jamii Afya prompt")
    if int(data.get("max_length", 0)) != 512:
        raise ValueError("submission max_length must be 512")
    lora = config.get("lora", {})
    targets = tuple(lora.get("target_modules", ()))
    if targets != TARGET_MODULES:
        raise ValueError(f"submission LoRA targets must be exactly {list(TARGET_MODULES)}")
    if any(forbidden in targets for forbidden in FORBIDDEN_MODULES):
        raise ValueError("out_proj and conv1d are forbidden LoRA targets")
    if (lora.get("r"), lora.get("lora_alpha"), lora.get("lora_dropout"), lora.get("bias")) != (16, 32, 0.05, "none"):
        raise ValueError("submission LoRA hyperparameters drifted from the fixed configuration")
    hardware = config.get("hardware", {})
    if tuple(hardware.get("minimum_cuda_capability", ())) < (7, 5):
        raise ValueError("submission hardware policy must require sm75+")
    if hardware.get("compute_dtype") != "float16" or not hardware.get("optimized_mamba_required"):
        raise ValueError("submission hardware policy must require optimized fp16 Mamba")
    training = config.get("training", {})
    exact_training = {
        "max_length": 512,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "ddp_find_unused_parameters": False,
        "gradient_checkpointing": False,
        "optimizer": "adamw_torch",
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
    }
    for key, expected in exact_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}, got {training.get(key)!r}")
    if sum(int(config["stages"][stage]["optimizer_steps"]) for stage in STAGES) != 136:
        raise ValueError("submission trajectory must contain exactly 136 optimizer steps")
    for stage in STAGES:
        mixture_name = config["stages"][stage]["mixture"]
        mixture = config["data"]["mixture"][mixture_name]
        expected_buckets = {"clinical_safety", "normal_clinical", "general_conversational", "mcqa_sft", "kiswahili_boost"}
        if set(mixture) != expected_buckets:
            raise ValueError(f"{stage}: mixture buckets are incomplete")
        if not math.isclose(sum(float(x) for x in mixture.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"{stage}: mixture must sum to one")


def cuda_capabilities(torch: Any) -> list[tuple[int, int]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to load Falcon-H1 submission model")
    capabilities = [tuple(int(x) for x in torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    if not capabilities or any(value < (7, 5) for value in capabilities):
        raise RuntimeError(f"all visible GPUs must be sm75+; detected capabilities={capabilities}")
    return capabilities


def require_fast_mamba(torch: Any, minimum: tuple[int, int] = (7, 5)) -> dict[str, Any]:
    """Hardware and import gate; called before tokenizer or model loading."""
    capabilities = cuda_capabilities(torch)
    if any(value < minimum for value in capabilities):
        raise RuntimeError(f"detected GPU below sm{minimum[0]}{minimum[1]}; refusing naive Mamba")
    try:
        import mamba_ssm
        from mamba_ssm import Mamba2
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        import causal_conv1d
        from causal_conv1d import causal_conv1d_fn
    except Exception as exc:  # noqa: BLE001 - this boundary must fail closed
        raise RuntimeError(
            "optimized mamba-ssm + causal-conv1d is unavailable; refusing Falcon-H1 model load"
        ) from exc
    if not callable(Mamba2) or not callable(selective_scan_fn) or not callable(causal_conv1d_fn):
        raise RuntimeError("Mamba/causal-conv1d fast-path symbols are not callable")
    return {
        "mamba_path": "optimized_mamba_ssm_causal_conv1d",
        "capabilities": [list(value) for value in capabilities],
        "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mamba_ssm": getattr(mamba_ssm, "__version__", "unknown"),
        "causal_conv1d": getattr(causal_conv1d, "__version__", "unknown"),
    }


def _user_text(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction") or "").strip()
    extra = str(row.get("input") or "").strip()
    if extra and extra not in instruction:
        instruction = f"{instruction}\n\n{extra}"
    if not instruction:
        raise ValueError(f"{row.get('example_id', '?')}: empty user text")
    return instruction


def tokenize_and_pack(rows: list[dict[str, Any]], tokenizer: Any, max_length: int = 512) -> "PackedDataset":
    """Render/tokenize the complete corpus once and pack within each bucket."""
    if max_length != 512:
        raise ValueError("the submission trainer only supports max_length=512")
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("format") != "sft" or row.get("assistant_only_loss") is not True:
            raise ValueError(f"{row.get('example_id', '?')}: submission data must be assistant-only SFT")
        bucket = str(row.get("mix_bucket") or row.get("category") or "")
        if bucket not in {"clinical_safety", "normal_clinical", "general_conversational", "mcqa_sft", "kiswahili_boost"}:
            raise ValueError(f"{row.get('example_id', '?')}: unsupported mix bucket {bucket!r}")
        by_bucket.setdefault(bucket, []).append(row)

    packed: list[dict[str, Any]] = []
    rejected: list[str] = []
    for bucket in sorted(by_bucket):
        current_ids: list[int] = []
        current_labels: list[int] = []
        current_examples: list[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current_ids, current_labels, current_examples, current_tokens
            if not current_ids:
                return
            if not any(label != -100 for label in current_labels):
                raise ValueError(f"packed {bucket} sequence has no supervised labels")
            packed.append({
                "input_ids": current_ids,
                "labels": current_labels,
                "mix_bucket": bucket,
                "loss_tokens": current_tokens,
                "example_ids": current_examples,
            })
            current_ids, current_labels, current_examples, current_tokens = [], [], [], 0

        for row in by_bucket[bucket]:
            system = row.get("system_prompt")
            messages: list[dict[str, str]] = []
            if system:
                if str(system) != SYSTEM_PROMPT:
                    raise ValueError(f"{row.get('example_id', '?')}: unexpected system prompt")
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            answer = str(row.get("output") or "").strip()
            if not answer:
                rejected.append(str(row.get("example_id", "?")))
                continue
            messages.append({"role": "user", "content": _user_text(row)})
            prompt_ids, target_ids = render_completion(tokenizer, messages, answer)
            if len(target_ids) > max_length:
                rejected.append(str(row.get("example_id", "?")))
                continue
            prompt_ids = prompt_ids[-max(0, max_length - len(target_ids)):]
            ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            if len(ids) > max_length or not target_ids:
                rejected.append(str(row.get("example_id", "?")))
                continue
            if current_ids and len(current_ids) + len(ids) > max_length:
                flush()
            if len(ids) > max_length:
                raise ValueError(f"{row.get('example_id', '?')}: packed sequence exceeds max_length")
            current_ids.extend(ids)
            current_labels.extend(labels)
            current_examples.append(str(row.get("example_id", "?")))
            current_tokens += len(target_ids)
        flush()
    if rejected:
        raise ValueError(f"{len(rejected)} rows failed pre-tokenization: {rejected[:8]}")
    if not packed:
        raise ValueError("submission corpus produced no packed sequences")
    return PackedDataset(packed)


class PackedDataset:
    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]

    def token_summary(self) -> dict[str, Any]:
        totals = Counter()
        for item in self.items:
            totals[str(item["mix_bucket"])] += int(item["loss_tokens"])
        denominator = max(1, sum(totals.values()))
        return {
            "loss_tokens_by_mix_bucket": dict(sorted(totals.items())),
            "loss_token_share_percent_by_mix_bucket": {
                key: round(100.0 * value / denominator, 4) for key, value in sorted(totals.items())
            },
            "packed_sequences": len(self.items),
        }


def collate_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention = []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class TokenShareSampler:
    """Deterministic replacement sampler weighted by supervised-token mass."""

    def __init__(self, dataset: PackedDataset, shares: dict[str, float], num_samples: int, seed: int):
        import torch

        totals = Counter(str(item["mix_bucket"]) for item in dataset.items)
        token_totals = Counter()
        for item in dataset.items:
            token_totals[str(item["mix_bucket"])] += max(1, int(item["loss_tokens"]))
        missing = [bucket for bucket, share in shares.items() if float(share) > 0 and token_totals[bucket] == 0]
        if missing:
            raise ValueError(f"requested mixture buckets have no supervised tokens: {missing}")
        self.weights = torch.tensor(
            [float(shares.get(str(item["mix_bucket"]), 0.0)) / max(1, token_totals[str(item["mix_bucket"])]) for item in dataset.items],
            dtype=torch.double,
        )
        if not bool(torch.isfinite(self.weights).all()) or float(self.weights.sum()) <= 0:
            raise ValueError("invalid token-share sampler weights")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.bucket_row_counts = dict(sorted(totals.items()))

    def __iter__(self) -> Iterator[int]:
        import torch

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True, generator=generator).tolist())

    def __len__(self) -> int:
        return self.num_samples


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).casefold()


class InitialBatchOOM(RuntimeError):
    """Signal the one permitted batch-size fallback before a stage mutates weights."""


def distributed_info(torch: Any) -> tuple[int, int, int, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return world_size, rank, local_rank, device


def barrier(torch: Any, world_size: int) -> None:
    if world_size > 1:
        torch.distributed.barrier()


def fast_dev_validate(
    model: Any,
    tokenizer: Any,
    battery_path: Path,
    output_dir: Path,
    stage_label: str,
    step: int,
    device: Any,
    torch: Any,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Run only the existing small development generation battery."""
    payload = json.loads(battery_path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts", []) if isinstance(payload, dict) else payload
    module = model.module if hasattr(model, "module") else model
    was_training = module.training
    module.eval()
    results: list[dict[str, Any]] = []
    try:
        from scripts.score_falcon_battery import rule_result

        stop_ids = generation_stop_ids(tokenizer, module.generation_config)
        for prompt in prompts:
            prompt_id = str(prompt.get("id") or "")
            text = str(prompt.get("text") or "")
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
            try:
                encoded = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_tensors="pt",
                )
            except TypeError:
                encoded = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                )
            if isinstance(encoded, dict) or hasattr(encoded, "keys"):
                inputs = {key: value.to(device) for key, value in encoded.items()}
                prompt_len = int(inputs["input_ids"].shape[-1])
            else:
                inputs = {"input_ids": encoded.to(device), "attention_mask": torch.ones_like(encoded).to(device)}
                prompt_len = int(encoded.shape[-1])
            with torch.inference_mode():
                generated = module.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    eos_token_id=stop_ids or None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated_ids = generated[0, prompt_len:].detach().cpu().tolist()
            output = tokenizer.decode(generated_ids, skip_special_tokens=True)
            quality = rule_result(prompt_id, output, prompt.get("quality", {}), 1)
            quality.update({
                "section": prompt.get("section", "unknown"),
                "check": prompt.get("check", ""),
                "new_tokens": len(generated_ids),
                "source_text": text,
            })
            results.append(quality)
    finally:
        if was_training:
            module.train()
    critical = [item["id"] for item in results if not item["passed"] and item.get("section") == "safety"]
    report = {
        "schema": "falcon-fast-dev-v1",
        "stage": stage_label,
        "step": int(step),
        "battery": str(battery_path),
        "max_new_tokens": int(max_new_tokens),
        "prompt_count": len(prompts),
        "passed_count": sum(bool(item["passed"]) for item in results),
        "failed_count": sum(not bool(item["passed"]) for item in results),
        "critical_failures": critical,
        "invalid": bool(critical),
        "results": results,
    }
    if int(os.environ.get("RANK", "0")) == 0:
        destination = output_dir / stage_label / f"step-{step}"
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "fast_dev_report.json", report)
    return report


def save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    checkpoint_dir: Path,
    step: int,
    stage: str,
    config_path: Path,
    data_path: Path,
    sampler: TokenShareSampler,
    torch: Any,
    rank: int,
) -> Path:
    checkpoint = checkpoint_dir / f"checkpoint-{step}"
    if rank == 0:
        checkpoint.mkdir(parents=True, exist_ok=True)
        module = model.module if hasattr(model, "module") else model
        module.save_pretrained(str(checkpoint), safe_serialization=True)
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
        if scaler is not None:
            scaler_state = scaler.state_dict()
            if scaler_state:
                torch.save(scaler_state, checkpoint / "scaler.pt")
        torch.save({"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}, checkpoint / "rng_state.pth")
        atomic_json(checkpoint / "sampler_state.json", {
            "seed": sampler.seed,
            "num_samples": sampler.num_samples,
            "algorithm": "torch.multinomial_supervised_token_share_v1",
        })
        atomic_json(checkpoint / "checkpoint_manifest.json", {
            "schema": "falcon-submission-checkpoint-v1",
            "complete": True,
            "stage": stage,
            "global_step": int(step),
            "optimizer_steps_in_stage": int(step),
            "config_sha256": sha256_file(config_path),
            "data_sha256": sha256_file(data_path),
            "adapter_commit": git_revision(),
            "target_modules": list(TARGET_MODULES),
            "forbidden_targets": list(FORBIDDEN_MODULES),
            "sampler_required": True,
            "scaler_present": bool(scaler is not None and scaler.state_dict()),
            "files": sorted(str(item.relative_to(checkpoint)) for item in checkpoint.rglob("*") if item.is_file()),
        })
    return checkpoint


def run_stage(
    *,
    model: Any,
    tokenizer: Any,
    dataset: PackedDataset,
    config: dict[str, Any],
    config_path: Path,
    data_path: Path,
    run_dir: Path,
    stage_name: str,
    stage_cfg: dict[str, Any],
    stage_mixture: dict[str, float],
    world_size: int,
    rank: int,
    device: Any,
    torch: Any,
    fast_battery: Path,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
) -> dict[str, Any]:
    """Run one stage on the already-loaded model/adapter."""
    from torch.utils.data import DataLoader

    steps = int(stage_cfg["optimizer_steps"])
    per_device_batch = int(per_device_batch_size)
    grad_accum = int(gradient_accumulation_steps)
    num_samples = steps * grad_accum * per_device_batch
    stage_seed = int(config["training"]["seed"]) + STAGES.index(stage_name) * 1009 + rank
    sampler = TokenShareSampler(dataset, stage_mixture, num_samples, stage_seed)
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        sampler=sampler,
        collate_fn=partial(collate_batch, pad_token_id=int(tokenizer.pad_token_id)),
        num_workers=int(config["training"].get("dataloader_num_workers", 2)),
        pin_memory=bool(config["training"].get("dataloader_pin_memory", True)),
        persistent_workers=bool(config["training"].get("dataloader_persistent_workers", True)),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(stage_cfg["learning_rate"]),
        betas=(float(config["training"]["beta1"]), float(config["training"]["beta2"])),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    warmup_steps = math.ceil(float(stage_cfg["warmup_ratio"]) * steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if stage_cfg["scheduler"] == "constant":
            return 1.0
        decay_steps = max(1, steps - warmup_steps)
        progress = min(1.0, max(0.0, (step + 1 - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    stage_dir = run_dir / stage_name
    events = stage_dir / "events.jsonl"
    stage_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(stage_dir / "stage_manifest.json", {
        "schema": "falcon-submission-stage-v1",
        "stage": stage_name,
        "config": stage_cfg,
        "mixture": stage_mixture,
        "world_size": world_size,
        "per_device_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": per_device_batch * world_size * grad_accum,
        "sampler": {"seed": stage_seed, "num_samples_per_rank": num_samples, "bucket_row_counts": sampler.bucket_row_counts},
    }) if rank == 0 else None
    iterator = iter(loader)
    model.train()
    completed = 0
    seen_loss_tokens = Counter()
    stage_started = time.monotonic()
    emit(events, "stage_start", stage=stage_name, optimizer_steps=steps, learning_rate=stage_cfg["learning_rate"], mixture=stage_mixture, rank=rank)
    try:
        while completed < steps:
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for micro in range(grad_accum):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                sync_context = model.no_sync() if hasattr(model, "no_sync") and micro < grad_accum - 1 else _null_context()
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(**batch)
                        loss = output.loss / grad_accum
                    scaler.scale(loss).backward()
                loss_value += float(loss.detach().float().cpu())
                supervised = int((batch["labels"] != -100).sum().detach().cpu())
                seen_loss_tokens["batch"] += supervised
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(config["training"]["max_grad_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            completed += 1
            current_lr = float(optimizer.param_groups[0]["lr"])
            emit(events, "optimizer_step", stage=stage_name, step=completed, loss=loss_value, learning_rate=current_lr, elapsed_seconds=round(time.monotonic() - stage_started, 3), supervised_tokens=int(sum(seen_loss_tokens.values())), rank=rank)
            if completed in {int(value) for value in stage_cfg["checkpoint_steps"]}:
                barrier(torch, world_size)
                checkpoint = save_checkpoint(model, optimizer, scheduler, scaler, stage_dir / "checkpoints", completed, stage_name, config_path, data_path, sampler, torch, rank)
                barrier(torch, world_size)
                if rank == 0:
                    emit(events, "checkpoint_saved", stage=stage_name, step=completed, checkpoint=str(checkpoint))
                report = fast_dev_validate(model, tokenizer, fast_battery, run_dir / "fast-dev", stage_name, completed, device, torch, int(config["training"]["fast_dev_max_new_tokens"]))
                if rank == 0:
                    emit(events, "fast_dev_complete", stage=stage_name, step=completed, invalid=report["invalid"], critical_failures=report["critical_failures"])
                barrier(torch, world_size)
    except Exception as exc:
        if is_oom(exc) and completed == 0:
            if rank == 0:
                emit(events, "initial_batch_oom", stage=stage_name, fallback_batch_size=1, fallback_gradient_accumulation_steps=8)
            raise InitialBatchOOM(str(exc)) from exc
        if rank == 0:
            emit(events, "stage_failed", stage=stage_name, step=completed, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        del loader
    if rank == 0:
        emit(events, "stage_complete", stage=stage_name, optimizer_steps=completed)
    barrier(torch, world_size)
    return {"stage": stage_name, "optimizer_steps": completed, "final_checkpoint": str(stage_dir / "checkpoints" / f"checkpoint-{completed}"), "mixture": stage_mixture}


def run_stage_with_oom_fallback(**kwargs: Any) -> dict[str, Any]:
    """Use batch 1/accumulation 8 only after a real initial batch-2 CUDA OOM."""
    try:
        return run_stage(**kwargs, per_device_batch_size=2, gradient_accumulation_steps=4)
    except InitialBatchOOM:
        torch = kwargs["torch"]
        model = kwargs["model"]
        for parameter in model.parameters():
            parameter.grad = None
        torch.cuda.empty_cache()
        return run_stage(**kwargs, per_device_batch_size=1, gradient_accumulation_steps=8)


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def load_model_once(config: dict[str, Any], device: Any, torch: Any) -> Any:
    # Compatibility patch is needed by the pinned PEFT/Transformers pair, but
    # it does not change the model implementation or enable a fallback path.
    from scripts.train_lora import patch_peft_transformers_compat

    patch_peft_transformers_compat()
    from peft import LoraConfig, get_peft_model
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1ForCausalLM

    model = FalconH1ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.config.use_cache = False
    model_targets = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = [name for name in TARGET_MODULES if name not in model_targets]
    if missing:
        raise RuntimeError(f"approved LoRA targets are absent from Falcon-H1: {missing}")
    if any(name in model_targets for name in FORBIDDEN_MODULES):
        # Presence in the base model is expected.  The PEFT target list below
        # is the enforcement boundary; this assertion documents the distinction.
        pass
    lora = config["lora"]
    adapter = get_peft_model(model, LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(TARGET_MODULES),
    ))
    adapter.config.use_cache = False
    return adapter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/falcon-production-v1.json"))
    parser.add_argument("--data", default=str(ROOT / "output/falcon-submission-sft-v1.jsonl"))
    parser.add_argument("--run-dir", default=str(ROOT / "experiments/falcon-submission-sft-v1"))
    parser.add_argument("--dry-run", action="store_true", help="run hardware/tokenizer/pre-tokenization checks, never load model weights")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    data_path = Path(args.data).resolve()
    run_dir = Path(args.run_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if not data_path.is_file():
        raise FileNotFoundError(f"submission SFT corpus is missing; run build_falcon_submission_sft.py: {data_path}")

    import torch

    # This is intentionally before AutoTokenizer and before every Falcon model
    # import/load.  A P100 must fail here, not spend time in naive Mamba.
    fast_path = require_fast_mamba(torch)
    world_size, rank, local_rank, device = distributed_info(torch)
    set_seed(int(config["training"]["seed"]) + rank, torch)
    root_events = run_dir / "events.jsonl"
    if rank == 0:
        emit(root_events, "startup", model=MODEL_ID, revision=MODEL_REVISION, hardware=fast_path, world_size=world_size, dry_run=args.dry_run)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    rows = load_jsonl(data_path)
    dataset = tokenize_and_pack(rows, tokenizer, 512)
    if rank == 0:
        atomic_json(run_dir / "pretokenized_manifest.json", {
            "schema": "falcon-submission-pretokenized-v1",
            "data": str(data_path),
            "data_sha256": sha256_file(data_path),
            "rows": len(rows),
            "token_summary": dataset.token_summary(),
            "max_length": 512,
            "assistant_only_loss": True,
            "gpu_loop_inputs": ["input_ids", "attention_mask", "labels"],
        })
    if args.dry_run:
        if rank == 0:
            print(json.dumps({"hardware": fast_path, "rows": len(rows), "token_summary": dataset.token_summary()}, indent=2))
        barrier(torch, world_size)
        return 0

    # Exactly one base load and one adapter creation for all three stages.
    model = load_model_once(config, device, torch)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    if rank == 0:
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        emit(root_events, "model_ready", trainable_parameters=trainable, total_parameters=total, target_modules=list(TARGET_MODULES), forbidden_targets=list(FORBIDDEN_MODULES), dtype="float16", gradient_checkpointing=False)

    fast_battery = (ROOT / config["training"]["fast_dev_battery"]).resolve()
    stage_reports = []
    for stage_name in STAGES:
        stage_cfg = config["stages"][stage_name]
        stage_mixture = config["data"]["mixture"][stage_cfg["mixture"]]
        report = run_stage_with_oom_fallback(
            model=model, tokenizer=tokenizer, dataset=dataset, config=config,
            config_path=config_path, data_path=data_path, run_dir=run_dir,
            stage_name=stage_name, stage_cfg=stage_cfg, stage_mixture=stage_mixture,
            world_size=world_size, rank=rank, device=device, torch=torch,
            fast_battery=fast_battery,
        )
        stage_reports.append(report)
    if rank == 0:
        atomic_json(run_dir / "trajectory_manifest.json", {
            "schema": "falcon-submission-trajectory-v1",
            "status": "training_complete",
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "hardware": fast_path,
            "config_sha256": sha256_file(config_path),
            "data_sha256": sha256_file(data_path),
            "adapter_commit": git_revision(),
            "adapter_trajectory": "one_adapter_no_model_reload_between_stages",
            "total_optimizer_steps": 136,
            "candidates": {
                "Stage1-step96": str(run_dir / "stage1_domain_lock_in/checkpoints/checkpoint-96"),
                "Stage2-final": str(run_dir / "stage2_safety_polish/checkpoints/checkpoint-24"),
                "Stage3-final": str(run_dir / "stage3_capability_replay/checkpoints/checkpoint-16"),
            },
            "stage_reports": stage_reports,
        })
        emit(root_events, "training_complete", total_optimizer_steps=136, candidates=3)
    barrier(torch, world_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
