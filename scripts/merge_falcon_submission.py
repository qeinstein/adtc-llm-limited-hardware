#!/usr/bin/env python3
"""Merge the selected Falcon adapter and verify adapter/merged equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_ID = "tiiuae/Falcon-H1-1.5B-Deep-Instruct"
MODEL_REVISION = "b6648636ddc906688974282de6e7a243395f5423"
FORBIDDEN = {"out_proj", "conv1d"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_targets(config: dict[str, Any]) -> list[str]:
    values = config.get("target_modules", [])
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("adapter target_modules must be a list")
    return [str(value) for value in values]


def assert_safe_adapter_config(path: Path) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    targets = adapter_targets(config)
    forbidden = sorted({target for target in targets if target in FORBIDDEN or target.rsplit(".", 1)[-1] in FORBIDDEN})
    if forbidden:
        raise ValueError(f"adapter contains forbidden LoRA targets: {forbidden}")
    serialized_targets = json.dumps(targets, sort_keys=True).casefold()
    if any(word in serialized_targets for word in FORBIDDEN):
        raise ValueError("adapter target configuration contains out_proj or conv1d")
    return config


def _chat_inputs(tokenizer: Any, device: Any) -> dict[str, Any]:
    from scripts.build_falcon_submission_sft import SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "A child is breathing very fast and has chest indrawing. What is the immediate disposition?"},
    ]
    try:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt",
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        return {key: value.to(device) for key, value in encoded.items()}
    return {"input_ids": encoded.to(device), "attention_mask": encoded.new_ones(encoded.shape)}


def deterministic_generation(model: Any, tokenizer: Any, torch: Any) -> dict[str, Any]:
    from scripts.falcon_format import generation_stop_ids

    device = next(model.parameters()).device
    inputs = _chat_inputs(tokenizer, device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    stop_ids = generation_stop_ids(tokenizer, model.generation_config)
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            eos_token_id=stop_ids or None,
            pad_token_id=tokenizer.pad_token_id,
        )
    ids = [int(value) for value in output[0, prompt_len:].detach().cpu().tolist()]
    return {"token_ids": ids, "text": tokenizer.decode(ids, skip_special_tokens=True), "stop_ids": stop_ids}


def git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--stock", action="store_true", help="export the pinned stock Falcon emergency fallback")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", default=str(ROOT / "configs/falcon-production-v1.json"), type=Path)
    parser.add_argument("--adapter-commit", default=None)
    args = parser.parse_args(argv)
    if bool(args.adapter) == bool(args.stock):
        raise ValueError("specify exactly one of --adapter or --stock")
    adapter = args.adapter.resolve() if args.adapter else None
    adapter_config = assert_safe_adapter_config(adapter) if adapter else None
    config_path = args.config.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty merged output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer
    from transformers.models.falcon_h1.modeling_falcon_h1 import FalconH1ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FalconH1ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    model.eval()
    before = None
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        before = deterministic_generation(model, tokenizer, torch)
        model = model.merge_and_unload()
    after = deterministic_generation(model, tokenizer, torch)
    if before is not None and before["text"] != after["text"]:
        raise RuntimeError("adapter-on and merged deterministic generations differ")
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    manifest = {
        "schema": "falcon-submission-merge-v1",
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "adapter": str(adapter) if adapter else None,
        "adapter_commit": args.adapter_commit or git_revision(),
        "adapter_config_targets": adapter_targets(adapter_config) if adapter_config else [],
        "training_config_sha256": sha256_file(config_path),
        "merged_hf": str(out),
        "adapter_on_generation": before,
        "merged_generation": after,
        "adapter_merged_equivalent": before is None or before["text"] == after["text"],
        "merged_files": {
            str(path.relative_to(out)): sha256_file(path)
            for path in sorted(out.rglob("*")) if path.is_file()
        },
    }
    (out / "merge_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("base_model", "base_revision", "adapter", "adapter_merged_equivalent", "training_config_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
