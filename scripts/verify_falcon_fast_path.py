#!/usr/bin/env python3
"""Fail-closed verification of Falcon-H1's CUDA Mamba fast path.

This command intentionally runs before any Falcon tokenizer or model load.  A
P100/sm60 or an installation that only exposes the Transformers naive Mamba
implementation is not a valid submission-training environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def cuda_capabilities(torch: Any) -> list[tuple[int, int]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Falcon-H1 submission training requires an sm75+ NVIDIA GPU")
    return [tuple(int(part) for part in torch.cuda.get_device_capability(index)) for index in range(torch.cuda.device_count())]


def verify_fast_path(minimum: tuple[int, int] = (7, 5)) -> dict[str, Any]:
    import torch

    capabilities = cuda_capabilities(torch)
    too_old = [capability for capability in capabilities if capability < minimum]
    if too_old:
        raise RuntimeError(
            f"CUDA capability {too_old} is below the required sm{minimum[0]}{minimum[1]} minimum; "
            "refusing to load Falcon-H1"
        )
    try:
        import mamba_ssm
        from mamba_ssm import Mamba2
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        import causal_conv1d
        from causal_conv1d import causal_conv1d_fn
    except Exception as exc:  # noqa: BLE001 - this is the fail-closed boundary
        raise RuntimeError(
            "optimized Falcon-H1 Mamba path unavailable: install compatible "
            "mamba-ssm and causal-conv1d before model loading"
        ) from exc
    if not callable(Mamba2) or not callable(selective_scan_fn) or not callable(causal_conv1d_fn):
        raise RuntimeError("optimized Mamba/causal-conv1d imports are present but not callable")
    result = {
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "device_count": torch.cuda.device_count(),
        "device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "capabilities": [list(value) for value in capabilities],
        "minimum_capability": list(minimum),
        "mamba_path": "optimized_mamba_ssm_causal_conv1d",
        "mamba_ssm": getattr(mamba_ssm, "__version__", "unknown"),
        "causal_conv1d": getattr(causal_conv1d, "__version__", "unknown"),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-major", type=int, default=7)
    parser.add_argument("--minimum-minor", type=int, default=5)
    args = parser.parse_args(argv)
    result = verify_fast_path((args.minimum_major, args.minimum_minor))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
