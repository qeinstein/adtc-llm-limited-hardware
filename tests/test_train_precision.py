"""Precision/compile routing for train_lora (pure logic, no torch/GPU needed)."""

from scripts.train_lora import (
    compile_supported,
    get_cuda_capability,
    parse_args,
    resolve_precision,
)


class _FakeCuda:
    @staticmethod
    def get_device_capability(index):
        assert index == 0
        return (6, 0)


class _FakeTorch:
    cuda = _FakeCuda()


def test_cuda_capability_uses_public_pytorch_api():
    assert get_cuda_capability(_FakeTorch) == (6, 0)


def test_defaults_preserve_locked_recipe():
    # Defaults must reproduce the QLoRA recipe exactly (4bit + bf16 on CUDA).
    assert parse_args([]).quantize == "4bit"
    assert parse_args([]).compute_dtype == "auto"
    assert parse_args([]).torch_compile is False
    assert parse_args([]).optim is None
    assert resolve_precision(use_cuda=True, cuda_capability=(8, 0)) == ("bnb4", "bf16")


def test_p100_falls_back_to_plain_fp16():
    # sm_60 (P100): bitsandbytes is unsupported; unquantized fp16 is the path.
    assert resolve_precision(use_cuda=True, cuda_capability=(6, 0),
                             quantize="none") == ("plain", "fp16")


def test_modern_gpu_plain_defaults_to_bf16():
    assert resolve_precision(use_cuda=True, cuda_capability=(8, 9),
                             quantize="none") == ("plain", "bf16")
    assert resolve_precision(use_cuda=True, cuda_capability=(7, 5),
                             quantize="none") == ("plain", "fp16")


def test_explicit_dtype_overrides_auto():
    assert resolve_precision(use_cuda=True, cuda_capability=(8, 0),
                             quantize="none", compute_dtype="fp32") == ("plain", "fp32")
    assert resolve_precision(use_cuda=True, cuda_capability=(6, 0),
                             compute_dtype="fp16") == ("bnb4", "fp16")


def test_no_cuda_paths():
    assert resolve_precision(use_cuda=False, mps=True) == ("plain", "fp16")
    assert resolve_precision(use_cuda=False, mps=False) == ("plain", "fp32")


def test_compile_gate():
    assert compile_supported(use_cuda=True, cuda_capability=(8, 0)) is True
    assert compile_supported(use_cuda=True, cuda_capability=(7, 0)) is True
    assert compile_supported(use_cuda=True, cuda_capability=(6, 0)) is False
    assert compile_supported(use_cuda=False) is False
