"""JOIN4 patch-anchor integrity: the bench lazy-env compat anchor must match
the frozen pin's parser text exactly once (replace_once would raise
otherwise, failing the runtime build; this test fails faster, offline)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_apply():
    spec = importlib.util.spec_from_file_location(
        "join4_apply", ROOT / "probes/edge0_port/join4_apply.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["join4_apply"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bench_lazy_env_anchor_matches_pin_once():
    a = _load_apply()
    fixture = (ROOT / "tests/fixtures/pin_bench_lazy_defaults.txt").read_text()
    assert fixture.count(a.BENCH_LAZY_ENV_ANCHOR) == 1


def test_bench_lazy_env_patch_maps_all_modes():
    a = _load_apply()
    p = a.BENCH_LAZY_ENV_PATCH
    assert 'getenv("LLAMA_ARG_LAZY_MODE")' in p
    for mode in ("LLAMA_LAZY_MODE_ON", "LLAMA_LAZY_MODE_AUTO",
                 "LLAMA_LAZY_MODE_OFF"):
        assert mode in p
    assert "cmd_params_defaults.lazy_mode" in p  # default preserved


def test_bounded_cache_has_apple_mmap_path():
    """The web demo is routinely built on Apple Silicon as well as Linux.

    The phase-6 header is embedded into llama.cpp by the build script, so a
    source-level contract test catches a platform regression before a 12 GB
    model download and build are attempted.
    """
    header = (ROOT / "probes/edge0_port/join4_phase6.h").read_text()
    assert "defined(__linux__) || defined(__APPLE__)" in header
    assert "MAP_ANONYMOUS MAP_ANON" in header
    assert "int phase6_map_flags = MAP_PRIVATE | MAP_ANONYMOUS;" in header


def test_bounded_cache_has_windows_path():
    header = (ROOT / "probes/edge0_port/join4_phase6.h").read_text()
    for marker in (
        "#include <windows.h>",
        "VirtualAlloc",
        "CreateThread",
        "WaitForSingleObject",
        "static phase6_ssize_t phase6_pread",
        "SwitchToThread",
    ):
        assert marker in header


def test_native_windows_build_and_download_entrypoints_exist():
    assert (ROOT / "scripts" / "build_runtime.py").is_file()
    assert (ROOT / "scripts" / "download_model.py").is_file()
