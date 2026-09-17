"""Matched bounded selective-Q4 follow-up with CPU repacking disabled.

The v1 bounded harness accidentally omitted --no-repack even though the
resident representation A/B used it.  This wrapper reuses the committed v1
harness, changes only that confounder plus timing precision, and executes the
same three-repeat control/challenger experiment.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


WORK = Path("/kaggle/working")
WRAP = Path("/tmp/native-sparse-q4-bounded-v2-wrapper")
BASE = WRAP / "q4_bounded_v2_expanded.py"
RESEARCH_COMMIT = "cae43e5fb930ec562bf5749c7df5cb0d8827a911"
BASE_URL = (
    "https://raw.githubusercontent.com/qeinstein/adtc-llm-limited-hardware/"
    f"{RESEARCH_COMMIT}/kaggle/native-sparse-q4-bounded-v1/q4_bounded_v1.py"
)


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one source anchor, found {count}: {old[:80]!r}")
    return text.replace(old, new)


def main() -> None:
    WRAP.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(BASE_URL, BASE)
    source = BASE.read_text(encoding="utf-8")
    original_sha = hashlib.sha256(source.encode()).hexdigest()
    source = source.replace("native-sparse-q4-bounded-v1", "native-sparse-q4-bounded-v2")
    source = replace_once(
        source,
        '           "--single-turn", "--no-display-prompt", "--no-warmup", "--perf",\n'
        '           "-lm", "mmap", "-lzm", "on" if bounded else "off", "-p", PROMPT]',
        '           "--single-turn", "--no-display-prompt", "--no-warmup", "--perf",\n'
        '           "--no-repack", "-lm", "mmap", "-lzm", "on" if bounded else "off", "-p", PROMPT]',
    )
    source = replace_once(
        source,
        '    return {"runtime": patch_runtime(), "quantizer": patch_quantizer()}\n',
        '    runtime = patch_runtime()\n'
        '    quantizer = patch_quantizer()\n'
        '    replace_once(LLAMA / "tools/cli/cli-context.cpp",\n'
        '        "[ Prompt: %.1f t/s | Generation: %.1f t/s ]",\n'
        '        "[ Prompt: %.6f t/s | Generation: %.6f t/s ]")\n'
        '    return {"runtime": runtime, "quantizer": quantizer,\n'
        '            "no_repack_matched": True, "precise_cli_timings": True}\n',
    )
    source = source.replace(
        'RESEARCH_SOURCE_COMMIT = "8ceed7d"',
        f'RESEARCH_SOURCE_COMMIT = "{RESEARCH_COMMIT}"',
    )
    source = source.replace(
        '"The validated selective Q4 attention/GDN challenger should retain its resident speed win under a real byte-bounded expert cache."',
        '"Disabling the accidental Q4_K CPU repack should remove the v1 anonymous-RSS penalty while preserving the selective-Q4 bounded speed gain."',
    )
    BASE.write_text(source, encoding="utf-8")
    print(f"expanded v1 source sha256={original_sha}", flush=True)
    namespace = {"__name__": "__main__", "__file__": str(BASE)}
    exec(compile(source, str(BASE), "exec"), namespace)


if __name__ == "__main__":
    main()
