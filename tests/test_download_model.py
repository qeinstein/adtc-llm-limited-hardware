"""Downloader portability tests (no network, no weights).

Guards the `make model` path against bash-4-only constructs: macOS ships
bash 3.2, where `${VAR,,}` dies with "bad substitution".
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "download_model.sh"

# Bash-4+ parameter expansions / builtins with no bash 3.2 equivalent.
BASH4_ONLY = [
    r"\$\{[A-Za-z_][A-Za-z0-9_]*,,",  # ${VAR,,} / ${VAR,,pat}
    r"\$\{[A-Za-z_][A-Za-z0-9_]*\^",  # ${VAR^} / ${VAR^^}
    r"(?m)^\s*mapfile\b",
    r"(?m)^\s*readarray\b",
    r"(?m)^\s*coproc\b",
]


def test_downloader_has_no_bash4_only_constructs():
    """The script must stay parseable by bash 3.2 (macOS system bash)."""
    lines = [ln for ln in SCRIPT.read_text(encoding="utf-8").splitlines()
             if not ln.strip().startswith("#")]
    text = "\n".join(lines)
    for pat in BASH4_ONLY:
        assert not re.search(pat, text), f"bash-4-only construct: {pat}"


def test_downloader_uses_an_immutable_model_url():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "/resolve/main/" not in text
    assert "e938cd2af04dd5f30731922bc4780ef2f264f032" in text


def test_downloader_keeps_hash_verification():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'MODEL_FILE="$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"' in text
    assert 'MODEL_URL="https://huggingface.co/Fluxx08/' in text
    assert "MODEL_SHA256" in text
    assert "MIN_SIZE=12000000000" in text
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


def test_cross_platform_downloader_uses_the_same_immutable_url():
    text = (ROOT / "scripts" / "download_model.py").read_text(encoding="utf-8")
    assert "/resolve/main/" not in text
    assert "e938cd2af04dd5f30731922bc4780ef2f264f032" in text
