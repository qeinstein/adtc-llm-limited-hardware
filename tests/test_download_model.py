"""Downloader portability tests (no network, no weights).

Guards the `make model` path against bash-4-only constructs: macOS ships
bash 3.2, where `${VAR,,}` dies with "bad substitution".
"""
import hashlib
import re
import shutil
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


def test_verify_accepts_uppercase_env_hash(tmp_path):
    """verify_model lowercases MODEL_SHA256 (the old `${VAR,,}` behavior).

    Runs a copy of the script with MIN_SIZE lowered against a tiny
    fixture; the script must take the already-verified exit-0 path.
    """
    work = tmp_path / "dl"
    work.mkdir()
    copy = work / "download_model.sh"
    shutil.copy(SCRIPT, copy)
    text = copy.read_text(encoding="utf-8")
    assert "MIN_SIZE=12000000000" in text
    copy.write_text(text.replace("MIN_SIZE=12000000000", "MIN_SIZE=1"),
                    encoding="utf-8")
    model_dir = work / "model"
    model_dir.mkdir()
    fixture = model_dir / "Qwen3.6-35B-A3B-UD-Q2K-experts.gguf"
    fixture.write_bytes(b"fake-gguf-bytes")
    digest = hashlib.sha256(b"fake-gguf-bytes").hexdigest().upper()
    assert digest != digest.lower()  # fixture hash must exercise lowercasing
    proc = subprocess.run(
        ["bash", str(copy)], cwd=work, capture_output=True, text=True,
        timeout=120, env={"PATH": "/usr/bin:/bin", "MODEL_SHA256": digest,
                          "OSTYPE": "linux-gnu"})
    assert proc.returncode == 0, proc.stderr
    assert "verified" in proc.stdout
