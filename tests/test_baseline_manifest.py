import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "baselines" / "semifinal.json"


def test_semifinal_baseline_has_immutable_artifact_identity():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert data["status"] == "immutable"
    artifact = data["artifact"]
    assert artifact["model_revision"] in artifact["immutable_url"]
    assert "/resolve/main/" not in artifact["immutable_url"]
    assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
    assert artifact["size_bytes"] > 350_000_000


def test_semifinal_baseline_records_metric_provenance_limits():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert data["repository"]["profiler_reported_git_sha"]
    assert data["reported_metrics"]["system"]["artifact_linkage"]
    assert data["known_reproducibility_gaps"]
