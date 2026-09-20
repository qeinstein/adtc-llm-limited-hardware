"""Data pipeline unit tests: MinHash/LSH behavior, contamination screen
positive control, quality-filter rules. No network, no datasets lib."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import build_dataset as bd


def test_minhash_identical_and_distant():
    a = bd.minhash(bd.shingles("the quick brown fox jumps over the lazy dog"))
    b = bd.minhash(bd.shingles("the quick brown fox jumps over the lazy dog"))
    c = bd.minhash(bd.shingles("quantum chromodynamics lattice gauge fixing"))
    assert bd.jacc_est(a, b) == 1.0
    assert bd.jacc_est(a, c) < 0.3


def test_lsh_finds_near_dup():
    lsh = bd.LSH()
    words = ("oral rehydration solution recipe for infant diarrhoea treatment "
             "at home with clean water sugar salt zinc supplements and follow "
             "up care instructions for caregivers in rural clinics every day "
             "while monitoring hydration status weight gain and urine output "
             "closely until full recovery is confirmed by examination").split()
    base = " ".join(words)
    lsh.add(bd.minhash(bd.shingles(base)))
    # exact duplicate: deterministic at any threshold
    assert lsh.query(bd.minhash(bd.shingles(base))) == [0]
    # 1-word change: J≈0.88; detected deterministically at 0.7
    near = " ".join(["ORS"] + words[1:])
    assert lsh.query(bd.minhash(bd.shingles(near)), thresh=0.7) == [0]
    far = "lattice quantum chromodynamics gauge fixing algorithms"
    assert lsh.query(bd.minhash(bd.shingles(far))) == []


def test_contamination_positive_control():
    quar = [{"src": "t", "src_id": "e1",
             "sha": bd.sha12(bd.norm("what causes rice water diarrhoea")),
             "text": bd.norm("what causes rice water diarrhoea")}]
    recs = [{"src": "s", "src_id": "x",
             "prompt": "What causes rice-water diarrhoea?",
             "response": "cholera", "kind": "saq", "converted": False}]
    counts = Counter()
    kept, report = bd.contamination_screen(recs, quar, counts)
    assert kept == [] and report["exact_hits"] == 1
    clean = [{"src": "s", "src_id": "y", "prompt": "how to splint a finger",
              "response": "immobilize it", "kind": "saq", "converted": False}]
    kept2, _ = bd.contamination_screen(clean, quar, Counter())
    assert len(kept2) == 1


def test_quality_pii_and_numeric_guard():
    counts, review = Counter(), []
    bad = [{"src": "s", "src_id": "pii", "prompt": "how can I call the clinic",
            "response": "reach me at joe@example.com please", "kind": "saq",
            "converted": False}]
    assert bd.quality_filter(bad, counts, review) == []
    assert counts["q_pii"] == 1
    conv = [{"src": "s", "src_id": "num", "prompt": "which dose is used here",
             "response": "give 500 mg twice daily for seven days until better",
             "kind": "mcq-conv", "converted": True,
             "source_fields": ["which dose is used here", "A", "B", "plain text"]}]
    assert bd.quality_filter(conv, counts, review) == []
    assert review and review[-1]["reason"] == "converted-numerics-not-in-source"


def test_ledger_schema():
    import json

    p = Path(__file__).resolve().parent.parent / "data" / "LICENSE_LEDGER.json"
    if not p.exists():
        import pytest
        pytest.skip("pipeline not run yet")
    ledger = json.loads(p.read_text())
    assert set(ledger["sources"]) >= {"afrimed", "medqa", "medmcqa", "pubmedqa", "oasst1"}
    for name, src in ledger["sources"].items():
        for k in ("repo", "revision", "license", "train_use"):
            assert src.get(k), (name, k)
    assert ledger["counts"]["afrimed_quarantine"] == 5903
