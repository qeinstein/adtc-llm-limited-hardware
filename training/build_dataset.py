#!/usr/bin/env python3
"""Licensed, versioned post-training data pipeline for Jamii Afya.

Reads: pinned HF revisions (see SOURCES).
Writes: data/processed/*.jsonl, data/manifests/*.jsonl|json,
  data/contamination_report.json, data/requires_clinician_review.jsonl,
  data/LICENSE_LEDGER.json.

Rules enforced here (not in prompts):
- AfriMed-QA v2: internal `split` column; train slice only, and only rows
  WITH gold responses (saq rationale + mcq; consumer queries lack answers
  and are eval-bank only). Test slice hashed/quarantined, never trained.
- MedQA/MedMCQA: TRAIN splits only; val/test quarantined.
- PubMedQA pqa_labeled (official 1k heldout): EVAL ONLY. pqa_artificial
  excluded (low-confidence generated, unverifiable).
- MMLU test (cais/mmlu): EVAL ONLY quarantine (contamination screen).
- rated_bias (AfriMed): direction AMBIGUOUS (higher in quality=True rows)
  -> never used as a filter. All other rating directions verified against
  the quality flag (see data/DATA_CARD.md).
- MCQ->conversational conversion ONLY with source rationale; converted
  numerics must all appear in source fields or the record goes to
  requires_clinician_review.jsonl instead of training.
- No synthetic CoT, no invented explanations, no dose/unit changes.

Usage (from repo root, data venv):
  HF_DATASETS_CACHE=data/.cache /tmp/venv-data/bin/python training/build_dataset.py [--pilot-only]
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
MANI = DATA / "manifests"

SOURCES = {
    "afrimed": {
        "repo": "afrimedqa/afrimedqa_v2",
        "revision": "3b4382fa0bb51bfc026f5813021ab0ec7be9de8f",
        "license": "CC-BY-4.0",
        "file": "afri_med_qa_15k_v2.4_phase_2_15275.csv",
        "citation": "AfriMed-QA v2 (Intron Health / AfriMed-QA consortium)",
        "train_use": "split==train AND quality==True AND rated filters AND response present",
        "redistributable": "yes (CC-BY-4.0, attribution required)",
    },
    "medqa": {
        "repo": "GBaker/MedQA-USMLE-4-options",
        "revision": "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
        "license": "CC-BY-4.0 (dataset card; upstream MedQA Jin et al.)",
        "train_use": "train split; short-answer pairs (no source rationale -> no invented explanation)",
        "redistributable": "yes (CC-BY-4.0, attribution required)",
    },
    "medmcqa": {
        "repo": "openlifescienceai/medmcqa",
        "revision": "91c6572c454088bf71b679ad90aa8dffcd0d5868",
        "license": "Apache-2.0",
        "train_use": "train split, single-choice, stratified sample (cap 12000)",
        "redistributable": "yes (Apache-2.0)",
    },
    "pubmedqa": {
        "repo": "qiaojin/PubMedQA",
        "revision": "9001f2853fb87cab8d220904e0de81ac6973b318",
        "license": "MIT",
        "train_use": "NONE (pqa_labeled is the official heldout -> eval-only; pqa_artificial excluded as unverifiable)",
        "redistributable": "eval artifacts only",
    },
    "oasst1": {
        "repo": "OpenAssistant/oasst1",
        "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "license": "Apache-2.0",
        "train_use": "vetted EN assistant turns (reviewed, rank 0, not deleted/synthetic), cap 2500",
        "redistributable": "yes (Apache-2.0)",
    },
    "mmlu": {
        "repo": "cais/mmlu",
        "revision": None,  # resolved at build; test split only, quarantine only
        "license": "MIT",
        "train_use": "NONE - quarantine/contamination screen only",
        "redistributable": "no (hashes bank only)",
    },
}

MEDMCQA_CAP = 12000
OASST_CAP = 2500
PILOT_N = 1500


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  fetch {url[:100]}...", flush=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def adapt_afrimed(counts: Counter) -> tuple[list[dict], list[dict]]:
    """Returns (train_records, quarantine_prompts)."""
    src = SOURCES["afrimed"]
    url = (f"https://huggingface.co/datasets/{src['repo']}/resolve/"
           f"{src['revision']}/{src['file']}")
    path = fetch(url, DATA / ".cache" / src["file"])
    train, quar = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            counts["afrimed_rows"] += 1
            q = (r.get("question_clean") or r.get("question") or "").strip()
            if r.get("split") != "train":
                if q:
                    quar.append({"src": "afrimed", "src_id": r["sample_id"],
                                 "sha": sha12(norm(q)), "text": norm(q)})
                    counts["afrimed_quarantine"] += 1
                continue
            counts["afrimed_train_rows"] += 1
            if not q:
                counts["afrimed_empty_q"] += 1
                continue
            # Vendor review metadata is sparse (quality 28% labeled, rated_*
            # 0.7%): drop RECORDED-bad, keep missing (our own quality
            # pipeline still applies). rated_bias ignored (ambiguous).
            qflag = (r.get("quality") or "").strip().lower()
            if qflag in ("false", "0"):
                counts["afrimed_quality_false"] += 1
                continue

            def _num(key):
                try:
                    v = (r.get(key) or "").strip()
                    return float(v) if v else None
                except ValueError:
                    return None

            harmful = _num("rated_harmful")
            hallucin = _num("rated_hallucination")
            correct = _num("rated_correct")
            omission = _num("rated_omission")
            reasonable = _num("rated_reasonable")
            if harmful == 1 or (hallucin is not None and hallucin > 1) \
                    or (correct is not None and correct < 4) \
                    or (omission is not None and omission > 1) \
                    or (reasonable is not None and reasonable < 4):
                counts["afrimed_rating_drop"] += 1
                continue
            qt = r.get("question_type")
            rationale = (r.get("answer_rationale") or "").strip()
            if qt == "saq":
                if not rationale:
                    counts["afrimed_no_response"] += 1
                    continue
                train.append({"src": "afrimed", "src_id": r["sample_id"],
                              "kind": "saq", "prompt": q, "response": rationale,
                              "converted": False,
                              "meta": {"specialty": r.get("specialty"),
                                       "tier": r.get("tier")}})
            elif qt == "mcq":
                letter = (r.get("correct_answer") or "").strip().upper()[:1]
                try:
                    import ast
                    opts = ast.literal_eval(r.get("answer_options") or "[]")
                except Exception:
                    opts = []
                if letter not in "ABCD" or len(opts) != 4 or len(rationale) < 20:
                    counts["afrimed_mcq_unusable"] += 1
                    continue
                prompt = (f"{q}\nA. {opts[0]}\nB. {opts[1]}\n"
                          f"C. {opts[2]}\nD. {opts[3]}")
                response = (f"The correct answer is {letter}) "
                            f"{opts['ABCD'.index(letter)]}.\n\n{rationale}")
                train.append({"src": "afrimed", "src_id": r["sample_id"],
                              "kind": "mcq-conv", "prompt": prompt,
                              "response": response, "converted": True,
                              "source_fields": [q] + [str(o) for o in opts]
                              + [rationale],
                              "meta": {"specialty": r.get("specialty")}})
            else:  # consumer_queries: no gold response -> never SFT
                counts["afrimed_consumer_no_gold"] += 1
    return train, quar


def adapt_medqa(counts: Counter) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset
    src = SOURCES["medqa"]
    train, quar = [], []
    for split in ("train", "test"):
        ds = load_dataset(src["repo"], split=split, revision=src["revision"],
                          trust_remote_code=False)
        for r in ds:
            counts[f"medqa_{split}_rows"] += 1
            q = (r.get("question") or "").strip()
            if split != "train":
                if q:
                    quar.append({"src": "medqa", "src_id": str(r.get("id", "")),
                                 "sha": sha12(norm(q)), "text": norm(q)})
                    counts["medqa_quarantine"] += 1
                continue
            opts = r.get("options") or {}
            ans = (r.get("answer") or "").strip()
            if not q or not ans:
                counts["medqa_empty"] += 1
                continue
            # No source rationale: short-answer pair, no invented explanation.
            opt_text = ""
            if isinstance(opts, dict) and opts:
                letters = sorted(opts)
                opt_text = "\n".join(f"{L}. {opts[L]}" for L in letters)
            prompt = q if not opt_text else f"{q}\n{opt_text}"
            train.append({"src": "medqa", "src_id": str(r.get("id", "")),
                          "kind": "short-answer", "prompt": prompt,
                          "response": ans, "converted": False, "meta": {}})
    return train, quar


def adapt_medmcqa(counts: Counter) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset
    src = SOURCES["medmcqa"]
    train, quar = [], []
    for split in ("train", "validation", "test"):
        ds = load_dataset(src["repo"], split=split, revision=src["revision"],
                          trust_remote_code=False)
        pool = []
        for r in ds:
            counts[f"medmcqa_{split}_rows"] += 1
            q = (r.get("question") or "").strip()
            if split != "train":
                if q:
                    quar.append({"src": "medmcqa",
                                 "src_id": str(r.get("id", "")),
                                 "sha": sha12(norm(q)), "text": norm(q)})
                    counts["medmcqa_quarantine"] += 1
                continue
            if (r.get("choice_type") or "") != "single":
                counts["medmcqa_multi"] += 1
                continue
            opts = [str(r.get(f"op{c}", "") or "") for c in "abcd"]
            try:
                cop = "abcd"[int(r.get("cop", -1))]
            except (ValueError, TypeError, IndexError):
                cop = str(r.get("cop", "") or "").strip().lower()
            exp = (r.get("exp") or "").strip()
            if not q or cop not in "abcd" or len(exp) < 20 \
                    or any(not o for o in opts):
                counts["medmcqa_unusable"] += 1
                continue
            pool.append(r)
        if split != "train":
            continue
        # Stratified round-robin sample so MedMCQA cannot dominate.
        rng = random.Random(7)
        by_subj: dict[str, list] = defaultdict(list)
        for r in pool:
            by_subj[r.get("subject_name") or "Unknown"].append(r)
        for rs in by_subj.values():
            rng.shuffle(rs)
        idx = {k: 0 for k in by_subj}
        order = sorted(by_subj)
        taken = 0
        while taken < MEDMCQA_CAP:
            progressed = False
            for k in order:
                if idx[k] < len(by_subj[k]) and taken < MEDMCQA_CAP:
                    r = by_subj[k][idx[k]]
                    idx[k] += 1
                    taken += 1
                    progressed = True
                    q = r["question"].strip()
                    opts = [str(r[f"op{c}"]) for c in "abcd"]
                    try:
                        cop = "abcd"[int(r["cop"])]
                    except (ValueError, TypeError, IndexError):
                        cop = str(r["cop"]).strip().lower()
                    exp = r["exp"].strip()
                    prompt = (f"{q}\nA. {opts[0]}\nB. {opts[1]}\n"
                              f"C. {opts[2]}\nD. {opts[3]}")
                    response = (f"The correct answer is "
                                f"{cop.upper()}) {opts['abcd'.index(cop)]}.\n\n{exp}")
                    train.append({"src": "medmcqa", "src_id": str(r["id"]),
                                  "kind": "mcq-conv", "prompt": prompt,
                                  "response": response, "converted": True,
                                  "source_fields": [q] + opts + [exp],
                                  "meta": {"subject": r.get("subject_name"),
                                           "topic": r.get("topic_name")}})
            if not progressed:
                break
        counts["medmcqa_subjects"] = len(by_subj)
        counts["medmcqa_pool"] = len(pool)
    return train, quar


def adapt_pubmedqa_eval(counts: Counter) -> list[dict]:
    """pqa_labeled is the official heldout: eval-only hashes + refs."""
    from datasets import load_dataset
    src = SOURCES["pubmedqa"]
    ds = load_dataset(src["repo"], "pqa_labeled", split="train",
                      revision=src["revision"], trust_remote_code=False)
    out = []
    for r in ds:
        counts["pubmedqa_labeled_rows"] += 1
        q = (r.get("question") or "").strip()
        if q:
            out.append({"src": "pubmedqa", "src_id": str(r.get("pubid", "")),
                        "sha": sha12(norm(q)), "text": norm(q),
                        "ref": {"long_answer": r.get("long_answer", ""),
                                "final_decision": r.get("final_decision", "")}})
            counts["pubmedqa_quarantine"] += 1
    return out


def adapt_oasst(counts: Counter) -> list[dict]:
    from datasets import load_dataset
    src = SOURCES["oasst1"]
    ds = load_dataset(src["repo"], split="train", revision=src["revision"])
    by_id = {}
    for r in ds:
        counts["oasst_rows"] += 1
        by_id[r["message_id"]] = r
    pairs = []
    for r in ds:
        if r.get("role") != "assistant":
            continue
        if r.get("lang") != "en" or r.get("deleted") or r.get("synthetic"):
            counts["oasst_filter"] += 1
            continue
        if r.get("review_result") is not True or (r.get("rank") or 9) != 0:
            counts["oasst_unvetted"] += 1
            continue
        parent = by_id.get(r.get("parent_id") or "")
        if parent is None or parent.get("role") != "prompter":
            counts["oasst_orphan"] += 1
            continue
        pt, at = (parent.get("text") or "").strip(), (r.get("text") or "").strip()
        if len(pt) < 10 or len(at) < 20:
            counts["oasst_short"] += 1
            continue
        pairs.append({"src": "oasst1", "src_id": r["message_id"],
                      "kind": "replay", "prompt": pt, "response": at,
                      "converted": False, "meta": {}})
    rng = random.Random(7)
    rng.shuffle(pairs)
    counts["oasst_pairs"] = len(pairs)
    return pairs[:OASST_CAP]


def adapt_mmlu_quarantine(counts: Counter) -> list[dict]:
    from datasets import load_dataset
    src = SOURCES["mmlu"]
    ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=False)
    try:
        rev = ds._fingerprint  # informational only
    except Exception:
        rev = None
    src["revision_resolved"] = rev
    out = []
    for r in ds:
        counts["mmlu_rows"] += 1
        q = (r.get("question") or "").strip()
        if q:
            out.append({"src": "mmlu", "src_id": f"{r.get('subject', '?')}",
                        "sha": sha12(norm(q)), "text": norm(q)})
            counts["mmlu_quarantine"] += 1
    return out


# ---------------- quality pipeline ----------------

PHONE_RE = re.compile(r"(\+\d[\d\s().-]{7,}\d|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ADDR_RE = re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+(\s+[A-Z][a-z]+){0,3}\s+(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Close|Estate)\b")
DOSE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|mL|units?|tablets?|drops?|puffs?)\b", re.I)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def shingles(text: str, k: int = 5) -> set[str]:
    toks = norm(text).split()
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def minhash(sig_set: set[str], perms: int = 64) -> list[int]:
    masks = []
    for seed in range(perms):
        h = min(int(hashlib.sha256(f"{seed}:{s}".encode()).hexdigest(), 16)
                & 0xFFFFFFFF for s in sig_set) if sig_set else 0
        masks.append(h)
    return masks


def jacc_est(a: list[int], b: list[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / max(1, len(a))


class LSH:
    """Band-row LSH over MinHash signatures (avoids O(n^2) all-pairs)."""

    def __init__(self, bands: int = 16, rows: int = 4) -> None:
        assert bands * rows == 64
        self.bands = bands
        self.rows = rows
        self.table: dict[tuple, list[int]] = defaultdict(list)
        self.sigs: list[list[int]] = []

    def _keys(self, sig: list[int]):
        for b in range(self.bands):
            yield (b, tuple(sig[b * self.rows:(b + 1) * self.rows]))

    def add(self, sig: list[int]) -> int:
        idx = len(self.sigs)
        self.sigs.append(sig)
        for k in self._keys(sig):
            self.table[k].append(idx)
        return idx

    def query(self, sig: list[int], thresh: float = 0.9) -> list[int]:
        cands: set[int] = set()
        for k in self._keys(sig):
            cands.update(self.table.get(k, ()))
        return [i for i in cands if jacc_est(sig, self.sigs[i]) >= thresh]


def quality_filter(records: list[dict], counts: Counter,
                   review: list[dict]) -> list[dict]:
    seen_exact: set[str] = set()
    kept: list[dict] = []
    lsh: LSH = LSH()
    for rec in records:
        key = f"{rec['src']}:{rec['src_id']}"
        p, r = (rec["prompt"] or "").strip(), (rec["response"] or "").strip()
        if len(p) < 10 or len(r) < 5:
            counts["q_malformed"] += 1
            continue
        if len(r) > 6000 or len(p) > 4000:
            counts["q_too_long"] += 1
            continue
        # repetition check on response
        grams = [r[i:i + 40] for i in range(0, max(0, len(r) - 40), 10)]
        if grams and max(Counter(grams).values()) > max(3, len(grams) // 3):
            counts["q_repetitive"] += 1
            continue
        if PHONE_RE.search(p + r) or EMAIL_RE.search(p + r) or ADDR_RE.search(p + r):
            counts["q_pii"] += 1
            continue
        if rec.get("kind") == "mcq-conv" and len(r) < 30:
            counts["q_answer_leak"] += 1  # letter-only conversion slipped through
            continue
        # converted numerics must all exist in source fields (no invented doses)
        if rec.get("converted"):
            src_nums = set()
            for f in rec.get("source_fields", []):
                src_nums.update(NUM_RE.findall(str(f)))
            resp_nums = set(NUM_RE.findall(r))
            if not resp_nums.issubset(src_nums):
                counts["q_numeric_review"] += 1
                review.append({"reason": "converted-numerics-not-in-source",
                               "record": {k: rec[k] for k in
                                          ("src", "src_id", "prompt", "response")}})
                continue
        ex = sha12(norm(p) + "\n" + norm(r))
        if ex in seen_exact:
            counts["q_exact_dup"] += 1
            continue
        mh = minhash(shingles(p + " " + r))
        if lsh.query(mh):
            counts["q_near_dup"] += 1
            continue
        seen_exact.add(ex)
        lsh.add(mh)
        rec["prompt"], rec["response"] = p, r
        # dose-containing converted records stay trainable (licensed gold)
        # but join the clinician audit sample, never silently.
        if rec.get("converted") and DOSE_RE.search(r):
            review.append({"reason": "audit-dose-mention",
                           "record": {"src": rec["src"], "src_id": rec["src_id"],
                                      "dose_spans": DOSE_RE.findall(r)[:5]}})
        kept.append(rec)
    return kept


def contamination_screen(records: list[dict], quar: list[dict],
                         counts: Counter) -> tuple[list[dict], dict]:
    q_exact = {q["sha"] for q in quar}
    lsh: LSH = LSH()
    q_meta: list[tuple[str, str]] = []
    for q in quar:
        lsh.add(minhash(shingles(q["text"])))
        q_meta.append((q["src"], q["src_id"]))
    kept, hits = [], []
    for rec in records:
        if sha12(norm(rec["prompt"])) in q_exact:
            counts["contam_exact"] += 1
            hits.append({"src": rec["src"], "src_id": rec["src_id"],
                         "kind": "exact"})
            continue
        mh = minhash(shingles(rec["prompt"]))
        near = lsh.query(mh)
        if near:
            counts["contam_near"] += 1
            hits.append({"src": rec["src"], "src_id": rec["src_id"],
                         "kind": "near", "against": q_meta[near[0]]})
            continue
        kept.append(rec)
    report = {"n_quarantine": len(quar), "n_screened": len(records),
              "n_kept": len(kept), "exact_hits": counts["contam_exact"],
              "near_hits": counts["contam_near"], "hits": hits[:50]}
    return kept, report


def stratify(records: list[dict]) -> dict:
    by_src: Counter = Counter()
    by_subj: Counter = Counter()
    risk = Counter()
    for rec in records:
        by_src[rec["src"]] += 1
        subj = (rec.get("meta") or {}).get("specialty") or \
               (rec.get("meta") or {}).get("subject") or "general"
        by_subj[f"{rec['src']}/{subj}"] += 1
        low = (rec["prompt"] + " " + rec["response"]).lower()
        for name, pats in (
                ("pregnancy", ("pregnan", "postpartum", "breastfeed")),
                ("pediatric", ("infant", "neonat", "pediatric", "child", "baby")),
                ("emergency", ("emergency", "chest pain", "stroke", "sepsis",
                               "anaphyla", "bleeding", "overdose", "poison"))):
            if any(p in low for p in pats):
                risk[name] += 1
    return {"by_source": dict(by_src), "by_subject_top": dict(by_subj.most_common(15)),
            "risk_buckets": dict(risk), "n": len(records)}


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    MANI.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    review: list[dict] = []
    print("== adapters ==", flush=True)
    af_tr, af_q = adapt_afrimed(counts)
    print(f"  afrimed train={len(af_tr)} quar={len(af_q)}", flush=True)
    mq_tr, mq_q = adapt_medqa(counts)
    print(f"  medqa train={len(mq_tr)} quar={len(mq_q)}", flush=True)
    mc_tr, mc_q = adapt_medmcqa(counts)
    print(f"  medmcqa train={len(mc_tr)} quar={len(mc_q)}", flush=True)
    pm_q = adapt_pubmedqa_eval(counts)
    print(f"  pubmedqa quar={len(pm_q)} (eval-only)", flush=True)
    oa_tr = adapt_oasst(counts)
    print(f"  oasst train={len(oa_tr)}", flush=True)
    mm_q = adapt_mmlu_quarantine(counts)
    print(f"  mmlu quar={len(mm_q)}", flush=True)
    quar = af_q + mq_q + mc_q + pm_q + mm_q
    print("== quality ==", flush=True)
    per_src: dict[str, list[dict]] = {}
    for name, recs in (("afrimed", af_tr), ("medqa", mq_tr),
                       ("medmcqa", mc_tr), ("oasst1", oa_tr)):
        kept = quality_filter(recs, counts, review)
        per_src[name] = kept
        (PROC / f"{name}_train.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
        print(f"  {name}: {len(recs)} -> {len(kept)}", flush=True)
    all_train = [r for v in per_src.values() for r in v]
    print("== contamination ==", flush=True)
    clean, report = contamination_screen(all_train, quar, counts)
    (DATA / "contamination_report.json").write_text(json.dumps(report, indent=1))
    print(f"  {report['n_screened']} -> {report['n_kept']} "
          f"(exact={report['exact_hits']} near={report['near_hits']})", flush=True)
    (PROC / "quarantine_eval_prompts.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in quar))
    (DATA / "requires_clinician_review.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review))
    # mixtures
    rng = random.Random(7)
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in clean:
        by_src[r["src"]].append(r)
    for v in by_src.values():
        rng.shuffle(v)
    full = []
    weights = {"afrimed": 2.0, "medqa": 1.0, "medmcqa": 1.0, "oasst1": 0.5}
    for src, w in weights.items():
        reps = by_src.get({"afrimed": "afrimed", "medqa": "medqa",
                           "medmcqa": "medmcqa", "oasst1": "oasst1"}[src], [])
        full.extend(reps * int(w) + rng.sample(reps, int(len(reps) * (w % 1))) if reps else [])
    rng.shuffle(full)
    (MANI / "full_mixture.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in full))
    pilot = []
    for src, n in (("afrimed", 900), ("medmcqa", 300), ("medqa", 200), ("oasst1", 100)):
        pilot.extend(by_src.get(src, [])[:n])
    rng.shuffle(pilot)
    pilot = pilot[:PILOT_N]
    (MANI / "pilot_mixture.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pilot))
    (MANI / "eval_bank.json").write_text(json.dumps(
        {"quarantine_file": "data/processed/quarantine_eval_prompts.jsonl",
         "n_quarantine": len(quar),
         "by_src": dict(Counter(q["src"] for q in quar))}, indent=1))
    (MANI / "build_meta.json").write_text(json.dumps(
        {"utc": datetime.now(timezone.utc).isoformat(), "seed": 7,
         "mixture_weights": weights,
         "n_full": len(full), "n_pilot": len(pilot)}, indent=1))
    ledger = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "sources": SOURCES, "counts": dict(counts),
              "stratification": stratify(clean)}
    (DATA / "LICENSE_LEDGER.json").write_text(json.dumps(ledger, indent=1))
    print(f"== done: full={len(full)} pilot={len(pilot)} "
          f"review={len(review)} ==", flush=True)


if __name__ == "__main__":
    main()

