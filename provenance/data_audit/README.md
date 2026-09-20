# Data audit pointers (not duplicated)

The audit artifacts are large and live at their canonical paths:

- training/data_audit.json — full §K/§L/§M audit (24,931 rows, percentiles,
  mask 100/100, contamination 0/0, seq-2048 decision)
- data/DATA_CARD.md — dataset card
- data/LICENSE_LEDGER.json — per-source licenses
- data/contamination_report.json — quarantine screen record (1 exact removal)
- data/manifests/build_meta.json — mixture SHAs + seed
- data/requires_clinician_review.jsonl — 402-row review queue (0 reviewed)
