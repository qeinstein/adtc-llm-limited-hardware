# Provenance package — Jamii Afya Q2K submission

The shipping model has **NO weight-level fine-tuning**. There are therefore
no adapter checkpoints, no training logs, and no merged tuned weights to
show — and this package does not invent any. (Proof-of-training artifacts
are not required when no weight-level fine-tuning occurred.)

What this package DOES prove, file by file:

| claim | evidence |
|---|---|
| base artifact identity | `base_model.json` + `checksums.txt` |
| final artifact identity | `checksums.txt` + `model/manifest.json` (repo root) |
| exact transcode recipe | `quantization_or_conversion_scripts/q2k_upload_v1.py` (pinned `llama-quantize` + per-tensor overrides; asserts byte-identity before upload) |
| exact runtime transformation | `quantization_or_conversion_scripts/join4_apply.py` (K4/16 graph patch, lazy experts, bounded executor) + `runtime_provenance.json` |
| training was prepared but NOT run | `training_preflight/` (config, gate script, env lock, data audit, NO-GO report) |
| training data curation/audit | `data_audit/` (originals live in `training/data_audit.json`, `data/DATA_CARD.md`, `data/LICENSE_LEDGER.json`, `data/contamination_report.json`) |

## Layout

```
provenance/
  README.md (this file)
  checksums.txt
  base_model.json
  runtime_provenance.json
  quantization_or_conversion_scripts/
    q2k_upload_v1.py      # canonical transcode+upload (copy; live: kaggle/...)
    join4_apply.py        # runtime patch applier (copy; live: probes/edge0_port/)
  training_preflight/     # COPIES for the record; live files under training/
    kaggle_train.yaml
    preflight.py
    environment.lock.txt
    PREFLIGHT.md          # the NO-GO verdict — NOT a completed fine-tune
  data_audit/             # POINTERS (files too large to duplicate)
    README.md
```

Copies are byte-snapshots at release time; the live files under
`training/` and `kaggle/` are authoritative for any future run.
