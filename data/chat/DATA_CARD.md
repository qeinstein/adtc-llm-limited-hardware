# Jamii Afya SFT Data Card

Generated: 2026-09-19T23:09:09.620872+00:00 (pipeline run, deterministic).
Builder: training/build_dataset.py. Ledger: data/LICENSE_LEDGER.json.

## Mixtures

- Full: 24931 examples afrimed=1730, medmcqa=11967, medqa=9992, oasst1=1242
- Pilot: 1465 examples afrimed=865, medmcqa=300, medqa=200, oasst1=100
- Build meta: full_sha=56c8a00358c41227... pilot_sha=f43522b0d31ca820...

## Sources (all train-split or fully held-in; test splits quarantined)

### afrimed
- HF repo: `afrimedqa/afrimedqa_v2` @ `3b4382fa0bb51bfc026f5813021ab0ec7be9de8f`
- License: CC-BY-4.0
- Train use: split==train AND quality==True AND rated filters AND response present
- Redistributable: yes (CC-BY-4.0, attribution required)

### medqa
- HF repo: `GBaker/MedQA-USMLE-4-options` @ `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18`
- License: CC-BY-4.0 (dataset card; upstream MedQA Jin et al.)
- Train use: train split; short-answer pairs (no source rationale -> no invented explanation)
- Redistributable: yes (CC-BY-4.0, attribution required)

### medmcqa
- HF repo: `openlifescienceai/medmcqa` @ `91c6572c454088bf71b679ad90aa8dffcd0d5868`
- License: Apache-2.0
- Train use: train split, single-choice, stratified sample (cap 12000)
- Redistributable: yes (Apache-2.0)

### pubmedqa
- HF repo: `qiaojin/PubMedQA` @ `9001f2853fb87cab8d220904e0de81ac6973b318`
- License: MIT
- Train use: NONE (pqa_labeled is the official heldout -> eval-only; pqa_artificial excluded as unverifiable)
- Redistributable: eval artifacts only

### oasst1
- HF repo: `OpenAssistant/oasst1` @ `fdf72ae0827c1cda404aff25b6603abec9e3399b`
- License: Apache-2.0
- Train use: vetted EN assistant turns (reviewed, rank 0, not deleted/synthetic), cap 2500
- Redistributable: yes (Apache-2.0)

### mmlu
- HF repo: `cais/mmlu` @ `c30699e8356da336a370243923dbaf21066bb9fe`
- License: MIT
- Train use: NONE - quarantine/contamination screen only
- Redistributable: no (hashes bank only)

## Quarantine (NEVER trained on)

- medqa_test_rows: 1273
- medmcqa_quarantine: 10333
- medmcqa_test_rows: 6150
- pubmedqa_quarantine: 1000
- afrimed_quarantine: 5903

## Contamination screen vs eval banks

- screened=25310 kept=25309 exact_hits=1 near_hits=0
- method: strict-norm exact + LSH near-dup

## Review queue

- 402 examples flagged for clinician review (data/requires_clinician_review.jsonl).

## Stratification (kept set)

- n=25309, by_source={'afrimed': 865, 'medqa': 9992, 'medmcqa': 11967, 'oasst1': 2485}
- risk_buckets={'pediatric': 2905, 'pregnancy': 1156, 'emergency': 3960}
- top subjects={'medqa/general': 9992, 'oasst1/general': 2485, 'afrimed/general': 863, 'medmcqa/Anatomy': 572, 'medmcqa/Dental': 572, 'medmcqa/ENT': 572, 'medmcqa/Gynaecology & Obstetrics': 571, 'medmcqa/Orthopaedics': 571, 'medmcqa/Biochemistry': 570, 'medmcqa/Medicine': 570, 'medmcqa/Microbiology': 570, 'medmcqa/Ophthalmology': 570, 'medmcqa/Pathology': 570, 'medmcqa/Pediatrics': 570, 'medmcqa/Pharmacology': 570}
