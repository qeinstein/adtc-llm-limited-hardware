# Jamii Afya Clinical Guidance Layer

Small, versioned, high-value clinical safety knowledge. It complements the
model — it does not replace it and it does not diagnose.

## Pipeline position

```
User
 ↓
risk/fact extraction        (runtime/safety/risk.py + facts.py)
 ↓
deterministic high-risk rules  (runtime/safety/rules.py over guidance/)
 ↓
optional structured-guidance retrieval (top ≤3 cards, small payload)
 ↓
Qwen3.6 Q2_K+K4/16 (untuned final artifact; see TRAINING.md)
 ↓
output safety lint          (runtime/safety/output_lint.py + repetition.py)
 ↓
UI (urgency badge; override banner never hidden in reasoning)
```

## Domains (18 files, 35 cards, guidance/manifest.json v1.0.0)

| file | domain | cards |
|---|---|---|
| pediatric_imci.yaml | child danger signs / triage | 3 |
| dehydration_ors.yaml | dehydration + ORS | 2 |
| pregnancy.yaml | antenatal danger signs | 3 |
| labour_postpartum.yaml | labour/postpartum emergencies | 2 |
| neonatal.yaml | young-infant danger signs | 2 |
| trauma.yaml | bleeding / open fractures | 2 |
| poisoning.yaml | poisoning / caustic / bleach | 2 |
| respiratory.yaml | distress / asthma | 3 |
| anaphylaxis.yaml | severe allergy | 1 |
| fever_sepsis.yaml | fever / serious infection | 2 |
| malaria.yaml | malaria triage | 2 |
| neuro.yaml | stroke / seizures | 3 |
| cardiac.yaml | chest pain | 1 |
| diabetes.yaml | glucose emergencies | 1 |
| medication.yaml | dosing variables / high-risk meds | 2 |
| mental_health.yaml | self-harm | 1 |
| reproductive.yaml | STI info / assault support | 2 |
| public_health.yaml | outbreak escalation + NCDC | 1 |

Optional burns/snakebite domains were NOT added: no cleanly sourced,
testable card set within this closeout. Do not add domains without a
verified source and regression cases.

## Rule philosophy

- Safety floors, not diagnoses. Cards enforce danger recognition,
  escalation, missing-information requirements, and unsafe-advice
  prohibitions. The model may discuss possible causes with uncertainty.
- Example GOOD: child + severe breathing difficulty + inability to drink
  → emergency referral. Example BAD (never done): cough + fever → "pneumonia".
- Every card carries: id, domain, version, priority, scope, trigger_facts
  (+ optional trigger_any_of), required_missing_information, risk_level,
  required/prohibited/safe-interim actions, uncertainty notes, full source
  provenance, and review status.
- Over-triage beats under-triage at this layer. Known cost: informational
  queries that name symptoms (e.g. "what causes chest pain?") can match
  urgent cards. The model + lint carry precision; suites measure the floor.

## Retrieval behavior

Cards are injected into the system prompt only when: rules risk is urgent/
emergency, a dosing/medication fact fires, a pregnancy/pediatric fact fires,
or the user asks what WHO/NCDC/IMCI recommend. Otherwise questions stay
fast (no retrieval). Payload is ≤3 cards with exact attribution strings;
the authority lint rejects WHO/IMCI/NCDC claims not backed by them.

## Sources and licensing

Provenance lives in guidance/manifest.json (checked 2026-09-20):

- WHO IMCI Chart Booklet (Mar 2014, ISBN 9789241506823)
- WHO diarrhoea clinical-management manual (IRIS 10665/43456)
- WHO antenatal-care recommendations (2016, ISBN 9789241549912)
- WHO/ICRC Basic Emergency Care (2018, ISBN 9789241513081)
- WHO malaria living guideline (version-checked 2026-09-20)
- Nigeria NCDC Connect Centre (toll-free 6232, 24/7)
- Jamii Afya product safety policy v1 (this repo, for pure prohibitions)

Cards are ORIGINAL summaries with links — no WHO text is redistributed —
and nothing here implies WHO/UNICEF/NCDC endorsement of Jamii Afya.
No card carries numeric doses, concentrations, or thresholds except
package-label/NCDC-contact facts; thresholds need clinician verification
before any card may state them.

## Review status

`machine_verified: true` (schema + regression suites), `clinician_reviewed:
false` everywhere. No card has had clinician review. See SAFETY.md.
