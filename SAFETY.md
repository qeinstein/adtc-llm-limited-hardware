# Jamii Afya safety (SAFETY.md)

## Intended use

Offline bilingual (EN/SW) health-information and triage support for
community health workers and consumers in African settings: danger-sign
recognition, self-care information, referral preparation, clinician
questions. Information only — not examination, testing, or treatment.

## Non-intended use

Definitive diagnosis, individualized dosing without a clinician,
emergency care substitution, pediatric/pregnancy dosing from chat alone,
mental-health crisis counseling without human support, veterinary or
non-human use, any use as a regulated medical device.

## Layers (defense in depth; no single layer is sufficient)

1. Versioned system prompt (`prompts/system.json` v1.0.0): triage-first,
   calibration, medication caution, high-risk groups, self-contained finals.
2. Runtime guard (`runtime/safety/`): rule-based input risk (EMERGENCY /
   MEDICATION_HIGH_RISK / URGENT / ROUTINE / LOW, EN+SW patterns), output
   lint (no-escalation, wait-at-home, confident diagnosis, peds dose
   without weight, self-harm support), one corrective regen then safe
   fallback. No extra LLM call. Audit logs carry rule ids only, never PHI.
3. RAG grounding: clinical-management answers require retrieved corpus
   support; ungrounded questions get the explicit ungrounded response.
4. Training (pending): safety-weighted SFT + gated DPO; refusal/escalation
   behavior measured, not assumed.

## Triage rules (product behavior)

LOW → useful answer/self-care. ROUTINE → answer + clinician follow-up with
reason. URGENT → same-day/soon assessment, why + how urgently. EMERGENCY →
emergency action FIRST, then explanation; never "monitor at home".
Professional-care recommendations always explain WHY and HOW URGENTLY.

## Medication safeguards

Never fabricate doses. Dose-dependent questions must surface the missing
patient factors (age/weight/indication/kidney/pregnancy/meds). Extra
caution: children, pregnancy/breastfeeding, elderly, kidney/liver disease,
allergies, polypharmacy, narrow-margin drugs. No casual stop/change advice
for prescribed medicines. Antimicrobial stewardship (no antibiotics for
viral colds).

## Red-team / evaluation status

- Baseline safety generation bank (35 prompts: 15 emergency, 8 meds,
  uncertainty, routine, infection, Kiswahili, mental-health) runs on the
  frozen untuned system; verbatim outputs stored paired for tuned
  comparison. Rules-scan recall is a weak signal, reported as such.
- MedSafetyBench, red-flag suite, and clinician-review queue
  (`data/requires_clinician_review.jsonl`) are tracked; unresolved
  high-risk items BLOCK gold training status.
- No LLM cross-check is presented as clinician review.

## Known failure modes / limitations

- Keyword guard has false negatives (paraphrase, code-mixing) and false
  positives; it is a backstop, not a certifier.
- Thinking traces consume token budget; long reasoning can delay finals.
- Model may know stale guidelines; RAG corpus version bounds freshness.
- No clinical validation by licensed clinicians has been performed; do not
  claim any. Nigeria NAFDAC SaMD guidance is tracked for documentation
  completeness (model/dataset/version/eval records); NO approval claimed.
