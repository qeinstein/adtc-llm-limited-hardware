# Jamii Afya Safety

## Historical judge failures and mitigations

1. **WHO IMCI pediatric triage** (3yo, fever, RR55, indrawing, far care):
   vague/high-risk language, improvised oxygen/IV advice. →
   `imci-severe-respiratory` (emergency + override) prohibits improvised
   regimens; 10 paraphrase variants in evals/judge_regressions (J1-*).
2. **Third-trimester reduced fetal movement** reduced to bad heart-rate
   thresholds. → `preg-reduced-movement` requires same-day assessment,
   prohibits home-threshold rule-outs (J2-*).
3. **Dehydration/ORS incomplete.** → `dehydration-assess` + `ors-preparation`
   (J3-*); lint blocks invented concentrations via dose rules.
4. **Unsafe bleach "flush"** invented and falsely attributed to WHO/IMCI. →
   `caustic-bleach` prohibitions + `bleach-recipe` and
   `unsupported-authority-claim` hard lint gates (J4-*, S01, S11).
5. **Compound fracture drifted to field amputation.** → `open-fracture`
   refuses procedural amputation; `diy-amputation` lint gate (J5-*, S02).
6. **False authority** (invented WHO/IMCI protocols). → attribution lint:
   authority claims must match retrieved card attributions or the answer
   regenerates without them (J6-*).
7. **Kiswahili looping to context limit.** → repetition lint (n-gram,
   trailing-loop, length caps) + regen/fallback (J7-*, K suite).

Note: the one-regen-else-fallback backstop runs on the non-stream
endpoint; the streamed UI path runs the same lint but advisory (a
failure surfaces as a visible caution banner), since regenerating
mid-stream would double latency. All pre-generation deterministic
gates apply identically on both paths.

## Regression suites (rules tier, runs without the model)

- evals/judge_regressions (70), evals/clinical_guidance (72),
  evals/kiswahili (15), evals/safety (14) = 171 cases, all green via
  tests/test_guidance_regressions.py.
- Hard failures with zero tolerance in the frozen suite: oral bleach
  recipe, DIY amputation, invented WHO/IMCI protocol, invented dose,
  missed obvious emergency, wait-and-see for red flags, unsafe
  pediatric/pregnancy dosing, severe looping, fabricated citations
  (specific-author/journal/doi strings), confident diagnosis from
  thin data.
- Second tier (model-backed judging of correctness/concision/usefulness)
  needs a live server and is NOT yet run; the suites pin the
  deterministic floor only.

## Known limitations

- No clinician has reviewed any card, prompt, or reply in this repo.
- Numeric thresholds are deliberately absent from cards; RR cutoffs etc.
  need clinician sourcing before encoding.
- Over-triage on informational queries (symptom mentions match urgent
  cards); the model carries precision.
- Kiswahili coverage is core clinical terms only; native-speaker review
  and expansion required.
- Lint is heuristic: a bleach recipe that embeds negations ("do not
  exceed") can slip the recipe gate; regen + cards + review carry the rest.
- The deployed model is the UNTUNED Q2_K+K4/16 artifact (final; the
  prepared QLoRA run was NO-GO on available hardware, see TRAINING.md).
  Safety does not depend on tuned weights: rules, retrieval, lint, and
  fallbacks are identical across Fast/Medium/High reasoning modes, and
  emergency banners render before reasoning completes.

## Clinician-review status

- Guidance cards: 0/35 reviewed.
- Data review queue: 402 examples in data/requires_clinician_review.jsonl.
- Required before any clinical claim beyond the current floors: review of
  cards, thresholds, Kiswahili strings, and refusal/fallback wording.
