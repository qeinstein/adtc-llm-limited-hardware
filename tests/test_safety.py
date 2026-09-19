"""Safety guard tests: rule loading, input risk levels (EN + SW), output
lint failures/passes, regen instruction, and no-PHI audit logging."""
from runtime.safety import (SAFE_FALLBACK, classify_input, corrective_instruction,
                            lint_output, log_event, reminder_for)


def test_rules_load():
    r = classify_input("mild headache after a long day")
    assert r["level"] in ("LOW", "ROUTINE")
    assert r["hits"] == []


def test_emergency_en():
    r = classify_input("crushing chest pain down my left arm, sweating")
    assert r["level"] == "EMERGENCY"
    assert "chest-pain" in r["hits"]
    assert "emergency" in reminder_for(r).lower()


def test_emergency_sw():
    r = classify_input("Mtoto ana homa kali na degedege, hazioni")
    assert r["level"] == "EMERGENCY"
    assert "seizure" in r["hits"]


def test_pediatric_redflag():
    r = classify_input("My 3-month-old baby has fever of 39 and refuses to feed")
    assert r["level"] == "EMERGENCY"
    assert "pediatric-redflag" in r["hits"]


def test_selfharm():
    r = classify_input("my friend says he wants to kill himself tonight")
    assert r["level"] == "EMERGENCY"
    assert "self-harm" in r["hits"]


def test_meds_high_risk():
    r = classify_input("Can I take ibuprofen with my blood thinner warfarin?")
    assert r["level"] == "MEDICATION_HIGH_RISK"
    assert "interaction" in r["hits"]
    assert "dose" in reminder_for(r).lower()


def test_meds_beats_routine_not_emergency():
    r = classify_input("My child weighs 12 kg, how much paracetamol syrup?")
    assert r["level"] == "MEDICATION_HIGH_RISK"


def test_lint_emergency_no_escalation():
    risk = {"level": "EMERGENCY", "hits": ["chest-pain"]}
    out = lint_output(risk, "chest pain", "Chest pain can have many causes. Rest and drink water.")
    assert out["ok"] is False
    assert "emergency-no-escalation" in out["failures"]
    good = lint_output(risk, "chest pain",
                       "Call emergency services immediately and go to the hospital. Do not drive yourself.")
    assert good["ok"] is True


def test_lint_wait_at_home():
    risk = {"level": "EMERGENCY", "hits": ["stroke"]}
    out = lint_output(risk, "face droop", "Just monitor at home and see how it goes.")
    assert "wait-at-home-despite-acuity" in out["failures"]


def test_lint_confident_diagnosis():
    risk = {"level": "ROUTINE", "hits": []}
    out = lint_output(risk, "cough", "You have bacterial pneumonia. Take amoxicillin.")
    assert "confident-diagnosis" in out["failures"]


def test_lint_peds_dose_without_weight():
    risk = {"level": "MEDICATION_HIGH_RISK", "hits": ["peds-dosing"]}
    out = lint_output(risk, "my baby has fever", "Give 5 ml of paracetamol syrup now.")
    assert "peds-dose-without-weight" in out["failures"]
    ok = lint_output({"level": "MEDICATION_HIGH_RISK", "hits": ["peds-dosing"]},
                     "my 12 kg child has fever",
                     "Ask your pharmacist; paracetamol dosing depends on weight (12 kg here).")
    assert ok["ok"] is True


def test_corrective_and_fallback():
    assert "emergency-no-escalation" in corrective_instruction(["emergency-no-escalation"])
    assert "in-person care" in SAFE_FALLBACK


def test_audit_log_has_no_phi():
    ev = log_event({"level": "EMERGENCY", "hits": ["chest-pain"]},
                   {"lint": ["emergency-no-escalation"]})
    blob = str(ev)
    assert "chest-pain" in blob and ev["level"] == "EMERGENCY"
    assert "crushing" not in blob  # prompt/response text never logged
