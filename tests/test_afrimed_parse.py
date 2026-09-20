"""Regression: AfriMed v2 mcq options/gold parsing.

v4 scored 0/431: the kernel assumed a 4-element option LIST, but the CSV
carries a JSON DICT option1..option5, and correct_answer is 'optionN'
(sometimes multi-answer 'option1,option3', which the single-answer argmax
task must exclude, not mis-score).
"""
import sys
from pathlib import Path

BDIR = Path(__file__).resolve().parents[1] / "kaggle" / \
    "native-sparse-edge0base-v1"
sys.path.insert(0, str(BDIR))

import edge0base_v1 as kbase


def test_afrimed_single_answer():
    row = {"answer_options": '{"option1": "Aspirin", "option2": "Ibuprofen", '
                             '"option3": "Paracetamol", "option4": "Heparin", '
                             '"option5": "Warfarin"}',
           "correct_answer": "option3"}
    opts, gold = kbase.afrimed_options_gold(row)
    assert opts == ["Aspirin", "Ibuprofen", "Paracetamol", "Heparin", "Warfarin"]
    assert gold == "C"


def test_afrimed_multi_answer_rejected():
    import re
    assert not re.fullmatch(r"option\d+", "option1,option3")
    assert re.fullmatch(r"option\d+", "option5")
