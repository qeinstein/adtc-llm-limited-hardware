"""Jamii Afya runtime safety guard (rules layer, no extra LLM calls)."""
from .facts import extract_facts
from .output_lint import SAFE_FALLBACK, corrective_instruction, lint_output
from .repetition import check_repetition
from .risk import classify_input, log_event, reminder_for
from .rules import evaluate as evaluate_rules
from .rules import load_cards, retrieval_payload

__all__ = ["classify_input", "reminder_for", "log_event", "lint_output",
           "corrective_instruction", "SAFE_FALLBACK", "extract_facts",
           "evaluate_rules", "load_cards", "retrieval_payload",
           "check_repetition"]
