"""Task profile for VQA task00002."""

from ..question_bank_shared import QAProfile

TASK_CONTEXT = (
    "High-level task: Put the grapes into the black basket.\n"
    "Object inventory (CanonicalRef only):\n"
    "- black basket\n"
    "- brown basket\n"
    "- grapes"
)

TASK_PROFILE = QAProfile(
    task_goal="Put the grapes into the black basket.",
    objects=["grapes", "black basket", "brown basket"],
    containers=["black basket", "brown basket"],
    movable_objects=["grapes"],
    target_mapping={"grapes": "black basket"},
)

MIN_QUESTIONS_PER_TYPE = 50
