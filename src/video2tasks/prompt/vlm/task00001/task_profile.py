"""Task profile for VQA task00001."""

from ..question_bank_shared import QAProfile

TASK_CONTEXT = (
    "High-level task: Put the grapes into the black basket.\nObject inventory (CanonicalRef only):\n- black basket\n- brown basket\n- grape\n"
)

TASK_PROFILE = QAProfile(
    task_goal="Put the grapes into the black basket.",
    objects=["black basket", "brown basket", "grape"],
    containers=["black basket", "brown basket"],
    movable_objects=["grape"],
    target_mapping={"grape": "black basket"},
)

MIN_QUESTIONS_PER_TYPE = 50
