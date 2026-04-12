"""Task profile for VQA task00011."""

from ..question_bank_shared import QAProfile

TASK_CONTEXT = (
    "High-level task: Sort fruits into the brown basket.\nObject inventory (CanonicalRef only):\n- brown basket\n- grape\n- avocado\n- kiwi\n- apple\n- orange\n"
)

TASK_PROFILE = QAProfile(
    task_goal="Sort fruits into the brown basket.",
    objects=["brown basket", "grape", "avocado", "kiwi", "apple", "orange"],
    containers=["brown basket"],
    movable_objects=["grape", "avocado", "kiwi", "apple", "orange"],
    target_mapping={"grape": "brown basket", "avocado": "brown basket", "kiwi": "brown basket", "apple": "brown basket", "orange": "brown basket"},
)

MIN_QUESTIONS_PER_TYPE = 50
