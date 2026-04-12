"""Task profile for VQA task00009."""

from ..question_bank_shared import QAProfile

TASK_CONTEXT = (
    "High-level task: Sort and recover fruits from stand into the brown basket.\nObject inventory (CanonicalRef only):\n- brown basket\n- grape\n- avocado\n- kiwi\n- apple\n- orange\n- pumpkin\n"
)

TASK_PROFILE = QAProfile(
    task_goal="Sort and recover fruits from stand into the brown basket.",
    objects=["brown basket", "grape", "avocado", "kiwi", "apple", "orange", "pumpkin"],
    containers=["brown basket"],
    movable_objects=["grape", "avocado", "kiwi", "apple", "orange", "pumpkin"],
    target_mapping={"grape": "brown basket", "avocado": "brown basket", "kiwi": "brown basket", "apple": "brown basket", "orange": "brown basket", "pumpkin": "brown basket"},
)

MIN_QUESTIONS_PER_TYPE = 50
