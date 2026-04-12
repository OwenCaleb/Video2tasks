"""Task profile for VQA task00003."""

from ..question_bank_shared import QAProfile

TASK_CONTEXT = (
    "High-level task: Sort fruits into the black basket and brown basket.\nObject inventory (CanonicalRef only):\n- black basket\n- brown basket\n- grape\n- avocado\n- kiwi\n- apple\n- orange\n"
)

TASK_PROFILE = QAProfile(
    task_goal="Sort fruits into the black basket and brown basket.",
    objects=["black basket", "brown basket", "grape", "avocado", "kiwi", "apple", "orange"],
    containers=["black basket", "brown basket"],
    movable_objects=["grape", "avocado", "kiwi", "apple", "orange"],
    target_mapping={"grape": "black basket", "avocado": "brown basket", "kiwi": "brown basket", "apple": "brown basket", "orange": "brown basket"},
)

MIN_QUESTIONS_PER_TYPE = 50
