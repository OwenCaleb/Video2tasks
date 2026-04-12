"""Task profile for VQA task00004."""

from ..question_bank_shared import QAProfile

TASK_CONTEXT = (
    "High-level task: Sort fruits including pumpkin into the baskets.\nObject inventory (CanonicalRef only):\n- black basket\n- brown basket\n- grape\n- avocado\n- kiwi\n- apple\n- orange\n- pumpkin\n"
)

TASK_PROFILE = QAProfile(
    task_goal="Sort fruits including pumpkin into the baskets.",
    objects=["black basket", "brown basket", "grape", "avocado", "kiwi", "apple", "orange", "pumpkin"],
    containers=["black basket", "brown basket"],
    movable_objects=["grape", "avocado", "kiwi", "apple", "orange", "pumpkin"],
    target_mapping={"grape": "black basket", "avocado": "brown basket", "kiwi": "brown basket", "apple": "brown basket", "orange": "brown basket", "pumpkin": "brown basket"},
)

MIN_QUESTIONS_PER_TYPE = 50
