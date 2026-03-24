"""Task profile for VQA task00001."""

from ..question_bank_shared import QAProfile

TASK_CONTEXT = (
    "High-level task: Put the toy cars into the brown basket and put the fruit into the black basket.\n"
    "Object inventory (CanonicalRef only):\n"
    "- red car\n"
    "- gold car\n"
    "- black car\n"
    "- kiwi\n"
    "- avocado\n"
    "- brown basket\n"
    "- black basket"
)

TASK_PROFILE = QAProfile(
    task_goal="Put the toy cars into the brown basket and put the fruit into the black basket.",
    objects=["red car", "gold car", "black car", "kiwi", "avocado", "brown basket", "black basket"],
    containers=["brown basket", "black basket"],
    movable_objects=["red car", "gold car", "black car", "kiwi", "avocado"],
    target_mapping={
        "red car": "brown basket",
        "gold car": "brown basket",
        "black car": "brown basket",
        "kiwi": "black basket",
        "avocado": "black basket",
    },
)

MIN_QUESTIONS_PER_TYPE = 50
