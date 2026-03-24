"""Prompt builder for segment task00001."""

import time
from typing import List

from ...common import DEFAULT_HIGH_LEVEL_TASK, DEFAULT_OBJECT_INVENTORY_BLOCK
from .blocks import (
    ROLE_AND_GOAL_BLOCK,
    CORE_LOGIC_BLOCK,
    CANONICAL_POLICY_BLOCK,
    OUTPUT_FORMAT_BLOCK,
)
from .examples import SEGMENT_EXAMPLES


def _pick_examples(now_ns: int, k: int = 1) -> List[dict]:
    if not SEGMENT_EXAMPLES:
        return []
    size = min(k, len(SEGMENT_EXAMPLES))
    start = now_ns % len(SEGMENT_EXAMPLES)
    picked = []
    for i in range(size):
        picked.append(SEGMENT_EXAMPLES[(start + i) % len(SEGMENT_EXAMPLES)])
    return picked


def prompt_switch_detection(n_images: int) -> str:
    """Build segment switch-detection prompt with time-rotated few-shot examples."""
    lines = [
        (
            f"You are a robotic vision analyzer watching a {n_images}-frame video clip "
            "of household manipulation tasks."
        ),
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.",
        "",
        ROLE_AND_GOAL_BLOCK,
        CORE_LOGIC_BLOCK,
        "### Task Context (Optional Prior Knowledge)",
        "High-level task:",
        DEFAULT_HIGH_LEVEL_TASK,
        "",
        DEFAULT_OBJECT_INVENTORY_BLOCK,
        CANONICAL_POLICY_BLOCK,
        OUTPUT_FORMAT_BLOCK,
        "",
        "### Representative Example",
    ]

    for ex in _pick_examples(time.time_ns(), k=1):
        lines.append(f"**{ex['title']}**")
        lines.append(ex["json"])

    return "\n".join(lines)
