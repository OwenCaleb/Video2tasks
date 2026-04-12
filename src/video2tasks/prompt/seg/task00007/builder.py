"""Prompt builder for segment task00007."""

import time
from typing import List

from .blocks import (
    ROLE_AND_GOAL_BLOCK,
    CORE_LOGIC_BLOCK,
    CANONICAL_POLICY_BLOCK,
    OUTPUT_FORMAT_BLOCK,
)
from .examples import SEGMENT_EXAMPLES

TASK00007_HIGH_LEVEL = "Sort fruits from stand into the brown basket."
TASK00007_OBJECT_INVENTORY = (
    '''Object inventory:
Descriptor : CanonicalRef
A brown basket : brown basket
A grape : grape
A avocado : avocado
A kiwi : kiwi
A apple : apple
A orange : orange
A pumpkin : pumpkin'''
)


def _pick_examples(now_ns: int, k: int = 1) -> List[dict]:
    if not SEGMENT_EXAMPLES:
        return []
    size = min(k, len(SEGMENT_EXAMPLES))
    start = now_ns % len(SEGMENT_EXAMPLES)
    return [SEGMENT_EXAMPLES[(start + i) % len(SEGMENT_EXAMPLES)] for i in range(size)]


def prompt_switch_detection(n_images: int) -> str:
    """Build segment prompt for task00007 with compact rotating examples."""
    lines = [
        (
            f"You are a robotic vision analyzer watching a {n_images}-frame video clip "
            "of household manipulation tasks."
        ),
        f"Image indices range from 0 to {n_images - 1}.",
        "",
        ROLE_AND_GOAL_BLOCK,
        CORE_LOGIC_BLOCK,
        "### Task Context",
        f"High-level task: {TASK00007_HIGH_LEVEL}",
        "",
        TASK00007_OBJECT_INVENTORY,
        CANONICAL_POLICY_BLOCK,
        OUTPUT_FORMAT_BLOCK,
        "",
        "### Representative Example",
    ]

    for ex in _pick_examples(time.time_ns(), k=1):
        lines.append(f"**{ex['title']}**")
        lines.append(ex["json"])

    return "\n".join(lines)
