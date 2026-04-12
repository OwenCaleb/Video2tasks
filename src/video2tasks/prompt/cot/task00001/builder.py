"""Prompt builder for CoT task00001."""

import time
from typing import List

from .blocks import ROLE_BLOCK, CANONICAL_POLICY_BLOCK, SLOT_RULE_BLOCK, OUTPUT_BLOCK
from .examples import COT_EXAMPLES

TASK00001_OBJECT_INVENTORY = (
    '''Object inventory:
Descriptor : CanonicalRef
A black basket : black basket
A brown basket : brown basket
A grape : grape'''
)


def _pick_examples(now_ns: int, k: int = 1) -> List[dict]:
    if not COT_EXAMPLES:
        return []
    size = min(k, len(COT_EXAMPLES))
    start = now_ns % len(COT_EXAMPLES)
    return [COT_EXAMPLES[(start + i) % len(COT_EXAMPLES)] for i in range(size)]


def build_cot_prompt(high_level_instruction: str, subtask: str, n_images: int) -> str:
    """Build CoT prompt for task00001 using rotating compact demos."""
    lines = [
        ROLE_BLOCK,
        "",
        "### Inputs",
        f'High-level task: "{high_level_instruction}"',
        f'Subtask performed: "{subtask}"',
        f"Visual evidence: {n_images} frame(s)",
        "",
        TASK00001_OBJECT_INVENTORY,
        CANONICAL_POLICY_BLOCK,
        "### Your job",
        "Produce a compact state snapshot to justify next action selection.",
        SLOT_RULE_BLOCK,
        OUTPUT_BLOCK,
        "",
        "### Demo",
    ]

    for ex in _pick_examples(time.time_ns(), k=1):
        lines.append(f"**{ex['title']}**")
        lines.append(ex["json"])

    return "\n".join(lines)
