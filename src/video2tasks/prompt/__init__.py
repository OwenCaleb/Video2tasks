"""Unified prompt entrypoints."""

from .seg.task00001 import prompt_switch_detection
from .cot.task00001 import build_cot_prompt
from .vlm.task00001 import (
    VQAFixedSlotTemplate,
    VQAPromptRegistry,
    get_default_prompts,
    get_default_question_types,
)

__all__ = [
    "prompt_switch_detection",
    "build_cot_prompt",
    "VQAFixedSlotTemplate",
    "VQAPromptRegistry",
    "get_default_prompts",
    "get_default_question_types",
]
