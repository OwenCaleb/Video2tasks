"""VLM prompt templates."""

from .task00001 import (
    VQAFixedSlotTemplate,
    VQAPromptRegistry,
    get_default_prompts,
    get_default_question_types,
)

__all__ = [
    "VQAFixedSlotTemplate",
    "VQAPromptRegistry",
    "get_default_prompts",
    "get_default_question_types",
]
