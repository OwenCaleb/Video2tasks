"""Prompt registry for VQA task00001 (adapter-based)."""

from typing import List

from ..adapter import VQAFixedSlotTemplate, VQAPromptRegistry, VQATaskAdapter, build_question_types, build_registry
from .demos import VQA_DEMOS
from .task_profile import MIN_QUESTIONS_PER_TYPE, TASK_CONTEXT, TASK_PROFILE


def _task_adapter() -> VQATaskAdapter:
    return VQATaskAdapter(
        profile=TASK_PROFILE,
        task_context=TASK_CONTEXT,
        min_per_type=MIN_QUESTIONS_PER_TYPE,
        demos=VQA_DEMOS,
    )


def get_default_prompts() -> VQAPromptRegistry:
    return build_registry(_task_adapter())


def get_default_question_types() -> List[str]:
    return build_question_types(_task_adapter())
