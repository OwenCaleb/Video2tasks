"""VQA (Visual Question Answering) module for frame-level annotation."""

from .types import VQAResult, VQAQuestion, VQAJobData
from ..prompt.vlm.task00001 import VQAPromptRegistry
from ..prompt.loader import create_vqa_prompt_registry
from .job_builder import VQAJobBuilder
from .writer import VQAWriter

__all__ = [
    "VQAResult",
    "VQAQuestion", 
    "VQAJobData",
    "VQAPromptRegistry",
    "create_vqa_prompt_registry",
    "VQAJobBuilder",
    "VQAWriter",
]
