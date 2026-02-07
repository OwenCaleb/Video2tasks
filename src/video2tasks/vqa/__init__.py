"""VQA (Visual Question Answering) module for frame-level annotation."""

from .types import VQAResult, VQAQuestion, VQAJobData
from .prompts import VQAPromptRegistry, get_default_prompts
from .job_builder import VQAJobBuilder
from .writer import VQAWriter

__all__ = [
    "VQAResult",
    "VQAQuestion", 
    "VQAJobData",
    "VQAPromptRegistry",
    "get_default_prompts",
    "VQAJobBuilder",
    "VQAWriter",
]
