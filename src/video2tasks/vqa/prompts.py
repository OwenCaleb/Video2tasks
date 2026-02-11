"""VQA prompt templates with fixed question slots and Task Context.

Each question type defines a **fixed ordered list of questions** with
constrained answer spaces. Questions may contain ``[Object]`` placeholders;
the VLM must pick a single relevant object and replace ``[Object]`` with its
CanonicalRef, producing exactly one QA per template.

The output format is the classic
``{"qas": [{"type": ..., "question": ..., "answer": ...}]}``

The number of question *templates* per type is controlled by
``questions_per_type`` in config — it takes the first N from the fixed list.
"""

from typing import Dict, List, Optional
import random
import time
from dataclasses import dataclass, field


@dataclass
class VQAFixedSlotTemplate:
    """A VQA template with a fixed ordered list of (question, answer_hint) slots.

    If a question contains ``[Object]``, the VLM should answer it once for a
    single relevant object (no per-object expansion).
    """
    type_name: str
    description: str
    slots: List[Dict[str, str]] = field(default_factory=list)
    #  Each slot: {"question": "...", "answer_space": "..."}


# Default templates ------------------------------------------------------------
_DEFAULT_PROMPTS: Dict[str, VQAFixedSlotTemplate] = {}


def _register(t: VQAFixedSlotTemplate) -> VQAFixedSlotTemplate:
    _DEFAULT_PROMPTS[t.type_name] = t
    return t


_register(VQAFixedSlotTemplate(
    type_name="spatial",
    description="Spatial relationship questions (per-object)",
    slots=[
        {"question": "What objects are to the LEFT of [Object]?",
         "answer_space": "Single string listing CanonicalRef names, or 'none'"},
        {"question": "What objects are to the RIGHT of [Object]?",
         "answer_space": "Single string listing CanonicalRef names, or 'none'"},
        {"question": "What objects are IN FRONT of [Object]?",
         "answer_space": "Single string listing CanonicalRef names, or 'none'"},
        {"question": "What objects are BEHIND [Object]?",
         "answer_space": "Single string listing CanonicalRef names, or 'none'"},
        {"question": "What objects are ABOVE [Object]?",
         "answer_space": "Single string listing CanonicalRef names, or 'none'"},
        {"question": "What objects are BELOW [Object]?",
         "answer_space": "Single string listing CanonicalRef names, or 'none'"},
        {"question": "What is the distance between the gripper and [Object]?",
         "answer_space": "One of: touching, close, medium, far"},
    ],
))


_register(VQAFixedSlotTemplate(
    type_name="attribute",
    description="Per-object attribute questions",
    slots=[
        {"question": "What is the color of [Object]?",
         "answer_space": "Color name(s)"},
        {"question": "What is the shape of [Object]?",
         "answer_space": "Brief shape description"},
        {"question": "What is the material of [Object]?",
         "answer_space": "One of: metal, plastic, wood, fabric, glass, wicker, rubber, other"},
        {"question": "Where is [Object] relative to the gripper?",
         "answer_space": "Brief spatial description (e.g. left, right, in front, behind, in hand)"},
        {"question": "Is the gripper grasping [Object]?",
         "answer_space": "yes / no"},
    ],
))


_register(VQAFixedSlotTemplate(
    type_name="existence",
    description="Per-object existence/visibility questions",
    slots=[
        {"question": "Is [Object] visible in the scene?",
         "answer_space": "yes / no"},
        {"question": "Is [Object] on the surface?",
         "answer_space": "yes / no"},
        {"question": "Is [Object] inside a container?",
         "answer_space": "yes (which container CanonicalRef) / no"},
        {"question": "Is [Object] being held by the gripper?",
         "answer_space": "yes / no"},
        {"question": "Is the workspace clear?",
         "answer_space": "yes / no, with brief reason"},
    ],
))


_register(VQAFixedSlotTemplate(
    type_name="count",
    description="Scene-level counting questions",
    slots=[
        {"question": "How many graspable objects are visible?",
         "answer_space": "integer"},
        {"question": "How many containers are visible?",
         "answer_space": "integer"},
        {"question": "How many objects are on the surface?",
         "answer_space": "integer"},
        {"question": "How many objects are inside containers?",
         "answer_space": "integer"},
    ],
))


_register(VQAFixedSlotTemplate(
    type_name="manipulation",
    description="Manipulation state questions (per-object and scene)",
    slots=[
        {"question": "Is [Object] graspable?",
         "answer_space": "yes / no, with brief reason"},
        {"question": "Where should the gripper grasp [Object]?",
         "answer_space": "Brief spatial description of grasp point"},
        {"question": "Where should [Object] be placed?",
         "answer_space": "Target container/location CanonicalRef"},
        {"question": "What action should happen next?",
         "answer_space": "One of: approach, grasp, lift, move, place, release, idle"},
        {"question": "What is the gripper state?",
         "answer_space": "One of: open, closed, holding <CanonicalRef>"},
    ],
))


# Registry ---------------------------------------------------------------------

class VQAPromptRegistry:
    """Registry for fixed-question VQA prompt templates."""

    def __init__(self) -> None:
        self._prompts: Dict[str, VQAFixedSlotTemplate] = {}

    def register(self, template: VQAFixedSlotTemplate) -> None:
        self._prompts[template.type_name] = template

    def get(self, type_name: str) -> Optional[VQAFixedSlotTemplate]:
        return self._prompts.get(type_name)

    def list_types(self) -> List[str]:
        return list(self._prompts.keys())

    def build_single_type_prompt(
        self,
        type_name: str,
        n_images: int = 1,
        max_questions: Optional[int] = None,
        task_context: Optional[str] = None,
    ) -> str:
        """Build a prompt for one question type.

        Parameters
        ----------
        type_name : str
            Question type (e.g. "spatial").
        n_images : int
            Number of images provided.
        max_questions : int, optional
            Only use the first N question templates.
        task_context : str, optional
            Free-form task context block (high-level task, object inventory,
            CanonicalRef policy).  Prepended verbatim after the header.
        """
        template = self.get(type_name)
        if template is None:
            raise ValueError(
                f"Unknown question type: {type_name}. Available: {self.list_types()}"
            )

        slots = template.slots
        if max_questions is not None and 0 < max_questions < len(slots):
            rng = random.Random(time.time_ns())
            slots = rng.sample(slots, k=max_questions)

        # --- Header ---
        header = (
            f"You are analyzing {'an image' if n_images == 1 else f'{n_images} images'} "
            f"from a robot manipulation scenario."
        )

        lines = [header, ""]

        # --- Task Context (optional) ---
        if task_context:
            lines.append("### Task Context (Prior Knowledge)")
            lines.append(task_context.strip())
            lines.append("")

        # --- Canonical Reference Policy (always present) ---
        lines.append("### Canonical Reference Policy")
        lines.append(
            "1. CanonicalRef is the ONLY allowed name to mention any entity in "
            "the output JSON.  Never invent synonyms, abbreviations, or "
            "alternative names."
        )
        lines.append(
            "2. If a question contains [Object], choose ONE relevant object "
            "and replace [Object] with its CanonicalRef in the \"question\" field."
        )
        lines.append("")

        # --- Questions ---
        has_object_placeholder = any("[Object]" in s["question"] for s in slots)
        lines.append(
            f"Answer EXACTLY the {len(slots)} question(s) below about "
            f"\"{type_name}\" in the listed order."
        )
        if has_object_placeholder:
            lines.append(
                "For any template containing [Object], choose ONE relevant object "
                "and replace [Object] with its CanonicalRef."
            )
        lines.append(
            "Generate exactly questions_per_type QAs for this type. "
            "You MUST choose only from the templates provided below; "
            "do not invent new questions or add extra QAs."
        )
        lines.append("")

        for i, slot in enumerate(slots, 1):
            lines.append(f"{i}. \"{slot['question']}\"")
            lines.append(f"   Answer space: {slot['answer_space']}")
        lines.append("")

        # --- Output schema ---
        lines.append(
            "IMPORTANT: Return ONLY a valid JSON object. "
            "No markdown, no code fences, no extra text."
        )
        lines.append("")
        lines.append("Output format:")
        lines.append("{")
        lines.append('  "qas": [')

        for i, slot in enumerate(slots):
            comma = "," if i < len(slots) - 1 else ""
            lines.append(
                f'    {{"type": "{type_name}", '
                f'"question": "{slot["question"]}", '
                f'"answer": "..."}}{comma}'
            )

        lines.append("  ]")
        lines.append("}")

        return "\n".join(lines)


def get_default_prompts() -> VQAPromptRegistry:
    """Get a registry populated with default fixed-question prompts."""
    registry = VQAPromptRegistry()
    for template in _DEFAULT_PROMPTS.values():
        registry.register(template)
    return registry


def get_default_question_types() -> List[str]:
    """Get list of default question types."""
    return list(_DEFAULT_PROMPTS.keys())
