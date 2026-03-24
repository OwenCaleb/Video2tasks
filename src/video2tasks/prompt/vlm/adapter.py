"""Generic VQA adapter for profile-driven task onboarding.

New tasks only need a task profile (and optional demos/context).
No task-specific question bank module is required.
"""

import json
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .question_bank_shared import QAProfile, VQAFixedSlotTemplate, build_shared_question_bank


@dataclass
class VQATaskAdapter:
    """Lightweight adapter config for one VQA task."""

    profile: QAProfile
    task_context: str = ""
    min_per_type: int = 50
    demos: Optional[Dict[str, List[Dict]]] = None


class VQAPromptRegistry:
    """Registry for fixed-question VQA prompt templates."""

    def __init__(
        self,
        question_bank: Dict[str, VQAFixedSlotTemplate],
        demos: Optional[Dict[str, List[Dict]]] = None,
        default_context: str = "",
        canonical_refs: Optional[List[str]] = None,
    ) -> None:
        self._prompts: Dict[str, VQAFixedSlotTemplate] = dict(question_bank)
        self._demos = demos or {}
        self._default_context = default_context.strip()
        self._canonical_refs = list(canonical_refs or [])

    def get(self, type_name: str) -> Optional[VQAFixedSlotTemplate]:
        return self._prompts.get(type_name)

    def list_types(self) -> List[str]:
        return list(self._prompts.keys())

    def _pick_slots(
        self,
        slots: List[Dict[str, str]],
        max_questions: Optional[int],
        now_ns: int,
    ) -> List[Dict[str, str]]:
        if not slots:
            return []
        if max_questions is None or max_questions <= 0 or max_questions >= len(slots):
            return slots
        rng = random.Random(now_ns)
        return rng.sample(slots, k=max_questions)

    def _pick_demo(self, type_name: str, now_ns: int) -> Optional[Dict]:
        candidates = self._demos.get(type_name, [])
        if not candidates:
            return None
        return candidates[now_ns % len(candidates)]

    def _canonical_ref_line(self) -> str:
        refs = sorted(self._canonical_refs)
        if not refs:
            return "Use only CanonicalRef names from task context."
        return "Use only CanonicalRef names: " + ", ".join(refs) + "."

    def build_single_type_prompt(
        self,
        type_name: str,
        n_images: int = 1,
        max_questions: Optional[int] = None,
        task_context: Optional[str] = None,
    ) -> str:
        template = self.get(type_name)
        if template is None:
            raise ValueError(
                f"Unknown question type: {type_name}. Available: {self.list_types()}"
            )

        now_ns = time.time_ns()
        selected_slots = self._pick_slots(template.slots, max_questions, now_ns)
        demo = self._pick_demo(type_name, now_ns)
        context = task_context.strip() if task_context else self._default_context

        lines = [
            (
                f"You are analyzing {'an image' if n_images == 1 else f'{n_images} images'} "
                "for visual question answering (QA)."
            ),
            f"Question type: {type_name}",
            "",
        ]
        if context:
            lines.extend(["### Task Context", context, ""])

        lines.extend(
            [
                "### Canonical Reference Policy",
                self._canonical_ref_line(),
                "Return one QA for each selected template.",
                "If visual evidence is insufficient, answer 'unknown' instead of guessing.",
                "Do not invent objects, containers, or events not present in context/evidence.",
                "Keep each answer concise and evidence-grounded.",
                "",
                "### Selected Templates",
            ]
        )

        for idx, slot in enumerate(selected_slots, 1):
            lines.append(f'{idx}. "{slot["question"]}"')
            lines.append(f'   Answer space: {slot["answer_space"]}')

        lines.append("")
        if demo is not None:
            lines.append("### Demo")
            lines.append(json.dumps(demo, ensure_ascii=False))
            lines.append("")

        lines.extend(
            [
                "### Output",
                "Return ONLY valid JSON:",
                "Rules:",
                "1) Output key must be exactly 'qas'.",
                "2) Number of QAs must equal number of selected templates.",
                "3) For each QA, type/question must match the selected template semantics.",
                "4) No markdown, no commentary, no extra keys.",
                '{"qas":[{"type":"' + type_name + '","question":"...","answer":"..."}]}',
            ]
        )
        return "\n".join(lines)


def build_registry(adapter: VQATaskAdapter) -> VQAPromptRegistry:
    question_bank = build_shared_question_bank(adapter.profile, min_per_type=adapter.min_per_type)
    return VQAPromptRegistry(
        question_bank=question_bank,
        demos=adapter.demos,
        default_context=adapter.task_context,
        canonical_refs=adapter.profile.objects,
    )


def build_question_types(adapter: VQATaskAdapter) -> List[str]:
    return list(build_shared_question_bank(adapter.profile, min_per_type=adapter.min_per_type).keys())
