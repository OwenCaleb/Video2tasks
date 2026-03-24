"""Shared QA question-bank builders (task-agnostic).

Design goals:
1) QA-oriented (not embodied control prompts)
2) Abstract/reusable across tasks via profile injection
3) High diversity in phrasing without synthetic '(variant N)' padding
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class VQAFixedSlotTemplate:
    type_name: str
    description: str
    slots: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class QAProfile:
    """Task profile used to instantiate shared question templates."""

    task_goal: str
    objects: List[str]
    containers: List[str]
    movable_objects: List[str]
    target_mapping: Dict[str, str]  # object -> target container


def _dedup_slots(slots: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, str]] = []
    for slot in slots:
        key = (slot["question"].strip(), slot["answer_space"].strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(slot)
    return out


def _require_min(slots: List[Dict[str, str]], min_required: int, type_name: str) -> List[Dict[str, str]]:
    unique_slots = _dedup_slots(slots)
    if len(unique_slots) < min_required:
        raise ValueError(
            f"Insufficient diversity for '{type_name}': {len(unique_slots)} < {min_required}"
        )
    return unique_slots[:min_required]


def _build_spatial_slots(profile: QAProfile, min_required: int) -> List[Dict[str, str]]:
    slots: List[Dict[str, str]] = []
    rel_phrases = [
        "left of",
        "right of",
        "in front of",
        "behind",
        "above",
        "below",
    ]
    loc_prompts = [
        "Where is {a} relative to {b}",
        "Relative to {b}, what is the position of {a}",
        "Describe {a}'s location with respect to {b}",
        "How is {a} placed in relation to {b}",
    ]
    yn_prompts = [
        "Is {a} {rel} {b}",
        "Does {a} appear {rel} {b}",
        "Would you say {a} is {rel} {b}",
    ]

    objects = profile.objects
    for a in objects:
        for b in objects:
            if a == b:
                continue
            for p in loc_prompts:
                slots.append(
                    {
                        "question": p.format(a=a, b=b),
                        "answer_space": "left/right/front/behind/above/below/inside/on-top/overlapping/near/far",
                    }
                )
            for rel in rel_phrases:
                for p in yn_prompts:
                    slots.append({"question": p.format(a=a, b=b, rel=rel), "answer_space": "yes/no"})
            slots.extend(
                [
                    {
                        "question": f"Is {a} inside {b}",
                        "answer_space": "yes/no",
                    },
                    {
                        "question": f"Is {a} on top of {b}",
                        "answer_space": "yes/no",
                    },
                    {
                        "question": f"Is {a} closer to {b} than to other objects",
                        "answer_space": "yes/no/uncertain",
                    },
                ]
            )

    for c in profile.containers:
        for m in profile.movable_objects:
            slots.extend(
                [
                    {
                        "question": f"Is {m} currently in {c}",
                        "answer_space": "yes/no",
                    },
                    {
                        "question": f"How far is {m} from {c}",
                        "answer_space": "touching/very close/near/medium/far",
                    },
                ]
            )

    return _require_min(slots, min_required, "spatial")


def _build_attribute_slots(profile: QAProfile, min_required: int) -> List[Dict[str, str]]:
    slots: List[Dict[str, str]] = []

    question_styles = [
        "What is the color of {obj}",
        "What shape best describes {obj}",
        "Which material does {obj} seem to be made of",
        "Is {obj} a container",
        "Is {obj} a movable item",
        "What category does {obj} belong to",
        "How would you describe the surface texture of {obj}",
        "Is {obj} rigid or deformable",
        "Is {obj} transparent, opaque, or semi-transparent",
        "Does {obj} have a visible opening",
        "Is {obj} likely to be the task target",
        "What visual cue uniquely identifies {obj}",
        "Is {obj} visually distinct from the other objects",
        "What is the dominant appearance trait of {obj}",
        "Would {obj} be considered a receptacle",
        "Is {obj} likely edible",
        "Does {obj} appear empty or occupied",
    ]
    answer_spaces = {
        "color": "single color phrase",
        "shape": "short shape phrase",
        "material": "material word/phrase",
        "binary": "yes/no",
        "category": "container/object/food/other",
        "texture": "smooth/rough/woven/glossy/matte/other",
        "rigidity": "rigid/deformable/uncertain",
        "visibility": "transparent/opaque/semi-transparent",
        "desc": "short grounded phrase",
    }

    for obj in profile.objects:
        for style in question_styles:
            a_space = answer_spaces["desc"]
            if "color" in style:
                a_space = answer_spaces["color"]
            elif "shape" in style:
                a_space = answer_spaces["shape"]
            elif "material" in style:
                a_space = answer_spaces["material"]
            elif "rigid" in style:
                a_space = answer_spaces["rigidity"]
            elif "transparent" in style:
                a_space = answer_spaces["visibility"]
            elif any(x in style for x in ["Is", "Would", "Does"]):
                a_space = answer_spaces["binary"]
            elif "category" in style:
                a_space = answer_spaces["category"]
            elif "texture" in style:
                a_space = answer_spaces["texture"]
            slots.append({"question": style.format(obj=obj), "answer_space": a_space})

    return _require_min(slots, min_required, "attribute")


def _build_existence_slots(profile: QAProfile, min_required: int) -> List[Dict[str, str]]:
    slots: List[Dict[str, str]] = []
    obj_prompts = [
        "Is {obj} visible in this frame",
        "Can {obj} be clearly observed",
        "Is {obj} partially occluded",
        "Is {obj} fully visible",
        "Is {obj} outside the camera view",
        "Is {obj} present in the workspace",
        "Is {obj} absent from the scene",
        "Do you detect {obj} at all",
        "Is {obj} ambiguous due to blur or occlusion",
        "Is {obj} identifiable with high confidence",
        "Is {obj} likely hidden behind another object",
        "Is {obj} near image boundary",
    ]

    for obj in profile.objects:
        for p in obj_prompts:
            slots.append({"question": p.format(obj=obj), "answer_space": "yes/no"})

    scene_prompts = [
        "Are all required task objects visible",
        "Is any required object missing",
        "Is there any unknown object outside the canonical set",
        "Do at least two canonical objects appear simultaneously",
        "Is the target container visible",
        "Is the movable target visible",
        "Are both containers visible",
        "Is the scene sufficiently clear for QA annotation",
        "Is heavy occlusion present",
        "Is object overlap severe",
        "Can the task state be judged from this frame",
        "Is the frame too ambiguous for reliable annotation",
        "Are there duplicate instances for any canonical object",
        "Is there visual evidence of object disappearance",
        "Is the canonical object set complete in frame",
        "Are non-canonical distractors present",
        "Is container interior visible",
        "Is task-relevant evidence missing",
        "Can target placement state be determined",
        "Is the scene interpretable without temporal context",
    ]
    for p in scene_prompts:
        slots.append({"question": p, "answer_space": "yes/no"})

    return _require_min(slots, min_required, "existence")


def _build_count_slots(profile: QAProfile, min_required: int) -> List[Dict[str, str]]:
    slots: List[Dict[str, str]] = []

    count_targets = [
        "visible objects",
        "visible containers",
        "visible movable objects",
        "task-relevant objects",
        "objects inside target container",
        "objects inside non-target containers",
        "objects touching each other",
        "occluded objects",
        "clearly identifiable objects",
        "objects near center region",
        "objects near image boundary",
        "objects with uncertain identity",
        "containers that appear empty",
        "containers that appear occupied",
        "instances of each canonical object",
    ]
    forms = [
        "How many {t} are there",
        "Count the number of {t}",
        "What is the total count of {t}",
        "How many {t} can be confirmed",
        "Provide the number of {t}",
    ]

    for t in count_targets:
        for f in forms:
            slots.append({"question": f.format(t=t), "answer_space": "integer"})

    specific_prompts = [
        f"How many instances of {obj} are visible" for obj in profile.objects
    ] + [
        f"How many items are inside {c}" for c in profile.containers
    ]
    specific_prompts += [
        "How many canonical classes are present",
        "How many object classes are missing",
        "How many objects are likely target candidates",
    ]
    for p in specific_prompts:
        slots.append({"question": p, "answer_space": "integer"})

    return _require_min(slots, min_required, "count")


def _build_task_reasoning_slots(profile: QAProfile, min_required: int) -> List[Dict[str, str]]:
    """Task-level QA slots (named as 'manipulation' for compatibility).

    This is QA reasoning, not embodied control policy.
    """
    slots: List[Dict[str, str]] = []
    for obj, container in profile.target_mapping.items():
        slots.extend(
            [
                {
                    "question": f"According to the goal, where should {obj} be placed",
                    "answer_space": "target container name",
                },
                {
                    "question": f"Is placing {obj} into {container} consistent with the goal",
                    "answer_space": "yes/no",
                },
                {
                    "question": f"Would placing {obj} into a non-target container violate the goal",
                    "answer_space": "yes/no",
                },
                {
                    "question": f"Has the goal for {obj} been completed in this frame",
                    "answer_space": "yes/no/uncertain",
                },
            ]
        )

    status_prompts = [
        "Is the overall task completed",
        "Is the task still in progress",
        "Is there enough evidence to declare completion",
        "What part of the goal remains unfinished",
        "Which object currently appears to be pending placement",
        "Does the frame show a goal-consistent state",
        "Does the frame show a goal-conflicting state",
        "Is the target mapping inferable from context",
        "What is the most likely current subtask",
        "What is the immediate goal-consistent next state",
        "What is the key visual evidence for current task status",
        "Is the target container currently receiving the correct object",
        "Is any object in a wrong container",
        "Is correction needed to satisfy the goal",
        "Could the scene already satisfy the goal despite ambiguity",
        "Is there contradictory evidence against completion",
        "Can completion be judged from this single frame",
        "What uncertainty prevents confident status labeling",
        "Does the scene indicate pre-placement, in-placement, or post-placement stage",
        "Which canonical object should be tracked to verify completion",
        "Is the target object nearer to target container than non-target container",
        "Does object-container arrangement align with goal semantics",
        "Is the scene state reversible without violating goal",
        "Is there any risk of mislabeling completion due to occlusion",
        "Would an unknown answer be safer for task status",
        "Does the frame provide direct evidence of placement outcome",
        "Is the goal statement sufficient to disambiguate target container",
        "Could this frame belong to a different but similar task",
        "Are all canonical objects relevant to current task state",
        "Is the task state consistent with earlier placement progress",
    ]
    for p in status_prompts:
        slots.append({"question": p, "answer_space": "short grounded answer"})

    comparative_forms = [
        "Which is more likely: task complete or task incomplete",
        "Which interpretation is better supported: correct placement or incorrect placement",
        "Is completion confidence high, medium, or low",
        "Should annotation mark status as complete, incomplete, or uncertain",
        "Is the evidence decisive or ambiguous for goal fulfillment",
        "Does visual evidence strongly support target-object-in-target-container",
        "Is there any stronger alternative explanation than goal completion",
        "Is task-state labeling robust under viewpoint uncertainty",
        "Would a conservative annotator mark this as uncertain",
        "Is this frame suitable for definitive task-status QA",
        "Does this frame require temporal context for reliable status",
        "Is the current frame semantically aligned with goal text",
        "Which object-container pair carries the strongest status signal",
        "Is the inferred status stable under small localization error",
        "Does container occupancy evidence support completion",
        "Can wrong-container evidence be ruled out confidently",
        "Is it plausible that placement happened outside visible area",
        "Does this frame show post-placement stabilization",
        "Does this frame show pre-placement intent only",
        "Is status annotation likely to change with next frame",
    ]
    for p in comparative_forms:
        slots.append({"question": p, "answer_space": "yes/no/label with short reason"})

    return _require_min(slots, min_required, "manipulation")


def build_shared_question_bank(profile: QAProfile, min_per_type: int = 50) -> Dict[str, VQAFixedSlotTemplate]:
    """Build a reusable, QA-oriented question bank for any task profile."""
    return {
        "spatial": VQAFixedSlotTemplate(
            type_name="spatial",
            description="Spatial relations and placement geometry",
            slots=_build_spatial_slots(profile, min_per_type),
        ),
        "attribute": VQAFixedSlotTemplate(
            type_name="attribute",
            description="Object attributes and category cues",
            slots=_build_attribute_slots(profile, min_per_type),
        ),
        "existence": VQAFixedSlotTemplate(
            type_name="existence",
            description="Presence, visibility, and ambiguity",
            slots=_build_existence_slots(profile, min_per_type),
        ),
        "count": VQAFixedSlotTemplate(
            type_name="count",
            description="Counting and cardinality",
            slots=_build_count_slots(profile, min_per_type),
        ),
        "manipulation": VQAFixedSlotTemplate(
            type_name="manipulation",
            description="Task-status and goal-consistency QA",
            slots=_build_task_reasoning_slots(profile, min_per_type),
        ),
    }
