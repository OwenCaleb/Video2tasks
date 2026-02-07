"""VQA prompt templates for different question types.

This module provides configurable and extensible prompt templates for VQA.
New question types can be added by registering new prompt templates.
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class VQAPromptTemplate:
    """A VQA prompt template for a specific question type."""
    type_name: str
    description: str
    prompt_template: str
    output_format: str = "json"
    examples: List[Dict[str, str]] = field(default_factory=list)


# Default prompt templates for robotics manipulation domain
_DEFAULT_PROMPTS: Dict[str, VQAPromptTemplate] = {}

# Compact questions for retry/fallback prompt
_COMPACT_QUESTIONS: Dict[str, List[str]] = {
    "spatial": [
        "What is left of the robot gripper?",
        "What is right of the robot gripper?",
    ],
    "attribute": [
        "What is the color of the most salient object?",
        "Is any object open or closed?",
    ],
    "existence": [
        "Is there a robot gripper visible?",
        "Is any object being held?",
    ],
    "count": [
        "How many graspable objects are visible?",
    ],
    "manipulation": [
        "Which objects are graspable?",
        "What is the gripper state?",
    ],
}


def _register_default(template: VQAPromptTemplate) -> VQAPromptTemplate:
    """Register a default prompt template."""
    _DEFAULT_PROMPTS[template.type_name] = template
    return template


# Spatial relationship questions
_register_default(VQAPromptTemplate(
    type_name="spatial",
    description="Spatial relationship questions (left/right/front/behind/inside/on/top-of/distance)",
    prompt_template="""Analyze the spatial relationships between objects in this image.
Focus on the robot gripper/hand and objects in the scene.

Answer the following spatial questions about the image:
1. What objects are to the LEFT of the robot gripper?
2. What objects are to the RIGHT of the robot gripper?
3. What objects are IN FRONT of the robot gripper?
4. What objects are BEHIND the robot gripper?
5. Are there any objects ON TOP of other objects?
6. Are there any objects INSIDE containers?
7. Estimate the DISTANCE (close/medium/far) from the gripper to the nearest graspable object.

Respond with a valid JSON object:
{
  "qas": [
    {"type": "spatial", "question": "What objects are to the LEFT of the gripper?", "answer": "..."},
    {"type": "spatial", "question": "What objects are to the RIGHT of the gripper?", "answer": "..."},
    {"type": "spatial", "question": "What objects are IN FRONT of the gripper?", "answer": "..."},
    {"type": "spatial", "question": "What objects are BEHIND the gripper?", "answer": "..."},
    {"type": "spatial", "question": "Are there objects ON TOP of other objects?", "answer": "..."},
    {"type": "spatial", "question": "Are there objects INSIDE containers?", "answer": "..."},
    {"type": "spatial", "question": "Distance to nearest graspable object?", "answer": "..."}
  ]
}""",
    examples=[
        {"question": "What objects are to the LEFT of the gripper?", "answer": "A red cup and a blue plate"},
        {"question": "Distance to nearest graspable object?", "answer": "close (approximately 5cm)"},
    ]
))


# Attribute questions
_register_default(VQAPromptTemplate(
    type_name="attribute",
    description="Object attribute questions (color, material, state, etc.)",
    prompt_template="""Analyze the attributes of objects visible in this image.
Focus on objects relevant to robot manipulation tasks.

For each visible object, identify:
1. COLOR: What color is each object?
2. MATERIAL: What material appears to be (metal/plastic/wood/fabric/glass)?
3. STATE: Is any object open/closed? On/off? Empty/full?
4. GRASP STATE: Is the robot currently grasping any object?

Respond with a valid JSON object:
{
  "qas": [
    {"type": "attribute", "question": "What color is the [object]?", "answer": "..."},
    {"type": "attribute", "question": "What material is the [object]?", "answer": "..."},
    {"type": "attribute", "question": "What is the state of the [object]?", "answer": "..."},
    {"type": "attribute", "question": "Is the robot grasping any object?", "answer": "..."}
  ]
}""",
    examples=[
        {"question": "What color is the cup?", "answer": "red"},
        {"question": "Is the drawer open or closed?", "answer": "open"},
    ]
))


# Existence / Yes-No questions
_register_default(VQAPromptTemplate(
    type_name="existence",
    description="Existence and yes/no questions",
    prompt_template="""Answer yes/no existence questions about this image.
Focus on objects and conditions relevant to robot manipulation.

Answer these questions:
1. Is there a robot gripper/hand visible in the image?
2. Are there any graspable objects on the table/surface?
3. Is there any object currently being held?
4. Are there any obstacles blocking the robot's path?
5. Is the workspace clear for manipulation?

Respond with a valid JSON object:
{
  "qas": [
    {"type": "existence", "question": "Is there a robot gripper visible?", "answer": "yes/no"},
    {"type": "existence", "question": "Are there graspable objects on the surface?", "answer": "yes/no"},
    {"type": "existence", "question": "Is any object currently being held?", "answer": "yes/no"},
    {"type": "existence", "question": "Are there obstacles blocking the path?", "answer": "yes/no"},
    {"type": "existence", "question": "Is the workspace clear?", "answer": "yes/no"}
  ]
}""",
    examples=[
        {"question": "Is there a robot gripper visible?", "answer": "yes"},
        {"question": "Is the workspace clear?", "answer": "no, there are multiple objects"},
    ]
))


# Counting questions
_register_default(VQAPromptTemplate(
    type_name="count",
    description="Object counting questions",
    prompt_template="""Count objects visible in this image.
Focus on countable objects relevant to robot manipulation.

Answer these counting questions:
1. How many distinct graspable objects are visible?
2. How many containers (cups, bowls, boxes) are visible?
3. How many tools or utensils are visible?
4. How many objects are currently on the table/surface?

Respond with a valid JSON object:
{
  "qas": [
    {"type": "count", "question": "How many graspable objects are visible?", "answer": "N"},
    {"type": "count", "question": "How many containers are visible?", "answer": "N"},
    {"type": "count", "question": "How many tools/utensils are visible?", "answer": "N"},
    {"type": "count", "question": "How many objects are on the surface?", "answer": "N"}
  ]
}""",
    examples=[
        {"question": "How many graspable objects are visible?", "answer": "5"},
        {"question": "How many containers are visible?", "answer": "2 (one cup and one bowl)"},
    ]
))


# Robot manipulation specific questions
_register_default(VQAPromptTemplate(
    type_name="manipulation",
    description="Robot manipulation specific questions (graspability, target position, action hints)",
    prompt_template="""Analyze this image from a robot manipulation perspective.

Answer these manipulation-related questions:
1. GRASPABLE: Which objects appear graspable by a robot gripper?
2. GRASP POINT: For the most prominent graspable object, where should the gripper grasp it?
3. TARGET POSITION: If there's an ongoing task, where should the object be placed?
4. ACTION HINT: What manipulation action appears to be happening or should happen next?
5. GRIPPER STATE: Is the gripper open, closed, or partially closed?

Respond with a valid JSON object:
{
  "qas": [
    {"type": "manipulation", "question": "Which objects are graspable?", "answer": "..."},
    {"type": "manipulation", "question": "Where should the gripper grasp the [object]?", "answer": "..."},
    {"type": "manipulation", "question": "Where should the object be placed?", "answer": "..."},
    {"type": "manipulation", "question": "What action should happen next?", "answer": "..."},
    {"type": "manipulation", "question": "What is the gripper state?", "answer": "..."}
  ]
}""",
    examples=[
        {"question": "Which objects are graspable?", "answer": "The red cup and the spoon are graspable"},
        {"question": "Where should the gripper grasp the cup?", "answer": "At the handle on the right side"},
        {"question": "What action should happen next?", "answer": "Pick up the cup and move it to the left"},
    ]
))


class VQAPromptRegistry:
    """Registry for VQA prompt templates.
    
    Supports adding custom prompt templates without modifying core code.
    """
    
    def __init__(self) -> None:
        self._prompts: Dict[str, VQAPromptTemplate] = {}
    
    def register(self, template: VQAPromptTemplate) -> None:
        """Register a prompt template."""
        self._prompts[template.type_name] = template
    
    def get(self, type_name: str) -> Optional[VQAPromptTemplate]:
        """Get a prompt template by type name."""
        return self._prompts.get(type_name)
    
    def list_types(self) -> List[str]:
        """List all registered question types."""
        return list(self._prompts.keys())
    
    def get_prompt(self, type_name: str) -> str:
        """Get the prompt string for a question type."""
        template = self.get(type_name)
        if template is None:
            raise ValueError(f"Unknown question type: {type_name}. Available: {self.list_types()}")
        return template.prompt_template
    
    def build_combined_prompt(self, question_types: List[str], n_images: int = 1) -> str:
        """Build a combined prompt for multiple question types."""
        if not question_types:
            question_types = self.list_types()
        
        parts = []
        parts.append(f"You are analyzing {'an image' if n_images == 1 else f'{n_images} images'} from a robot manipulation scenario.")
        parts.append("Answer the following VQA questions about the image(s).")
        parts.append("IMPORTANT: Return ONLY a single JSON object with key 'qas'.")
        parts.append("Do NOT add markdown/code fences or extra text. If unsure, answer 'unknown'.")
        parts.append("")
        
        for qtype in question_types:
            template = self.get(qtype)
            if template:
                parts.append(f"=== {qtype.upper()} QUESTIONS ===")
                parts.append(template.prompt_template)
                parts.append("")
        
        parts.append("Combine all answers into a single JSON response:")
        parts.append('{')
        parts.append('  "qas": [')
        parts.append('    {"type": "...", "question": "...", "answer": "..."},')
        parts.append('    ...')
        parts.append('  ]')
        parts.append('}')
        parts.append("")
        parts.append("Example (format only, do not copy content):")
        parts.append('{')
        parts.append('  "qas": [')
        parts.append('    {"type": "existence", "question": "Is there a robot gripper visible?", "answer": "yes"},')
        parts.append('    {"type": "count", "question": "How many graspable objects are visible?", "answer": "2"}')
        parts.append('  ]')
        parts.append('}')
        
        return "\n".join(parts)


def get_default_prompts() -> VQAPromptRegistry:
    """Get a registry populated with default prompts."""
    registry = VQAPromptRegistry()
    for template in _DEFAULT_PROMPTS.values():
        registry.register(template)
    return registry


def get_default_question_types() -> List[str]:
    """Get list of default question types."""
    return list(_DEFAULT_PROMPTS.keys())


def build_compact_prompt(question_types: List[str], n_images: int = 1) -> str:
    """Build a compact VQA prompt for retry/fallback.

    This reduces prompt length to improve JSON compliance.
    """
    types = question_types or list(_COMPACT_QUESTIONS.keys())

    parts = []
    parts.append(f"You are analyzing {'an image' if n_images == 1 else f'{n_images} images'} from a robot manipulation scene.")
    parts.append("Return ONLY JSON with key 'qas'. No extra text.")
    parts.append("If unsure, answer 'unknown'.")
    parts.append("")
    parts.append("Questions:")

    for qtype in types:
        qs = _COMPACT_QUESTIONS.get(qtype, [])
        for q in qs:
            parts.append(f"- [{qtype}] {q}")

    parts.append("")
    parts.append("JSON format:")
    parts.append('{')
    parts.append('  "qas": [')
    parts.append('    {"type": "existence", "question": "Is there a robot gripper visible?", "answer": "yes"}')
    parts.append('  ]')
    parts.append('}')

    return "\n".join(parts)
