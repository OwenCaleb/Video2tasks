"""CoT prompt for generating chain-of-thought reasoning from subtask + frames."""


def build_cot_prompt(instruction: str, n_images: int) -> str:
    """Build prompt for CoT reasoning.

    Given *n_images* frames from a robot manipulation video segment and a
    short subtask label (*instruction*), the VLM should produce a concise
    chain-of-thought that describes:
      1. The current scene / relevant object positions.
      2. The high-level goal (inferred from instruction & context).
      3. A step-by-step motion plan (grasp → lift → move → place …).
      4. Potential cautions (collision avoidance, fragile objects, etc.).

    The model must return **only** a JSON object.
    """
    return (
        f"You are a robotic manipulation planner.\n"
        f"You are given {n_images} frame(s) from a video segment where a robot performs the subtask:\n"
        f'  "{instruction}"\n\n'
        "Based on the visual scene, generate a chain-of-thought (CoT) that explains:\n"
        "1. SCENE: Describe the relevant objects and their positions.\n"
        "2. GOAL: What high-level goal does this subtask serve?\n"
        "3. PLAN: Step-by-step motion plan (e.g. approach, grasp, lift, move, place).\n"
        "4. CAUTION: Any risks to avoid (collisions, dropping, fragile items).\n\n"
        "IMPORTANT: Return ONLY a valid JSON object. No markdown, no code fences.\n\n"
        "Output format:\n"
        "{\n"
        '  "instruction": "<the subtask label>",\n'
        '  "cot": "<Your chain-of-thought reasoning in one paragraph>"\n'
        "}\n\n"
        "Example (format only, do not copy content):\n"
        "{\n"
        '  "instruction": "pick up red toy car",\n'
        '  "cot": "The red toy car is on the right side of the table. Goal is to clear the tabletop, '
        "so grasp it, lift it, and place it into the light-brown basket. Avoid bumping other items.\"\n"
        "}"
    )
