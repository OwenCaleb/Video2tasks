"""CoT prompt — structured state-machine reasoning: Instruction → CoT → Subtask.

The CoT is NOT free-form prose. It is a compact state snapshot with named
slots (plan decomposition, visible objects, next operand, reason) that
together deterministically justify the subtask selection.
"""


def build_cot_prompt(
    high_level_instruction: str,
    subtask: str,
    n_images: int,
) -> str:
    """Build a prompt that asks the VLM to reverse-engineer structured CoT.

    Given:
      * high_level_instruction — the overall goal
      * subtask — the concrete next action that was chosen
      * n_images frames from the video segment

    The VLM must produce a compact **state snapshot** (not prose) that
    bridges  Instruction → CoT → Subtask.  Only 4 slots:
      Plan / Objects / Next / Reason
    """
    return (
        "You are a robotic manipulation state reasoner.\n\n"
        "### Inputs\n"
        f"High-level task: \"{high_level_instruction}\"\n"
        f"Subtask performed: \"{subtask}\"\n"
        f"Visual evidence: {n_images} frame(s) from the video segment\n\n"
        "Object inventory (some props may repeat):\n"
        "- Containers:\n"
        "Descriptor : CanonicalRef\n"
        "A brown rectangular bamboo-woven storage basket : brown basket\n"
        "A black rectangular bamboo-woven storage basket : black basket\n"
        "- Fruit:\n"
        "A green round kiwifruit : kiwi\n"
        "Half a kiwifruit with a brown fuzzy skin and a green interior (a kind of ground fruit when look) : kiwi\n"
        "A purple bunch of grapes : grapes\n"
        "Half a red apple with a white interior : apple\n"
        "A yellow avocado with a white center and an orange pit : avocado\n"
        "A green avocado (alligator pear) with a bumpy, textured skin : avocado\n"
        "An orange with a bright orange peel : orange\n"
        "A small round orange-colored citrus fruit with a green stem : orange\n"
        "- Toy cars:\n"
        "A small red toy truck : red car\n"
        "A small red toy car : red car\n"
        "A small gold toy car : gold car\n"
        "A small black toy car : black car\n"
        "Various toy cars : <color> car\n"
        "Rule for Various toy cars: if the car color is identifiable, set CanonicalRef to \"<color> car\" (e.g., blue car, white car). Use a single color word.\n\n"
        "### Canonical Reference Policy\n"
        "1. In all answers, mention objects and containers using only CanonicalRef exactly as listed above.\n" 
        "### Your job\n"
        "Reconstruct the **compact state-based reasoning** (CoT) that a planner "
        "would use to decide this subtask is the correct next action.\n\n"
        "The CoT must be an **explicit state snapshot** with exactly 4 named slots. "
        "No prose, no extra commentary.\n\n"
        "### Required state slots\n"
        "Fill in every slot below by observing the frames:\n\n"
        "1. `Plan` — decompose the task into sub-goal mappings or an ordered sequence "
        "(e.g. {toy cars → brown basket, fruit → black basket}).\n"
        "2. `Objects` — list all task-relevant objects visible in the frames, "
        "with brief spatial notes (left / right / near gripper …).\n"
        "3. `Next` — the specific object to operate on now.\n"
        "4. `Reason` — one short phrase explaining why this object is chosen next "
        "(proximity / reachability / priority / category).\n\n"
        "### Output format\n"
        "Return ONLY a valid JSON object. No markdown, no code fences, no extra text.\n\n"
        "{\n"
        '  "instruction": "<high-level task, verbatim>",\n'
        '  "cot": "I need to determine the next object to be operated.\\n'
        "Plan: <sub-goal mappings or sequence>\\n"
        "Objects: [<object (position), …>]\\n"
        "Next: <object>\\n"
        'Reason: <why>"\n'
        "}\n\n"
        "### Example (format & style reference — do NOT copy content)\n"
        "{\n"
        '  "instruction": "Put the toy cars into the brown basket and put the fruit '
        'into the black basket",\n'
        '  "cot": "I need to determine the next object to be operated.\\n'
        "Plan: {toy cars → brown basket, fruit → black basket}\\n"
        "Objects: [red car (right), gold car (center), avocado (left edge), kiwi (far left), orange (near center)]\\n"
        "Next: avocado\\n"
        'Reason: closest to gripper, fruit category"\n'
        "}"
    )
