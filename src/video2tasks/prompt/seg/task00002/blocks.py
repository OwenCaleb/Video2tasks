"""Reusable text blocks for segment prompt task00002."""

ROLE_AND_GOAL_BLOCK = (
    "### Goal\n"
    "Detect atomic task boundaries in a short manipulation clip.\n"
    "A boundary exists only when interaction shifts from one object to a different object.\n\n"
)

CORE_LOGIC_BLOCK = (
    "### Core Logic\n"
    "1. True switch: release Object A and then grasp Object B.\n"
    "2. False switch: regrasping or adjusting the same object is continuous.\n"
    "3. Keep boundaries conservative; do not over-segment.\n\n"
)

CANONICAL_POLICY_BLOCK = (
    "### Canonical Reference Policy\n"
    "Use only these CanonicalRef names: black basket, brown basket, toy car, avocado, orange, pumpkin, apple.\n"
    "### Instruction Template Constraint\n"
    "Instruction format: Place the <OBJECT> in the <OBJECT>.\n"
    "Use CanonicalRef exactly.\n\n"
)

OUTPUT_FORMAT_BLOCK = (
    "### Output Format\n"
    "Return strict JSON with keys: thought, transitions, instructions.\n"
)
