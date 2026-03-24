"""Reusable text blocks for segment prompt task00001."""

ROLE_AND_GOAL_BLOCK = (
    "### Goal\n"
    "Detect **Atomic Task Boundaries** (Switch Points).\n"
    "A 'Switch' occurs strictly when the robot **completes** interaction with one object and "
    "**starts** interacting with a DIFFERENT object.\n\n"
)

CORE_LOGIC_BLOCK = (
    "### Core Logic (The 'Distinct Object' Rule)\n"
    "1. **True Switch:** Robot releases Object A and then grasps Object B. -> MARK SWITCH.\n"
    "2. **False Switch:** Different manipulations on the same object are NOT switches.\n"
    "3. **Visual Similarity:** For similar-looking objects, require clear release-then-grasp evidence.\n\n"
)

CANONICAL_POLICY_BLOCK = (
    "### Canonical Reference Policy\n"
    "1. CanonicalRef is the only allowed name for entities in output JSON, especially instructions.\n"
    "### Instruction Template Constraint\n"
    "- Each instruction must follow: Put the <OBJECT> into the <Object>\n"
    "- <OBJECT> must be CanonicalRef in the inventory.\n\n"
)

OUTPUT_FORMAT_BLOCK = (
    "### Output Format: Strict JSON\n"
    "Return a valid JSON object with keys: thought, transitions, instructions.\n"
)
