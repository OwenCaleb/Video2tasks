"""Reusable text blocks for CoT prompt task00002."""

ROLE_BLOCK = "You are a robotic manipulation state reasoner."

CANONICAL_POLICY_BLOCK = (
    "### Canonical Reference Policy\n"
    "Use only these CanonicalRef names: black basket, brown basket, grapes.\n"
)

SLOT_RULE_BLOCK = (
    "### Required slots\n"
    "Output must contain exactly:\n"
    "1. Plan\n"
    "2. Objects\n"
    "3. Next\n"
    "4. Reason\n"
)

OUTPUT_BLOCK = (
    "### Output format\n"
    "Return only strict JSON:\n"
    "{\n"
    '  "instruction": "<high-level task>",\n'
    '  "cot": "I need to determine the next object to be operated.\\n'
    "Plan: <...>\\n"
    "Objects: [<...>]\\n"
    "Next: <...>\\n"
    'Reason: <...>"\n'
    "}\n"
)
