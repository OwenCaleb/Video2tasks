"""Reusable text blocks for CoT prompt task00001."""

ROLE_BLOCK = "You are a robotic manipulation state reasoner."

CANONICAL_POLICY_BLOCK = (
    "### Canonical Reference Policy\n"
    "1. Mention objects and containers only with CanonicalRef names from inventory.\n"
)

SLOT_RULE_BLOCK = (
    "### Required state slots\n"
    "Fill exactly 4 slots:\n"
    "1. Plan\n"
    "2. Objects\n"
    "3. Next\n"
    "4. Reason\n"
)

OUTPUT_BLOCK = (
    "### Output format\n"
    "Return ONLY valid JSON, no markdown or extra text.\n"
    "{\n"
    '  "instruction": "<high-level task, verbatim>",\n'
    '  "cot": "I need to determine the next object to be operated.\\n'
    "Plan: <...>\\n"
    "Objects: [<...>]\\n"
    "Next: <...>\\n"
    'Reason: <...>"\n'
    "}"
)
