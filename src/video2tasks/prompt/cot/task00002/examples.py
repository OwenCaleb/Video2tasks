"""Few-shot examples for CoT prompt task00002."""

COT_EXAMPLES = [
    {
        "title": "Example A",
        "json": """{
  "instruction": "Put the grapes into the black basket.",
  "cot": "I need to determine the next object to be operated.\\nPlan: {grapes -> black basket}\\nObjects: [grapes (center), black basket (right), brown basket (left)]\\nNext: grapes\\nReason: target object for only remaining goal"
}""",
    },
    {
        "title": "Example B",
        "json": """{
  "instruction": "Put the grapes into the black basket.",
  "cot": "I need to determine the next object to be operated.\\nPlan: {grapes -> black basket}\\nObjects: [grapes (near gripper), black basket (far right), brown basket (rear)]\\nNext: black basket\\nReason: ensure target container is reachable and stable before placement"
}""",
    },
]
