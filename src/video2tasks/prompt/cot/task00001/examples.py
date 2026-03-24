"""Few-shot examples for CoT prompt task00001."""

COT_EXAMPLES = [
    {
        "title": "Example A",
        "json": """{
  "instruction": "Put the toy cars into the brown basket and put the fruit into the black basket",
  "cot": "I need to determine the next object to be operated.\\nPlan: {toy cars -> brown basket, fruit -> black basket}\\nObjects: [red car (right), gold car (center), avocado (left edge)]\\nNext: avocado\\nReason: closest to gripper, fruit category"
}""",
    },
    {
        "title": "Example B",
        "json": """{
  "instruction": "Put the toy cars into the brown basket and put the fruit into the black basket",
  "cot": "I need to determine the next object to be operated.\\nPlan: {toy cars -> brown basket, fruit -> black basket}\\nObjects: [black car (near gripper), kiwi (far left), brown basket (rear)]\\nNext: black car\\nReason: in-reach and belongs to pending toy-car goal"
}""",
    },
]
