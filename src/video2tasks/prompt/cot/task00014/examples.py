"""Few-shot examples for CoT prompt task00002."""

COT_EXAMPLES = [
    {
        "title": "Example A",
        "json": """{
  "instruction": "Place the toy car in the brown basket.",
  "cot": "I need to determine the next object to be operated.\nPlan: {toy car -> brown basket}\nObjects: [toy car (center), brown basket (left), black basket (right)]\nNext: toy car\nReason: toy car is the active object for the current goal"
}""",
    },
    {
        "title": "Example B",
        "json": """{
  "instruction": "Place the avocado in the black basket.\nPlace the orange in the black basket.\nPlace the pumpkin in the black basket.\nPlace the apple in the black basket.",
  "cot": "I need to determine the next object to be operated.\nPlan: {avocado -> black basket, orange -> black basket, pumpkin -> black basket, apple -> black basket}\nObjects: [avocado (near gripper), orange (center), pumpkin (rear), apple (left), black basket (right)]\nNext: avocado\nReason: avocado is closest to the gripper and belongs to the current placement sequence"
}""",
    },
]
