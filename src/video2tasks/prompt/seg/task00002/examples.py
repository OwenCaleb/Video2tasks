"""Few-shot examples for segment prompt task00002."""

SEGMENT_EXAMPLES = [
    {
        "title": "Example A: Toy car placement",
        "json": """{
  "thought": "Frames 0-4: Gripper approaches toy car. Frames 5-8: Gripper grasps and lifts toy car. Frames 9-12: Gripper moves toy car toward brown basket and releases. Single continuous object interaction.",
  "transitions": [],
  "instructions": ["Place the toy car in the brown basket."]
}""",
    },
    {
        "title": "Example B: Sequential fruit placements",
        "json": """{
  "thought": "Frames 0-2: Gripper places avocado in black basket. Frames 3-5: Gripper switches to orange and places it in black basket. Frames 6-8: Gripper switches to pumpkin and places it in black basket. Frames 9-11: Gripper switches to apple and places it in black basket.",
  "transitions": [3, 6, 9],
  "instructions": ["Place the avocado in the black basket.", "Place the orange in the black basket.", "Place the pumpkin in the black basket.", "Place the apple in the black basket."]
}""",
    },
]
