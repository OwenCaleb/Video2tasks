"""Few-shot examples for segment prompt task00002."""

SEGMENT_EXAMPLES = [
    {
        "title": "Example A: Grapes to black basket",
        "json": """{
  "thought": "Frames 0-4: Gripper approaches grapes. Frames 5-8: Gripper grasps and lifts grapes. Frames 9-12: Gripper moves grapes toward black basket and releases. Single continuous object interaction.",
  "transitions": [],
  "instructions": ["Put the grapes into the black basket"]
}""",
    },
    {
        "title": "Example B: Basket adjustment then grapes",
        "json": """{
  "thought": "Frames 0-3: Gripper adjusts black basket position. Frame 4: Releases black basket. Frames 5-7: Gripper grasps grapes. Object changed from black basket to grapes, so a switch occurs at frame 4.",
  "transitions": [4],
  "instructions": ["Adjust the black basket", "Put the grapes into the black basket"]
}""",
    },
]
