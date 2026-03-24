"""Few-shot examples for segment prompt task00001."""

SEGMENT_EXAMPLES = [
    {
        "title": "Example A: Sorting (True Switch)",
        "json": """{
  "thought": "Frames 0-4: Robot carries the red car. Frame 5: Releases red car into brown basket. Frames 6-7: Grasps gold car. Switch at frame 5.",
  "transitions": [5],
  "instructions": ["Put the red car into the brown basket", "Put the gold car into the brown basket"]
}""",
    },
    {
        "title": "Example B: Regrasping (No Switch)",
        "json": """{
  "thought": "Frames 0-6: Robot regrips the same red car for alignment. No object change, so no switch.",
  "transitions": [],
  "instructions": ["Put the red car into the brown basket"]
}""",
    },
    {
        "title": "Example C: Placement Adjustment (Continuous)",
        "json": """{
  "thought": "Frames 0-10: Robot places and adjusts kiwi in black basket without changing target object.",
  "transitions": [],
  "instructions": ["Put the kiwi into the black basket"]
}""",
    },
]
