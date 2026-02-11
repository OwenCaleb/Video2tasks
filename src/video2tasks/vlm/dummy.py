"""Dummy VLM backend for testing."""

from typing import List, Dict, Any
import numpy as np
from .base import VLMBackend


class DummyBackend(VLMBackend):
    """Dummy backend that returns mock results without loading models."""
    
    @property
    def name(self) -> str:
        return "dummy"
    
    def infer(self, images: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        """Return mock results for segmentation or VQA prompts."""
        n = len(images)

        if "\"qas\"" in prompt or "qas" in prompt:
            # Detect [Object] expansion prompts
            if "[Object]" in prompt:
                return {
                    "qas": [
                        {"type": "spatial", "question": "What objects are to the LEFT of brown basket?", "answer": "kiwi, avocado (dummy)"},
                        {"type": "spatial", "question": "What objects are to the LEFT of red car?", "answer": "brown basket (dummy)"},
                        {"type": "spatial", "question": "What objects are to the RIGHT of brown basket?", "answer": "red car (dummy)"},
                        {"type": "spatial", "question": "What objects are to the RIGHT of red car?", "answer": "gold car (dummy)"},
                    ]
                }
            return {
                "qas": [
                    {
                        "type": "count",
                        "question": "How many graspable objects are visible?",
                        "answer": "0 (dummy backend)",
                    },
                ]
            }

        # Simple heuristic: pretend to find a switch in the middle
        if n > 8:
            transitions = [n // 2]
            instructions = ["First task", "Second task"]
        else:
            transitions = []
            instructions = ["Single task"]

        return {
            "thought": f"Dummy analysis of {n} frames. No actual inference performed.",
            "transitions": transitions,
            "instructions": instructions,
        }