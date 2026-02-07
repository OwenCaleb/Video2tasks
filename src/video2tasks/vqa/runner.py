"""VQA runner for processing VQA jobs."""

import json
from typing import Dict, Any, List, Optional

from .types import VQAResult, VQAQuestion
from .prompts import VQAPromptRegistry, get_default_prompts


def parse_vqa_response(raw_output: str) -> List[VQAQuestion]:
    """Parse VQA response from model output.
    
    Handles various output formats:
    - Direct JSON with 'qas' list
    - JSON wrapped in markdown code blocks
    - Fallback to empty list on parse failure
    """
    if not raw_output:
        return []
    
    # Clean up common formatting issues
    text = raw_output.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    
    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            qas_data = data.get("qas", [])
            return [VQAQuestion(**qa) for qa in qas_data if isinstance(qa, dict)]
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                qas_data = data.get("qas", [])
                return [VQAQuestion(**qa) for qa in qas_data if isinstance(qa, dict)]
        except (json.JSONDecodeError, TypeError):
            pass
    
    return []


class VQARunner:
    """Runs VQA inference using a VLM backend."""
    
    def __init__(
        self,
        backend: Any,  # VLMBackend
        prompt_registry: Optional[VQAPromptRegistry] = None,
        default_question_types: Optional[List[str]] = None
    ):
        self.backend = backend
        self.prompt_registry = prompt_registry or get_default_prompts()
        self.default_question_types = default_question_types or self.prompt_registry.list_types()
    
    def build_prompt(self, question_types: Optional[List[str]] = None, n_images: int = 1) -> str:
        """Build prompt for VQA inference."""
        types = question_types or self.default_question_types
        return self.prompt_registry.build_combined_prompt(types, n_images)
    
    def run(
        self,
        images: List[Any],  # List of numpy arrays
        frame_id: str,
        frame_idx: Optional[int] = None,
        question_types: Optional[List[str]] = None,
    ) -> VQAResult:
        """Run VQA inference on images.
        
        Args:
            images: List of images (numpy arrays in BGR format)
            frame_id: Frame identifier
            frame_idx: Optional frame index
            question_types: Question types to ask (uses default if None)
            
        Returns:
            VQAResult with QA pairs
        """
        prompt = self.build_prompt(question_types, len(images))
        
        try:
            # Call backend - it may return dict with various formats
            output = self.backend.infer(images, prompt)
            
            # Handle different output formats from backends
            raw_output = ""
            qas: List[VQAQuestion] = []
            
            if isinstance(output, dict):
                # Check for direct qas in output
                if "qas" in output:
                    qas_data = output["qas"]
                    if isinstance(qas_data, list):
                        qas = [VQAQuestion(**qa) for qa in qas_data if isinstance(qa, dict)]
                
                # Check for text field that needs parsing
                elif "text" in output:
                    raw_output = str(output.get("text", ""))
                    qas = parse_vqa_response(raw_output)
                
                # Check for raw_output field
                elif "raw_output" in output:
                    raw_output = str(output.get("raw_output", ""))
                    qas = parse_vqa_response(raw_output)
                
                # Fallback: try to parse the whole output as containing qas
                else:
                    raw_output = json.dumps(output)
                    # If it has thought/transitions (segment format), skip
                    if "transitions" not in output:
                        qas = parse_vqa_response(raw_output)
            
            elif isinstance(output, str):
                raw_output = output
                qas = parse_vqa_response(output)
        
        except Exception as e:
            print(f"[VQA] Inference error for {frame_id}: {e}")
            raw_output = str(e)
            qas = []
        
        return VQAResult(
            frame_id=frame_id,
            frame_idx=frame_idx,
            qas=qas,
            raw_model_output=raw_output if raw_output else None
        )
