"""VQA data types and models."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class VQAQuestion(BaseModel):
    """A single VQA question-answer pair."""
    type: str = Field(..., description="Question type: spatial, attribute, existence, count, manipulation")
    question: str = Field(..., description="The question text")
    answer: str = Field(..., description="The answer text")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    evidence: Optional[str] = Field(None, description="Evidence or reasoning for the answer")


class VQAResult(BaseModel):
    """VQA result for a single frame."""
    frame_id: str = Field(..., description="Frame identifier (filename or index)")
    frame_idx: Optional[int] = Field(None, description="Frame index if applicable")
    timestamp_sec: Optional[float] = Field(None, description="Timestamp in seconds if applicable")
    qas: List[VQAQuestion] = Field(default_factory=list, description="List of QA pairs")
    raw_model_output: Optional[str] = Field(None, description="Raw model output for debugging")


class VQAJobData(BaseModel):
    """Data structure for a VQA job."""
    task_id: str = Field(..., description="Unique task identifier")
    frame_id: str = Field(..., description="Frame identifier")
    frame_path: str = Field(..., description="Path to the frame image")
    images_b64: List[str] = Field(default_factory=list, description="Base64 encoded images")
    context_frame_ids: List[str] = Field(default_factory=list, description="Context frame IDs")
    question_types: List[str] = Field(default_factory=list, description="Question types to ask")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


@dataclass
class VQADatasetCtx:
    """VQA dataset context."""
    data_root: str
    subset: str
    frames_dir: str
    output_dir: str
    frame_files: List[str] = field(default_factory=list)
