"""VQA job builder for enumerating frame samples."""

import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass

from .types import VQAJobData, VQADatasetCtx


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class VQAJobBuilderConfig:
    """Configuration for VQA job builder."""
    question_types: List[str]
    batch_size: int = 1  # Number of frames per job (usually 1 for single-frame VQA)


class VQAJobBuilder:
    """Builds VQA jobs from frame directories."""
    
    def __init__(self, config: VQAJobBuilderConfig):
        self.config = config
    
    def discover_frames(self, frames_dir: str) -> List[str]:
        """Discover all frame files in a directory."""
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            return []
        
        frame_files = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            frame_files.extend(frames_path.glob(f"*{ext}"))
            frame_files.extend(frames_path.glob(f"*{ext.upper()}"))
        
        # Sort by name for consistent ordering
        frame_files = sorted(frame_files, key=lambda p: p.name)
        return [str(f) for f in frame_files]
    
    def parse_frame_index(self, frame_path: str) -> Optional[int]:
        """Try to extract frame index from filename."""
        name = Path(frame_path).stem
        # Common patterns: frame_000001, 000001, img_001, etc.
        import re
        match = re.search(r'(\d+)', name)
        if match:
            return int(match.group(1))
        return None
    
    def encode_image_b64(self, image_path: str) -> str:
        """Encode an image file to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def build_job(
        self,
        frame_path: str,
        subset: str,
        sample_id: str,
        context_frame_paths: Optional[List[str]] = None
    ) -> VQAJobData:
        """Build a single VQA job for a frame."""
        frame_id = Path(frame_path).stem
        frame_idx = self.parse_frame_index(frame_path)
        
        # Build task ID
        task_id = f"vqa::{subset}::{sample_id}::{frame_id}"
        
        # Encode images
        images_b64 = []
        context_frame_ids = []
        
        # Add context frames before center (if any)
        if context_frame_paths:
            for ctx_path in context_frame_paths:
                images_b64.append(self.encode_image_b64(ctx_path))
                context_frame_ids.append(Path(ctx_path).stem)
        
        # Add center frame
        images_b64.append(self.encode_image_b64(frame_path))
        
        return VQAJobData(
            task_id=task_id,
            frame_id=frame_id,
            frame_path=frame_path,
            images_b64=images_b64,
            context_frame_ids=context_frame_ids,
            question_types=self.config.question_types,
            meta={
                "subset": subset,
                "sample_id": sample_id,
                "frame_idx": frame_idx,
                "n_context_frames": len(context_frame_ids),
            }
        )
    
    def build_jobs_for_sample(
        self,
        frames_dir: str,
        subset: str,
        sample_id: str,
        completed_frame_ids: Optional[set] = None
    ) -> Iterator[VQAJobData]:
        """Build VQA jobs for all frames in a sample directory."""
        frame_paths = self.discover_frames(frames_dir)
        if not frame_paths:
            return
        
        completed = completed_frame_ids or set()
        
        for i, frame_path in enumerate(frame_paths):
            frame_id = Path(frame_path).stem
            
            # Skip already completed frames
            if frame_id in completed:
                continue
            
            yield self.build_job(
                frame_path=frame_path,
                subset=subset,
                sample_id=sample_id,
            )
    
    def build_dataset_context(
        self,
        data_root: str,
        subset: str,
        output_base: str,
        run_id: str
    ) -> VQADatasetCtx:
        """Build a VQA dataset context."""
        frames_dir = str(Path(data_root) / subset)
        output_dir = str(Path(output_base) / subset / run_id / "vqa")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        frame_files = self.discover_frames(frames_dir)
        
        return VQADatasetCtx(
            data_root=data_root,
            subset=subset,
            frames_dir=frames_dir,
            output_dir=output_dir,
            frame_files=frame_files
        )
