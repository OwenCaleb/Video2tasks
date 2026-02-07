"""VQA output writer for JSONL format."""

import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Set

from .types import VQAResult


class VQAWriter:
    """Thread-safe JSONL writer for VQA results."""
    
    def __init__(self, output_dir: str, filename: str = "vqa_results.jsonl"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / filename
        self._lock = threading.Lock()
        self._completed_frames: Set[str] = set()
        self._load_completed()
    
    def _load_completed(self) -> None:
        """Load set of completed frame IDs from existing output file."""
        if self.output_path.exists():
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        frame_id = data.get("frame_id")
                        if frame_id:
                            self._completed_frames.add(frame_id)
                    except json.JSONDecodeError:
                        continue
    
    def get_completed_frames(self) -> Set[str]:
        """Get set of already completed frame IDs."""
        with self._lock:
            return self._completed_frames.copy()
    
    def is_completed(self, frame_id: str) -> bool:
        """Check if a frame has already been processed."""
        with self._lock:
            return frame_id in self._completed_frames
    
    def write(self, result: VQAResult) -> None:
        """Write a VQA result to the output file."""
        with self._lock:
            # Skip if already completed
            if result.frame_id in self._completed_frames:
                return
            
            # Convert to dict and write
            data = result.model_dump(exclude_none=True)
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            self._completed_frames.add(result.frame_id)
    
    def write_batch(self, results: List[VQAResult]) -> int:
        """Write multiple VQA results. Returns number of new results written."""
        count = 0
        with self._lock:
            with open(self.output_path, "a", encoding="utf-8") as f:
                for result in results:
                    if result.frame_id in self._completed_frames:
                        continue
                    data = result.model_dump(exclude_none=True)
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    self._completed_frames.add(result.frame_id)
                    count += 1
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the output."""
        with self._lock:
            return {
                "output_path": str(self.output_path),
                "completed_count": len(self._completed_frames),
            }
    
    def mark_done(self) -> None:
        """Create a .DONE marker file."""
        done_path = self.output_dir / ".DONE"
        done_path.touch()
    
    def is_done(self) -> bool:
        """Check if .DONE marker exists."""
        return (self.output_dir / ".DONE").exists()


class VQAMultiSampleWriter:
    """Manages VQA writers for multiple samples."""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self._writers: Dict[str, VQAWriter] = {}
        self._lock = threading.Lock()
    
    def get_writer(self, sample_id: str) -> VQAWriter:
        """Get or create a writer for a sample."""
        with self._lock:
            if sample_id not in self._writers:
                sample_dir = self.base_output_dir / sample_id
                self._writers[sample_id] = VQAWriter(str(sample_dir))
            return self._writers[sample_id]
    
    def write(self, sample_id: str, result: VQAResult) -> None:
        """Write a result for a specific sample."""
        writer = self.get_writer(sample_id)
        writer.write(result)
    
    def get_completed_frames(self, sample_id: str) -> Set[str]:
        """Get completed frames for a sample."""
        writer = self.get_writer(sample_id)
        return writer.get_completed_frames()
    
    def is_sample_done(self, sample_id: str) -> bool:
        """Check if a sample is marked as done."""
        writer = self.get_writer(sample_id)
        return writer.is_done()
    
    def mark_sample_done(self, sample_id: str) -> None:
        """Mark a sample as done."""
        writer = self.get_writer(sample_id)
        writer.mark_done()
