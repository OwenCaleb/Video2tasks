"""VQA Server application for job queue management."""

import os
import json
import time
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from ..config import Config


class VQASubmitModel(BaseModel):
    """Model for VQA job result submission."""
    task_id: str
    vlm_output: str = ""
    vlm_json: Dict[str, Any] = Field(default_factory=dict)
    latency_s: float = 0.0
    meta: Dict[str, Any] = Field(default_factory=dict)


class VQADatasetCtx:
    """VQA dataset context."""
    def __init__(
        self,
        data_root: str,
        subset: str,
        frames_dir: str,
        output_dir: str,
        sample_id: str
    ):
        self.data_root = data_root
        self.subset = subset
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.sample_id = sample_id
        self.frame_files: List[str] = []


def parse_vqa_datasets(config: Config) -> List[VQADatasetCtx]:
    """Parse VQA dataset configurations.
    
    For VQA mode, each dataset subset is treated as containing frame images directly.
    """
    ctxs = []
    vqa_config = getattr(config, 'vqa', None)
    
    for ds in config.datasets:
        data_dir = Path(ds.root) / ds.subset
        run_dir = Path(config.run.base_dir) / ds.subset / config.run.run_id / "vqa"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all sample directories or treat as single sample
        if data_dir.exists():
            subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
            if subdirs:
                # Has subdirectories - each is a sample
                for subdir in sorted(subdirs):
                    sample_id = subdir.name
                    sample_output = run_dir / sample_id
                    sample_output.mkdir(parents=True, exist_ok=True)
                    
                    ctx = VQADatasetCtx(
                        data_root=ds.root,
                        subset=ds.subset,
                        frames_dir=str(subdir),
                        output_dir=str(sample_output),
                        sample_id=sample_id
                    )
                    ctx.frame_files = _discover_frames(str(subdir))
                    ctxs.append(ctx)
            else:
                # No subdirectories - treat whole dir as single sample
                sample_id = ds.subset
                ctx = VQADatasetCtx(
                    data_root=ds.root,
                    subset=ds.subset,
                    frames_dir=str(data_dir),
                    output_dir=str(run_dir),
                    sample_id=sample_id
                )
                ctx.frame_files = _discover_frames(str(data_dir))
                ctxs.append(ctx)
    
    return ctxs


def _discover_frames(frames_dir: str) -> List[str]:
    """Discover frame image files in a directory."""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    frames_path = Path(frames_dir)
    frame_files = []
    for f in frames_path.iterdir():
        if f.is_file() and f.suffix.lower() in exts:
            frame_files.append(str(f))
    return sorted(frame_files)


def _encode_image_b64(image_path: str) -> str:
    """Encode image to base64."""
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _vqa_results_path(output_dir: str) -> str:
    """Get path to VQA results file."""
    return str(Path(output_dir) / "vqa_results.jsonl")


def _done_marker_path(output_dir: str) -> str:
    """Get path to done marker."""
    return str(Path(output_dir) / ".DONE")


def _load_completed_frames(output_dir: str) -> set:
    """Load set of completed frame IDs."""
    results_path = _vqa_results_path(output_dir)
    completed = set()
    if Path(results_path).exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "frame_id" in data:
                        completed.add(data["frame_id"])
                except json.JSONDecodeError:
                    continue
    return completed


def create_vqa_app(config: Config) -> FastAPI:
    """Create VQA FastAPI application."""
    app = FastAPI(title="Video2Tasks VQA Server")
    
    # Get VQA config
    vqa_cfg = getattr(config, 'vqa', None)
    question_types = vqa_cfg.question_types if vqa_cfg else ["spatial", "attribute", "existence", "count", "manipulation"]
    context_frames = vqa_cfg.context_frames if vqa_cfg else 0
    
    # Initialize dataset contexts
    dataset_ctxs = parse_vqa_datasets(config)
    
    # Thread-safe job management
    queue_lock = threading.Lock()
    job_queue: List[Dict[str, Any]] = []
    inflight: Dict[str, Dict[str, Any]] = {}
    retry_counts: Dict[str, int] = {}
    
    # Per-sample locks for thread-safe writes
    _sample_locks: Dict[str, threading.Lock] = {}
    _sample_locks_lock = threading.Lock()
    
    def get_sample_lock(sample_key: str) -> threading.Lock:
        with _sample_locks_lock:
            if sample_key not in _sample_locks:
                _sample_locks[sample_key] = threading.Lock()
            return _sample_locks[sample_key]
    
    @app.get("/get_job")
    def get_job() -> Dict[str, Any]:
        with queue_lock:
            if not job_queue:
                return {"status": "empty"}
            job = job_queue.pop(0)
            inflight[job["task_id"]] = {"ts": time.time(), "job": job}
            return {"status": "ok", "data": job}
    
    @app.post("/submit_result")
    def submit_result(res: VQASubmitModel) -> Dict[str, str]:
        tid = res.task_id
        job_info = None
        
        with queue_lock:
            if tid in inflight:
                job_info = inflight.pop(tid)
        
        # Empty result: trigger retry
        if not res.vlm_json:
            if job_info:
                with queue_lock:
                    retry_counts[tid] = retry_counts.get(tid, 0) + 1
                    if retry_counts[tid] <= config.server.max_retries_per_job:
                        job_queue.insert(0, job_info["job"])
                        print(f"[VQA] Task {tid} empty, re-queueing (attempt {retry_counts[tid]})")
                    else:
                        print(f"[VQA] Task {tid} failed max retries, dropping")
            return {"status": "retry_triggered"}
        
        # Write result
        meta = res.meta
        sample_id = str(meta.get("sample_id", "unknown"))
        output_dir = str(meta.get("output_dir", ""))
        frame_id = str(meta.get("frame_id", "unknown"))
        
        if not output_dir:
            # Find matching context
            for ctx in dataset_ctxs:
                if ctx.sample_id == sample_id:
                    output_dir = ctx.output_dir
                    break
        
        if output_dir:
            sample_key = f"{meta.get('subset', '')}::{sample_id}"
            with get_sample_lock(sample_key):
                results_path = _vqa_results_path(output_dir)
                record = {
                    "frame_id": frame_id,
                    "frame_idx": meta.get("frame_idx"),
                    "qas": res.vlm_json.get("qas", []),
                }
                with open(results_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        return {"status": "received"}
    
    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "mode": "vqa"}
    
    @app.get("/stats")
    def stats() -> Dict[str, Any]:
        with queue_lock:
            return {
                "queue_size": len(job_queue),
                "inflight": len(inflight),
                "datasets": len(dataset_ctxs),
                "mode": "vqa"
            }
    
    # Producer loop
    def producer_loop():
        total_frames = sum(len(ctx.frame_files) for ctx in dataset_ctxs)
        
        done = 0
        for ctx in dataset_ctxs:
            if Path(_done_marker_path(ctx.output_dir)).exists():
                done += len(ctx.frame_files)
            else:
                done += len(_load_completed_frames(ctx.output_dir))
        
        print(
            f"[VQA Server] Started. Mode=VQA\n"
            f"[Plan] Datasets={[(c.subset, c.sample_id, len(c.frame_files)) for c in dataset_ctxs]}\n"
            f"[Resume] Already done: {done}/{total_frames}"
        )
        
        # State tracking per dataset
        states = {}
        for i, ctx in enumerate(dataset_ctxs):
            states[i] = {
                "cur_idx": 0,
                "completed": _load_completed_frames(ctx.output_dir),
            }
        
        dataset_idx = 0
        global_done = done
        
        while True:
            # Check inflight timeouts
            now = time.time()
            with queue_lock:
                expired = [
                    tid for tid, info in inflight.items()
                    if now - info["ts"] > config.server.inflight_timeout_sec
                ]
                for tid in expired:
                    job = inflight.pop(tid)["job"]
                    retry_counts[tid] = retry_counts.get(tid, 0) + 1
                    if retry_counts[tid] <= config.server.max_retries_per_job:
                        job_queue.append(job)
            
            # All datasets done
            if dataset_idx >= len(dataset_ctxs):
                if config.server.auto_exit_after_all_done:
                    print(f"[VQA All Done] {global_done}/{total_frames}. Exiting.")
                    os._exit(0)
                time.sleep(1.0)
                continue
            
            ctx = dataset_ctxs[dataset_idx]
            st = states[dataset_idx]
            cur_idx = st["cur_idx"]
            completed = st["completed"]
            frame_files = ctx.frame_files
            
            # Current dataset done
            if cur_idx >= len(frame_files):
                with queue_lock:
                    if not job_queue and not inflight:
                        # Mark done
                        Path(_done_marker_path(ctx.output_dir)).touch()
                        print(f"[VQA] Completed {ctx.subset}/{ctx.sample_id}. Switching...")
                        dataset_idx += 1
                time.sleep(0.2)
                continue
            
            # Produce jobs if queue not full
            with queue_lock:
                q_len = len(job_queue)
            
            if q_len < config.server.max_queue:
                frame_path = frame_files[cur_idx]
                frame_id = Path(frame_path).stem
                
                # Skip if already done
                if frame_id in completed:
                    st["cur_idx"] += 1
                    continue
                
                task_id = f"vqa::{ctx.subset}::{ctx.sample_id}::{frame_id}"
                
                # Check if already in queue/inflight
                with queue_lock:
                    if any(j["task_id"] == task_id for j in job_queue) or task_id in inflight:
                        st["cur_idx"] += 1
                        continue
                
                # Build context frames
                images_b64 = []
                context_ids = []
                
                # Add context frames before
                if context_frames > 0:
                    start_idx = max(0, cur_idx - context_frames)
                    for j in range(start_idx, cur_idx):
                        images_b64.append(_encode_image_b64(frame_files[j]))
                        context_ids.append(Path(frame_files[j]).stem)
                
                # Add center frame
                images_b64.append(_encode_image_b64(frame_path))
                
                # Parse frame index
                import re
                match = re.search(r'(\d+)', frame_id)
                frame_idx = int(match.group(1)) if match else cur_idx
                
                job = {
                    "task_id": task_id,
                    "images": images_b64,
                    "question_types": question_types,
                    "meta": {
                        "subset": ctx.subset,
                        "sample_id": ctx.sample_id,
                        "frame_id": frame_id,
                        "frame_idx": frame_idx,
                        "output_dir": ctx.output_dir,
                        "context_frame_ids": context_ids,
                    }
                }
                
                with queue_lock:
                    job_queue.append(job)
                
                st["cur_idx"] += 1
                
                # Batch add more jobs
                added = 1
                while added < 10 and st["cur_idx"] < len(frame_files):
                    with queue_lock:
                        if len(job_queue) >= config.server.max_queue:
                            break
                    
                    next_idx = st["cur_idx"]
                    next_path = frame_files[next_idx]
                    next_frame_id = Path(next_path).stem
                    
                    if next_frame_id in completed:
                        st["cur_idx"] += 1
                        continue
                    
                    next_task_id = f"vqa::{ctx.subset}::{ctx.sample_id}::{next_frame_id}"
                    
                    with queue_lock:
                        if any(j["task_id"] == next_task_id for j in job_queue) or next_task_id in inflight:
                            st["cur_idx"] += 1
                            continue
                    
                    # Build images for this frame
                    next_images = []
                    next_context_ids = []
                    if context_frames > 0:
                        start = max(0, next_idx - context_frames)
                        for j in range(start, next_idx):
                            next_images.append(_encode_image_b64(frame_files[j]))
                            next_context_ids.append(Path(frame_files[j]).stem)
                    next_images.append(_encode_image_b64(next_path))
                    
                    match = re.search(r'(\d+)', next_frame_id)
                    next_frame_idx = int(match.group(1)) if match else next_idx
                    
                    next_job = {
                        "task_id": next_task_id,
                        "images": next_images,
                        "question_types": question_types,
                        "meta": {
                            "subset": ctx.subset,
                            "sample_id": ctx.sample_id,
                            "frame_id": next_frame_id,
                            "frame_idx": next_frame_idx,
                            "output_dir": ctx.output_dir,
                            "context_frame_ids": next_context_ids,
                        }
                    }
                    
                    with queue_lock:
                        job_queue.append(next_job)
                    
                    st["cur_idx"] += 1
                    added += 1
            
            time.sleep(0.05)
    
    # Start producer thread
    producer_thread = threading.Thread(target=producer_loop, daemon=True)
    producer_thread.start()
    
    return app


def run_vqa_server(config: Config) -> None:
    """Run the VQA server."""
    app = create_vqa_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower()
    )
