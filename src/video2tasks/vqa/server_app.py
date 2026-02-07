"""VQA Server application for job queue management."""

import os
import json
import time
import threading
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from ..config import Config, VQAConfig


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
    """Parse VQA dataset configurations."""
    ctxs: List[VQADatasetCtx] = []
    for ds in config.datasets:
        data_dir = Path(ds.root) / ds.subset
        run_dir = Path(config.run.base_dir) / ds.subset / config.run.run_id / "vqa"
        run_dir.mkdir(parents=True, exist_ok=True)

        if data_dir.exists():
            subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
            if subdirs:
                for subdir in sorted(subdirs):
                    sample_id = subdir.name
                    sample_output = run_dir / sample_id
                    sample_output.mkdir(parents=True, exist_ok=True)
                    ctx = VQADatasetCtx(
                        data_root=ds.root,
                        subset=ds.subset,
                        frames_dir=str(subdir),
                        output_dir=str(sample_output),
                        sample_id=sample_id,
                    )
                    ctx.frame_files = _discover_frames(str(subdir))
                    ctxs.append(ctx)
            else:
                sample_id = ds.subset
                ctx = VQADatasetCtx(
                    data_root=ds.root,
                    subset=ds.subset,
                    frames_dir=str(data_dir),
                    output_dir=str(run_dir),
                    sample_id=sample_id,
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


def _encode_image_b64(
    image_path: str,
    target_w: int = 0,
    target_h: int = 0,
) -> str:
    """Encode image to base64, optionally resizing first."""
    import base64 as _b64
    from PIL import Image as _Img
    from io import BytesIO as _Bio

    if target_w > 0 and target_h > 0:
        img = _Img.open(image_path)
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), _Img.LANCZOS)
        buf = _Bio()
        fmt = "JPEG" if image_path.lower().endswith((".jpg", ".jpeg")) else "PNG"
        img.save(buf, format=fmt)
        return _b64.b64encode(buf.getvalue()).decode("utf-8")

    with open(image_path, "rb") as f:
        return _b64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Per-type output helpers
# ---------------------------------------------------------------------------

def _type_jsonl_path(output_dir: str, qtype: str) -> str:
    """Per-type JSONL path: ``{output_dir}/{qtype}.jsonl``."""
    return str(Path(output_dir) / f"{qtype}.jsonl")


def _done_marker_path(output_dir: str) -> str:
    return str(Path(output_dir) / ".DONE")


def _load_completed_frames(
    output_dir: str, question_types: List[str]
) -> Dict[str, Set[str]]:
    """Load per-type sets of completed frame IDs from JSONL files."""
    completed: Dict[str, Set[str]] = {qt: set() for qt in question_types}
    for qt in question_types:
        path = Path(_type_jsonl_path(output_dir, qt))
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        fid = data.get("frame_id")
                        if fid is not None:
                            completed[qt].add(str(fid))
                    except json.JSONDecodeError:
                        continue
    return completed


def _parse_numeric_frame_id(frame_id: str) -> Optional[int]:
    import re
    match = re.search(r"(\d+)", frame_id)
    return int(match.group(1)) if match else None


def _compute_frame_idx(frame_id: str, fallback_idx: int, sample_hz: float) -> int:
    """frame_idx = numeric_id × sample_hz."""
    numeric_id = _parse_numeric_frame_id(frame_id)
    if numeric_id is None:
        return fallback_idx
    return int(round(numeric_id * sample_hz))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_vqa_app(config: Config) -> FastAPI:
    """Create VQA FastAPI application."""
    app = FastAPI(title="Video2Tasks VQA Server")

    vqa_cfg = getattr(config, "vqa", None)
    question_types: List[str] = (
        vqa_cfg.question_types
        if vqa_cfg
        else ["spatial", "attribute", "existence", "count", "manipulation"]
    )
    sample_hz: float = vqa_cfg.sample_hz if vqa_cfg else 1.0
    target_w: int = vqa_cfg.target_width if vqa_cfg else 0
    target_h: int = vqa_cfg.target_height if vqa_cfg else 0

    dataset_ctxs = parse_vqa_datasets(config)

    # Thread-safe job management
    queue_lock = threading.Lock()
    job_queue: List[Dict[str, Any]] = []
    inflight: Dict[str, Dict[str, Any]] = {}
    retry_counts: Dict[str, int] = {}

    # Per-sample locks for writes
    _sample_locks: Dict[str, threading.Lock] = {}
    _sample_locks_lock = threading.Lock()

    def get_sample_lock(sample_key: str) -> threading.Lock:
        with _sample_locks_lock:
            if sample_key not in _sample_locks:
                _sample_locks[sample_key] = threading.Lock()
            return _sample_locks[sample_key]

    # ----- endpoints -----

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

        by_type: Dict[str, Any] = res.vlm_json.get("by_type", {})

        # Completely empty → retry
        if not by_type:
            if job_info:
                with queue_lock:
                    retry_counts[tid] = retry_counts.get(tid, 0) + 1
                    if retry_counts[tid] <= config.server.max_retries_per_job:
                        job_queue.insert(0, job_info["job"])
                        print(f"[VQA] Task {tid} empty, re-queueing (attempt {retry_counts[tid]})")
                    else:
                        print(f"[VQA] Task {tid} failed max retries, dropping")
            return {"status": "retry_triggered"}

        meta = res.meta
        sample_id = str(meta.get("sample_id", "unknown"))
        output_dir = str(meta.get("output_dir", ""))
        frame_id = str(meta.get("frame_id", "unknown"))
        frame_idx = meta.get("frame_idx", 0)

        if not output_dir:
            for ctx in dataset_ctxs:
                if ctx.sample_id == sample_id:
                    output_dir = ctx.output_dir
                    break

        if output_dir:
            sample_key = f"{meta.get('subset', '')}::{sample_id}"
            with get_sample_lock(sample_key):
                for qtype, type_data in by_type.items():
                    record = {
                        "frame_id": frame_id,
                        "frame_idx": frame_idx,
                        "qas": type_data.get("qas", []) if isinstance(type_data, dict) else [],
                    }
                    p = _type_jsonl_path(output_dir, qtype)
                    with open(p, "a", encoding="utf-8") as f:
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
                "mode": "vqa",
            }

    # ----- producer loop -----

    def producer_loop() -> None:
        total_tasks = sum(len(ctx.frame_files) * len(question_types) for ctx in dataset_ctxs)

        done_tasks = 0
        for ctx in dataset_ctxs:
            if Path(_done_marker_path(ctx.output_dir)).exists():
                done_tasks += len(ctx.frame_files) * len(question_types)
            else:
                comp = _load_completed_frames(ctx.output_dir, question_types)
                for qt in question_types:
                    done_tasks += len(comp[qt])

        print(
            f"[VQA Server] Started. Mode=VQA\n"
            f"[Plan] Datasets={[(c.subset, c.sample_id, len(c.frame_files)) for c in dataset_ctxs]}\n"
            f"[Plan] Types={question_types}  sample_hz={sample_hz}\n"
            f"[Resume] Already done: {done_tasks}/{total_tasks} (frame×type)"
        )

        # State per dataset
        states: Dict[int, Dict[str, Any]] = {}
        for i, ctx in enumerate(dataset_ctxs):
            states[i] = {
                "cur_idx": 0,
                "completed": _load_completed_frames(ctx.output_dir, question_types),
            }

        dataset_idx = 0

        while True:
            # Inflight timeout check
            now = time.time()
            with queue_lock:
                expired = [
                    tid
                    for tid, info in inflight.items()
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
                    print(f"[VQA All Done] Exiting.")
                    os._exit(0)
                time.sleep(1.0)
                continue

            ctx = dataset_ctxs[dataset_idx]
            st = states[dataset_idx]
            cur_idx: int = st["cur_idx"]
            completed: Dict[str, Set[str]] = st["completed"]
            frame_files = ctx.frame_files

            # Current dataset done
            if cur_idx >= len(frame_files):
                with queue_lock:
                    if not job_queue and not inflight:
                        Path(_done_marker_path(ctx.output_dir)).touch()
                        print(f"[VQA] Completed {ctx.subset}/{ctx.sample_id}. Switching...")
                        dataset_idx += 1
                time.sleep(0.2)
                continue

            # Produce jobs
            with queue_lock:
                q_len = len(job_queue)

            if q_len < config.server.max_queue:
                frame_path = frame_files[cur_idx]
                frame_id = Path(frame_path).stem

                # Determine missing types for this frame
                missing_types = [
                    qt for qt in question_types if frame_id not in completed.get(qt, set())
                ]

                if not missing_types:
                    st["cur_idx"] += 1
                    continue

                task_id = f"vqa::{ctx.subset}::{ctx.sample_id}::{frame_id}"

                with queue_lock:
                    if any(j["task_id"] == task_id for j in job_queue) or task_id in inflight:
                        st["cur_idx"] += 1
                        continue

                # Single frame (no context frames)
                images_b64 = [_encode_image_b64(frame_path, target_w, target_h)]

                frame_idx = _compute_frame_idx(frame_id, cur_idx, sample_hz)

                job = {
                    "task_id": task_id,
                    "images": images_b64,
                    "question_types": missing_types,
                    "meta": {
                        "subset": ctx.subset,
                        "sample_id": ctx.sample_id,
                        "frame_id": frame_id,
                        "frame_idx": frame_idx,
                        "output_dir": ctx.output_dir,
                    },
                }

                with queue_lock:
                    job_queue.append(job)

                st["cur_idx"] += 1

                # Batch-add more
                added = 1
                while added < 10 and st["cur_idx"] < len(frame_files):
                    with queue_lock:
                        if len(job_queue) >= config.server.max_queue:
                            break

                    ni = st["cur_idx"]
                    np_ = frame_files[ni]
                    nfid = Path(np_).stem

                    n_missing = [
                        qt for qt in question_types if nfid not in completed.get(qt, set())
                    ]
                    if not n_missing:
                        st["cur_idx"] += 1
                        continue

                    ntid = f"vqa::{ctx.subset}::{ctx.sample_id}::{nfid}"
                    with queue_lock:
                        if any(j["task_id"] == ntid for j in job_queue) or ntid in inflight:
                            st["cur_idx"] += 1
                            continue

                    nfi = _compute_frame_idx(nfid, ni, sample_hz)

                    next_job = {
                        "task_id": ntid,
                        "images": [_encode_image_b64(np_, target_w, target_h)],
                        "question_types": n_missing,
                        "meta": {
                            "subset": ctx.subset,
                            "sample_id": ctx.sample_id,
                            "frame_id": nfid,
                            "frame_idx": nfi,
                            "output_dir": ctx.output_dir,
                        },
                    }

                    with queue_lock:
                        job_queue.append(next_job)

                    st["cur_idx"] += 1
                    added += 1

            time.sleep(0.05)

    producer_thread = threading.Thread(target=producer_loop, daemon=True)
    producer_thread.start()

    return app


def run_vqa_server(config: Config) -> None:
    """Run the VQA server with given configuration."""
    app = create_vqa_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower(),
    )
