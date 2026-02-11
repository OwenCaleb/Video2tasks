"""CoT Server application — reads segment outputs and dispatches CoT jobs."""

import os
import json
import time
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from ..config import Config
from ..server.windowing import (
    read_video_info,
    FrameExtractor,
    encode_image_720p_png,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CoTSubmitModel(BaseModel):
    """Model for CoT job result submission."""
    task_id: str
    vlm_output: str = ""
    vlm_json: Dict[str, Any] = Field(default_factory=dict)
    latency_s: float = 0.0
    meta: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segments_json_path(segment_samples_dir: str, sample_id: str) -> Path:
    """Path to the segment stage ``segments.json`` for a sample."""
    return Path(segment_samples_dir) / sample_id / "segments.json"


def _cot_output_path(cot_run_dir: str, sample_id: str) -> Path:
    return Path(cot_run_dir) / sample_id / "cot_results.json"


def _done_marker_path(cot_run_dir: str, sample_id: str) -> Path:
    return Path(cot_run_dir) / sample_id / ".DONE"


def _load_segments(segment_samples_dir: str, sample_id: str) -> Optional[Dict]:
    """Load segment outputs for a sample. Returns None if not found."""
    p = _segments_json_path(segment_samples_dir, sample_id)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _sample_frame_ids(start_frame: int, end_frame: int, n: int) -> List[int]:
    """Uniformly sample *n* frame indices from [start_frame, end_frame)."""
    if end_frame <= start_frame:
        return [start_frame]
    ids = np.linspace(start_frame, end_frame - 1, num=n).astype(int)
    return ids.tolist()


def _load_completed_seg_ids(cot_run_dir: str, sample_id: str) -> set:
    """Load set of seg_ids already completed from cot_results.json."""
    p = _cot_output_path(cot_run_dir, sample_id)
    completed: set = set()
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for rec in data.get("segments", []):
                if "seg_id" in rec and rec.get("cot"):
                    completed.add(rec["seg_id"])
        except (json.JSONDecodeError, KeyError):
            pass
    return completed


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_cot_app(config: Config) -> FastAPI:
    app = FastAPI(title="Video2Tasks CoT Server")

    cot_cfg = config.cot
    frames_per_seg = cot_cfg.frames_per_segment
    target_w = cot_cfg.target_width
    target_h = cot_cfg.target_height
    high_level_instruction = cot_cfg.high_level_instruction

    # Segment run directory (where segment outputs live)
    # segment outputs: {base_dir}/{subset}/{segment_run_id}/samples/{sample_id}/segments.json
    seg_run_id = cot_cfg.segment_run_id

    # CoT run directory
    # cot outputs: {base_dir}/{subset}/{run_id}/cot/{sample_id}/cot_results.json
    cot_run_id = config.run.run_id

    # Discover datasets & samples
    dataset_metas: List[Dict[str, Any]] = []
    for ds in config.datasets:
        data_dir = Path(ds.root) / ds.subset
        seg_samples_dir = str(
            Path(config.run.base_dir) / ds.subset / seg_run_id / "samples"
        )
        cot_run_dir = str(
            Path(config.run.base_dir) / ds.subset / cot_run_id / "cot"
        )
        Path(cot_run_dir).mkdir(parents=True, exist_ok=True)

        # Enumerate samples present in segment outputs
        seg_samples_path = Path(seg_samples_dir)
        if seg_samples_path.exists():
            sample_ids = sorted(
                p.name
                for p in seg_samples_path.iterdir()
                if p.is_dir() and (p / "segments.json").exists()
            )
        else:
            sample_ids = []

        dataset_metas.append({
            "subset": ds.subset,
            "data_dir": str(data_dir),
            "seg_samples_dir": seg_samples_dir,
            "cot_run_dir": cot_run_dir,
            "sample_ids": sample_ids,
        })

    # Thread-safe job management
    queue_lock = threading.Lock()
    job_queue: List[Dict[str, Any]] = []
    inflight: Dict[str, Dict[str, Any]] = {}
    retry_counts: Dict[str, int] = {}

    # Per-sample locks
    _sample_locks: Dict[str, threading.Lock] = {}
    _sample_locks_lock = threading.Lock()

    def get_sample_lock(key: str) -> threading.Lock:
        with _sample_locks_lock:
            if key not in _sample_locks:
                _sample_locks[key] = threading.Lock()
            return _sample_locks[key]

    # ---- endpoints ----

    @app.get("/get_job")
    def get_job() -> Dict[str, Any]:
        with queue_lock:
            if not job_queue:
                return {"status": "empty"}
            job = job_queue.pop(0)
            inflight[job["task_id"]] = {"ts": time.time(), "job": job}
            return {"status": "ok", "data": job}

    @app.post("/submit_result")
    def submit_result(res: CoTSubmitModel) -> Dict[str, str]:
        tid = res.task_id
        job_info = None
        with queue_lock:
            if tid in inflight:
                job_info = inflight.pop(tid)

        # Empty → retry
        if not res.vlm_json or not res.vlm_json.get("cot"):
            if job_info:
                with queue_lock:
                    retry_counts[tid] = retry_counts.get(tid, 0) + 1
                    if retry_counts[tid] <= config.server.max_retries_per_job:
                        job_queue.insert(0, job_info["job"])
                        print(f"[CoT] Task {tid} empty, re-queueing (attempt {retry_counts[tid]})")
                    else:
                        print(f"[CoT] Task {tid} failed max retries, dropping")
            return {"status": "retry_triggered"}

        meta = res.meta
        cot_run_dir = str(meta.get("cot_run_dir", ""))
        sample_id = str(meta.get("sample_id", "unknown"))
        seg_id = meta.get("seg_id", -1)
        subtask = str(meta.get("subtask", ""))

        if cot_run_dir:
            sample_key = f"{meta.get('subset', '')}::{sample_id}"
            with get_sample_lock(sample_key):
                out_path = _cot_output_path(cot_run_dir, sample_id)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Load existing
                existing: Dict[str, Any] = {"sample_id": sample_id, "segments": []}
                if out_path.exists():
                    try:
                        existing = json.loads(out_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass

                # Append / update
                segments = existing.get("segments", [])
                # Remove old entry for same seg_id if exists
                segments = [s for s in segments if s.get("seg_id") != seg_id]
                segments.append({
                    "seg_id": seg_id,
                    "instruction": res.vlm_json.get("instruction", ""),
                    "cot": res.vlm_json.get("cot", ""),
                    "start_frame": meta.get("start_frame"),
                    "end_frame": meta.get("end_frame"),
                })
                segments.sort(key=lambda s: s.get("seg_id", 0))
                existing["segments"] = segments

                out_path.write_text(
                    json.dumps(existing, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        return {"status": "received"}

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "mode": "cot"}

    @app.get("/stats")
    def stats() -> Dict[str, Any]:
        with queue_lock:
            return {
                "queue_size": len(job_queue),
                "inflight": len(inflight),
                "datasets": len(dataset_metas),
                "mode": "cot",
            }

    # ---- producer loop ----

    def producer_loop() -> None:
        total_segments = 0
        done_segments = 0

        for dm in dataset_metas:
            for sid in dm["sample_ids"]:
                seg_data = _load_segments(dm["seg_samples_dir"], sid)
                if seg_data:
                    segs = seg_data.get("segments", [])
                    total_segments += len(segs)
                    if _done_marker_path(dm["cot_run_dir"], sid).exists():
                        done_segments += len(segs)
                    else:
                        done_segments += len(
                            _load_completed_seg_ids(dm["cot_run_dir"], sid)
                        )

        print(
            f"[CoT Server] Started. Mode=CoT\n"
            f"[Plan] Datasets={[(dm['subset'], len(dm['sample_ids'])) for dm in dataset_metas]}\n"
            f"[Resume] Already done: {done_segments}/{total_segments} segments"
        )

        # Build flat work list: (dm_index, sample_id)
        work: List[tuple] = []
        for di, dm in enumerate(dataset_metas):
            for sid in dm["sample_ids"]:
                work.append((di, sid))

        work_idx = 0
        global_done = done_segments

        while True:
            # Inflight timeouts
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

            # All done
            if work_idx >= len(work):
                if config.server.auto_exit_after_all_done:
                    with queue_lock:
                        if not job_queue and not inflight:
                            print(f"[CoT All Done] {global_done}/{total_segments}. Exiting.")
                            os._exit(0)
                time.sleep(1.0)
                continue

            with queue_lock:
                q_len = len(job_queue)

            if q_len >= config.server.max_queue:
                time.sleep(0.1)
                continue

            di, sid = work[work_idx]
            dm = dataset_metas[di]

            # Skip if sample already done
            if _done_marker_path(dm["cot_run_dir"], sid).exists():
                work_idx += 1
                continue

            seg_data = _load_segments(dm["seg_samples_dir"], sid)
            if not seg_data:
                work_idx += 1
                continue

            segments = seg_data.get("segments", [])
            nframes = seg_data.get("nframes", 0)

            # Find video file
            s_dir = Path(dm["data_dir"]) / sid
            mp4s = list(s_dir.glob("Frame_*.mp4"))
            if not mp4s:
                print(f"[CoT] No video for {dm['subset']}/{sid}, skipping")
                work_idx += 1
                continue

            mp4 = str(mp4s[0])
            completed = _load_completed_seg_ids(dm["cot_run_dir"], sid)

            try:
                with FrameExtractor(mp4) as extractor:
                    all_done = True
                    for seg in segments:
                        seg_id = seg["seg_id"]
                        if seg_id in completed:
                            continue

                        all_done = False
                        tid = f"cot::{dm['subset']}::{sid}::seg{seg_id}"

                        with queue_lock:
                            if any(j["task_id"] == tid for j in job_queue) or tid in inflight:
                                continue

                        start_f = seg["start_frame"]
                        end_f = seg["end_frame"]
                        instruction = seg.get("instruction", "unknown")

                        frame_ids = _sample_frame_ids(start_f, end_f, frames_per_seg)
                        images_b64 = extractor.get_many_b64(
                            frame_ids, target_w, target_h, compression=1
                        )

                        job = {
                            "task_id": tid,
                            "images": images_b64,
                            "subtask": instruction,
                            "high_level_instruction": high_level_instruction,
                            "meta": {
                                "subset": dm["subset"],
                                "sample_id": sid,
                                "seg_id": seg_id,
                                "subtask": instruction,
                                "start_frame": start_f,
                                "end_frame": end_f,
                                "cot_run_dir": dm["cot_run_dir"],
                            },
                        }

                        with queue_lock:
                            job_queue.append(job)

                    if all_done:
                        # Mark sample done
                        marker = _done_marker_path(dm["cot_run_dir"], sid)
                        marker.parent.mkdir(parents=True, exist_ok=True)
                        marker.touch()
                        print(f"[CoT] Completed {dm['subset']}/{sid}")

            except Exception as e:
                print(f"[CoT Err] {dm['subset']}/{sid}: {e}")
                import traceback
                traceback.print_exc()

            work_idx += 1
            time.sleep(0.05)

    producer_thread = threading.Thread(target=producer_loop, daemon=True)
    producer_thread.start()

    return app


def run_cot_server(config: Config) -> None:
    """Run the CoT server with given configuration."""
    app = create_cot_app(config)
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower(),
    )
