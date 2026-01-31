#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import glob
import base64
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import cv2

# =========================
# Config
# =========================
DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/cpfs/wangyinxi/Long-Data")
SUBSET    = os.environ.get("SUBSET", "LongData1-600")

# 新增：多个数据集（支持不同 root）
# 形式1（推荐）：DATASETS="root1:subset1;root2:subset2"
# 例：DATASETS="/mnt/cpfs/wangyinxi/Long-Data:LongData1-600;/mnt/data/guchenyang/Data/Hand-Data/Long-Data:LongData601-1189"
DATASETS_SPEC = os.environ.get("DATASETS", "").strip()
if not DATASETS_SPEC:
    DATASETS_SPEC = (
        "/mnt/cpfs/wangyinxi/Long-Data:LongData601-1189"
    )

RUN_BASE  = os.environ.get("RUN_BASE", "/mnt/cpfs/wangyinxi/projects/vision_cut_runs")
RUN_ID    = os.environ.get("RUN_ID", "latest")
PORT      = int(os.environ.get("PORT", "8099"))

WINDOW_SEC = float(os.environ.get("WINDOW_SEC", "16.0"))
STEP_SEC   = float(os.environ.get("STEP_SEC", "8.0"))
FRAMES_PER_WINDOW = int(os.environ.get("FRAMES_PER_WINDOW", "16"))

TARGET_W = 720
TARGET_H = 480
PNG_COMPRESSION = 0

MAX_QUEUE       = int(os.environ.get("MAX_QUEUE", "32"))
INFLIGHT_TIMEOUT_SEC = float(os.environ.get("INFLIGHT_TIMEOUT_SEC", "300.0"))
MAX_RETRIES_PER_JOB   = int(os.environ.get("MAX_RETRIES_PER_JOB", "5"))

# 可选：强制进度总数显示为 1189（即使目录里少/多）
PROGRESS_TOTAL_OVERRIDE = int(os.environ.get("PROGRESS_TOTAL", "0"))
AUTO_EXIT_AFTER_ALL_DONE = os.environ.get("AUTO_EXIT_AFTER_ALL_DONE", "0").strip() in ("1", "true", "True", "yes", "YES")

app = FastAPI()

# =========================
# Dataset Context
# =========================
@dataclass
class DatasetCtx:
    data_root: str
    subset: str
    data_dir: str
    run_dir: str
    samples_dir: str
    sample_ids: List[str]

def parse_datasets(spec: str) -> List[Tuple[str, str]]:
    """
    返回 [(data_root, subset), ...]
    """
    if spec:
        parts = [p.strip() for p in spec.split(";") if p.strip()]
        out = []
        for p in parts:
            if ":" in p:
                root, subset = p.split(":", 1)
                out.append((root.strip(), subset.strip()))
            else:
                # 允许直接给 data_dir
                data_dir = p.rstrip("/")
                root = os.path.dirname(data_dir)
                subset = os.path.basename(data_dir)
                out.append((root, subset))
        return out
    return [(DATA_ROOT, SUBSET)]

def list_sample_ids(data_dir: str) -> List[str]:
    all_samples = sorted(glob.glob(os.path.join(data_dir, "*")))
    return [os.path.basename(p) for p in all_samples if os.path.isdir(p)]

def make_ctx(root: str, subset: str) -> DatasetCtx:
    data_dir = os.path.join(root, subset)
    run_dir = os.path.join(RUN_BASE, subset, RUN_ID)
    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    sids = list_sample_ids(data_dir) if os.path.isdir(data_dir) else []
    return DatasetCtx(root, subset, data_dir, run_dir, samples_dir, sids)

DATASET_CTXS: List[DatasetCtx] = [make_ctx(r, s) for (r, s) in parse_datasets(DATASETS_SPEC)]

# submit_result 要用 subset 找到对应 samples_dir（防止将来切换/并发写错）
SAMPLES_DIR_BY_SUBSET = {ctx.subset: ctx.samples_dir for ctx in DATASET_CTXS}
DATA_DIR_BY_SUBSET = {ctx.subset: ctx.data_dir for ctx in DATASET_CTXS}

# =========================
# Data Structures
# =========================
@dataclass
class Window:
    window_id: int
    start_frame: int
    end_frame: int
    frame_ids: List[int]

class SubmitModel(BaseModel):
    task_id: str
    vlm_output: str = ""
    vlm_json: Dict[str, Any] = {}
    latency_s: float = 0.0
    meta: Dict[str, Any] = {}

queue_lock = threading.Lock()
job_queue: List[Dict[str, Any]] = []
inflight: Dict[str, Dict[str, Any]] = {}
retry_counts: Dict[str, int] = {}

_sample_locks: Dict[str, threading.Lock] = {}
_sample_locks_lock = threading.Lock()

def get_sample_lock(sample_key: str) -> threading.Lock:
    with _sample_locks_lock:
        if sample_key not in _sample_locks:
            _sample_locks[sample_key] = threading.Lock()
        return _sample_locks[sample_key]

# =========================
# Paths (per-dataset)
# =========================
def sample_out_dir(samples_dir: str, sample_id: str, mkdir: bool = True) -> str:
    p = os.path.join(samples_dir, sample_id)
    if mkdir:
        os.makedirs(p, exist_ok=True)
    return p

def windows_jsonl_path(samples_dir: str, sample_id: str, mkdir: bool = False) -> str:
    d = sample_out_dir(samples_dir, sample_id, mkdir=mkdir)
    return os.path.join(d, "windows.jsonl")

def segments_path(samples_dir: str, sample_id: str, mkdir: bool = False) -> str:
    d = sample_out_dir(samples_dir, sample_id, mkdir=mkdir)
    return os.path.join(d, "segments.json")

def done_marker_path(samples_dir: str, sample_id: str, mkdir: bool = False) -> str:
    d = sample_out_dir(samples_dir, sample_id, mkdir=mkdir)
    return os.path.join(d, ".DONE")

# =========================
# Utils
# =========================
def read_video_info(mp4_path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {mp4_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps is None or fps != fps or abs(fps) < 1e-6:
        fps = 30.0
    return float(fps), max(0, nframes)

def encode_image_720p_png_lossless(img_bgr: np.ndarray) -> str:
    if img_bgr is None:
        return ""
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return ""
    if (w != TARGET_W) or (h != TARGET_H):
        img_bgr = cv2.resize(img_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(
        ".png",
        img_bgr,
        [int(cv2.IMWRITE_PNG_COMPRESSION), int(np.clip(PNG_COMPRESSION, 0, 9))]
    )
    return base64.b64encode(buf).decode("utf-8") if ok else ""

class FrameExtractor:
    def __init__(self, mp4_path: str):
        self.cap = cv2.VideoCapture(mp4_path)
    def close(self):
        if self.cap.isOpened():
            self.cap.release()
    def get_many_b64(self, frame_ids: List[int]) -> List[str]:
        sorted_indices = sorted(list(set(frame_ids)))
        frame_map: Dict[int, str] = {}
        for fid in sorted_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, bgr = self.cap.read()
            frame_map[fid] = encode_image_720p_png_lossless(bgr) if (ok and bgr is not None) else ""
        return [frame_map.get(fid, "") for fid in frame_ids]

def build_windows(fps: float, nframes: int) -> List[Window]:
    if fps < 1e-6:
        fps = 30.0
    win_len = max(1, int(round(WINDOW_SEC * fps)))
    step = max(1, int(round(STEP_SEC * fps)))
    windows: List[Window] = []

    def get_frames(s, e, num):
        idx = np.linspace(s, e, num=num).astype(int)
        return np.clip(idx, 0, nframes - 1).tolist()

    s = 0
    wid = 0
    while s < nframes:
        e = min(nframes - 1, s + win_len - 1)
        if (e - s < win_len // 2) and wid > 0:
            break
        windows.append(Window(wid, s, e, get_frames(s, e, FRAMES_PER_WINDOW)))
        wid += 1
        s += step
    return windows

def build_segments_via_cuts(sample_id, windows, by_wid, fps, nframes):
    if nframes == 0:
        return {}
    if fps < 1e-6:
        fps = 30.0

    raw_cuts = []
    instruction_timeline = [[] for _ in range(nframes)]
    frame_len = FRAMES_PER_WINDOW
    center_weights = np.hanning(frame_len + 2)[1:-1]

    for wid, w in enumerate(windows):
        rec = by_wid.get(wid)
        if not rec:
            continue
        vlm = rec.get("vlm_json", {})
        transitions = vlm.get("transitions", [])
        instructions = vlm.get("instructions", [])
        f_ids = w.frame_ids
        cur_len = len(f_ids)
        if cur_len == 0:
            continue

        for t_idx in transitions:
            try:
                idx = int(t_idx)
                if 0 <= idx < cur_len:
                    global_fid = f_ids[idx]
                    if cur_len == frame_len:
                        w_val = center_weights[idx]
                    else:
                        w_val = 1.0 if min(idx, cur_len - 1 - idx) > 2 else 0.5
                    raw_cuts.append((global_fid, float(w_val)))
            except:
                pass

        try:
            boundaries = [0] + [int(t) for t in transitions if 0 <= int(t) < cur_len] + [cur_len]
            boundaries = sorted(list(set(boundaries)))
            for i in range(len(boundaries) - 1):
                if i < len(instructions):
                    inst = str(instructions[i]).strip()
                    if inst and inst.lower() != "unknown":
                        s_local, e_local = boundaries[i], boundaries[i + 1]
                        for k in range(s_local, e_local):
                            if k < cur_len:
                                global_fid = f_ids[k]
                                if global_fid < nframes:
                                    instruction_timeline[global_fid].append(inst)
        except:
            pass

    final_cut_points = [0]
    if raw_cuts:
        raw_cuts.sort(key=lambda x: x[0])
        cluster_gap = max(1.0, 2.5 * fps)
        cur_frames = []
        cur_weights = []
        for fid, w in raw_cuts:
            if not cur_frames:
                cur_frames.append(fid)
                cur_weights.append(w)
                continue
            if (fid - cur_frames[-1]) < cluster_gap:
                cur_frames.append(fid)
                cur_weights.append(w)
            else:
                if cur_weights and sum(cur_weights) > 1e-9:
                    avg = np.average(cur_frames, weights=cur_weights)
                    final_cut_points.append(int(avg))
                else:
                    final_cut_points.append(int(np.mean(cur_frames)))
                cur_frames = [fid]
                cur_weights = [w]

        if cur_frames:
            if cur_weights and sum(cur_weights) > 1e-9:
                avg = np.average(cur_frames, weights=cur_weights)
                final_cut_points.append(int(avg))
            else:
                final_cut_points.append(int(np.mean(cur_frames)))

    final_cut_points.append(nframes)
    final_cut_points = sorted(list(set(final_cut_points)))

    final_output = []
    seg_id = 0
    for i in range(len(final_cut_points) - 1):
        s, e = int(final_cut_points[i]), int(final_cut_points[i + 1])
        min_frames = max(1, int(0.8 * fps))
        if (e - s) < min_frames:
            continue

        margin = int((e - s) * 0.2) if e > s else 0
        mid_s, mid_e = s + margin, e - margin

        candidates = []
        for f in range(mid_s, mid_e + 1):
            if f < nframes:
                candidates.extend(instruction_timeline[f])
        if not candidates:
            for f in range(s, e):
                if f < nframes:
                    candidates.extend(instruction_timeline[f])

        if candidates:
            best_inst = Counter(candidates).most_common(1)[0][0]
            final_output.append({
                "seg_id": seg_id,
                "start_frame": s,
                "end_frame": e,
                "instruction": best_inst,
                "confidence": 1.0
            })
            seg_id += 1

    return {"sample_id": sample_id, "nframes": nframes, "segments": final_output}

# =========================
# Server Loop
# =========================
@app.get("/get_job")
def get_job():
    with queue_lock:
        if not job_queue:
            return {"status": "empty"}
        job = job_queue.pop(0)
        inflight[job["task_id"]] = {"ts": time.time(), "job": job}
        return {"status": "ok", "data": job}

@app.post("/submit_result")
def submit_result(res: SubmitModel):
    tid = res.task_id
    job_info = None

    with queue_lock:
        if tid in inflight:
            job_info = inflight.pop(tid)

    # 结果为空：重试
    if not res.vlm_json:
        print(f"[Warn] Task {tid} returned EMPTY JSON (Token exploded?). Re-queueing...", flush=True)
        if job_info:
            with queue_lock:
                retry_counts[tid] = retry_counts.get(tid, 0) + 1
                if retry_counts[tid] <= MAX_RETRIES_PER_JOB:
                    job_queue.insert(0, job_info["job"])
                else:
                    print(f"[Err] Task {tid} failed {MAX_RETRIES_PER_JOB} times. Dropping.", flush=True)
        return {"status": "retry_triggered"}

    subset = str(res.meta.get("subset", SUBSET))
    sid = str(res.meta.get("sample_id", "unknown"))
    w_id = res.meta.get("window_id", None)

    samples_dir = SAMPLES_DIR_BY_SUBSET.get(subset)
    if not samples_dir:
        # 兜底：如果 meta 没带 subset 或 subset 不在列表里，仍然不阻塞
        samples_dir = os.path.join(RUN_BASE, subset, RUN_ID, "samples")
        os.makedirs(samples_dir, exist_ok=True)

    rec = {"task_id": tid, "window_id": w_id, "vlm_json": res.vlm_json}

    # 防止不同 subset 下 sample_id 重名导致锁冲突：用 subset::sid
    sample_key = f"{subset}::{sid}"
    with get_sample_lock(sample_key):
        with open(windows_jsonl_path(samples_dir, sid, mkdir=True), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"status": "received"}

# =========================
# Producer
# =========================
def compute_global_progress_total_and_done(ctxs: List[DatasetCtx]) -> Tuple[int, int]:
    total = 0
    done = 0
    for ctx in ctxs:
        total += len(ctx.sample_ids)
        for sid in ctx.sample_ids:
            if os.path.exists(os.path.join(ctx.samples_dir, sid, ".DONE")):
                done += 1
    return total, done

def producer_loop():
    # 计算全局 total / done（用于 114/1189）
    computed_total, computed_done = compute_global_progress_total_and_done(DATASET_CTXS)
    progress_total = PROGRESS_TOTAL_OVERRIDE if PROGRESS_TOTAL_OVERRIDE > 0 else computed_total
    global_done = computed_done

    print(
        f"[Server] Started. IMG=PNG, FIXED={TARGET_W}x{TARGET_H}, FRAMES_PER_WINDOW={FRAMES_PER_WINDOW}\n"
        f"[Plan] DATASETS={[(c.data_dir, c.subset) for c in DATASET_CTXS]}\n"
        f"[Resume] Already done: {global_done}/{progress_total} (computed_total={computed_total})",
        flush=True
    )

    # 每个 dataset 独立状态（但按顺序处理）
    states = {}
    for ctx in DATASET_CTXS:
        states[ctx.subset] = {
            "cur_idx": 0,
            "sample_status": {sid: 0 for sid in ctx.sample_ids},  # 0=need scan, 2=finalize, 3=done
        }

    dataset_idx = 0

    while True:
        # 1) inflight 超时重试
        now = time.time()
        with queue_lock:
            expired = [tid for tid, info in inflight.items() if now - info["ts"] > INFLIGHT_TIMEOUT_SEC]
            for tid in expired:
                job = inflight.pop(tid)["job"]
                retry_counts[tid] = retry_counts.get(tid, 0) + 1
                if retry_counts[tid] <= MAX_RETRIES_PER_JOB:
                    job_queue.append(job)

        # 2) 如果所有 datasets 都处理完
        if dataset_idx >= len(DATASET_CTXS):
            if AUTO_EXIT_AFTER_ALL_DONE:
                print(f"[All Done] {global_done}/{progress_total}. Exiting.", flush=True)
                os._exit(0)
            time.sleep(1.0)
            continue

        ctx = DATASET_CTXS[dataset_idx]
        st = states[ctx.subset]
        cur_idx = st["cur_idx"]
        sample_status = st["sample_status"]
        sample_ids = ctx.sample_ids

        # 3) 当前 dataset 处理完：等队列清空再切换到下一个
        if cur_idx >= len(sample_ids):
            with queue_lock:
                if not job_queue and not inflight:
                    print(f"[Dataset] Completed {ctx.subset}. Switching to next dataset...", flush=True)
                    dataset_idx += 1
            time.sleep(0.2)
            continue

        # 4) 正常生产任务（只在队列不满时）
        with queue_lock:
            q_len = len(job_queue)

        if q_len < MAX_QUEUE:
            sid = sample_ids[cur_idx]
            s_dir = os.path.join(ctx.data_dir, sid)

            # 已经 DONE：直接跳过
            if os.path.exists(os.path.join(ctx.samples_dir, sid, ".DONE")):
                sample_status[sid] = 3
                st["cur_idx"] += 1
                time.sleep(0.01)
                continue

            # 找视频
            mp4s = glob.glob(os.path.join(s_dir, "Frame_*.mp4"))
            if not mp4s:
                st["cur_idx"] += 1
                time.sleep(0.01)
                continue
            mp4 = mp4s[0]

            w_path = windows_jsonl_path(ctx.samples_dir, sid, mkdir=False)

            # Step A：生成窗口任务
            if sample_status[sid] == 0:
                try:
                    fps, nframes = read_video_info(mp4)
                    windows = build_windows(fps, nframes)

                    done_wids = set()
                    if os.path.exists(w_path):
                        with open(w_path, "r") as f:
                            for l in f:
                                try:
                                    done_wids.add(json.loads(l)["window_id"])
                                except:
                                    pass

                    extractor = FrameExtractor(mp4)
                    cnt = 0
                    for w in windows:
                        if w.window_id in done_wids:
                            continue

                        # task_id 带 subset，避免不同 dataset 同 sid 冲突
                        tid = f"{ctx.subset}::{sid}_w{w.window_id}"

                        active = False
                        with queue_lock:
                            if any(j["task_id"] == tid for j in job_queue) or tid in inflight:
                                active = True
                        if active:
                            continue

                        job = {
                            "task_id": tid,
                            "images": extractor.get_many_b64(w.frame_ids),
                            "meta": {
                                "subset": ctx.subset,
                                "sample_id": sid,
                                "window_id": w.window_id,
                                "frame_ids": w.frame_ids
                            }
                        }
                        with queue_lock:
                            job_queue.append(job)

                        cnt += 1
                        if cnt > 20:
                            break

                    extractor.close()

                    if cnt == 0:
                        # 没有要做的 window 了 -> 去 finalize
                        sample_status[sid] = 2

                except Exception as e:
                    print(f"[Err] {ctx.subset}/{sid}: {e}", flush=True)
                    st["cur_idx"] += 1

            # Step B：Finalize
            if sample_status[sid] == 2:
                try:
                    fps, nframes = read_video_info(mp4)
                    windows = build_windows(fps, nframes)

                    by_wid = {}
                    if os.path.exists(w_path):
                        with open(w_path, "r") as f:
                            for l in f:
                                try:
                                    d = json.loads(l)
                                    by_wid[d["window_id"]] = d
                                except:
                                    pass

                    if len(by_wid) >= len(windows):
                        print(f"[Finalize] {ctx.subset}/{sid}...", flush=True)

                        final_res = build_segments_via_cuts(sid, windows, by_wid, fps, nframes)
                        with open(segments_path(ctx.samples_dir, sid, mkdir=True), "w") as f:
                            json.dump(final_res, f, indent=2, ensure_ascii=False)

                        done_path = done_marker_path(ctx.samples_dir, sid, mkdir=True)
                        already_done = os.path.exists(done_path)
                        open(done_path, "w").close()

                        sample_status[sid] = 3
                        st["cur_idx"] += 1

                        # 打印全局进度：114/1189
                        if not already_done:
                            global_done += 1
                        print(f"[Progress] {global_done}/{progress_total}  (just finished: {ctx.subset}/{sid})", flush=True)

                except Exception as e:
                    print(f"[Err-Finalize] {ctx.subset}/{sid}: {e}", flush=True)

        time.sleep(0.1)

if __name__ == "__main__":
    t = threading.Thread(target=producer_loop, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
