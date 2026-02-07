"""VQA Worker runner implementation."""

import time
import json
import base64
import traceback
from io import BytesIO
from typing import Optional, Dict, Any, List

import requests
import numpy as np
from PIL import Image

from ..config import Config
from ..vlm import create_backend
from .prompts import get_default_prompts

MAX_LOCAL_RETRIES = 2


def _is_empty_vlm_json(vlm_json: Optional[Dict[str, Any]]) -> bool:
    return (not isinstance(vlm_json, dict)) or (not vlm_json)


def _log_parse_failure(task_id: str, qtype: str, raw: Any) -> None:
    """Debug: print first 500 chars of raw VLM output on parse failure."""
    snippet = str(raw)[:500] if raw else "<empty>"
    print(
        f"[VQA ParseFail] task={task_id}  type={qtype}\n"
        f"  raw (first 500): {snippet}"
    )


def decode_b64_to_numpy(b64_str: str) -> Optional[np.ndarray]:
    """Decode base64 string to numpy BGR array."""
    if not b64_str:
        return None
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        rgb_array = np.array(img)
        bgr_array = rgb_array[:, :, ::-1]
        return bgr_array
    except Exception:
        return None


def run_vqa_worker(config: Config) -> None:
    """Run the VQA worker loop.

    For each job (one frame), the worker iterates over *question_types*
    independently — one VLM call per type.  Results are collected into
    ``{"by_type": {"spatial": {"qas": [...]}, ...}}`` and submitted in
    a single HTTP POST.
    """
    server_url = config.worker.server_url

    # ---- Create backend ----
    backend_kwargs: Dict[str, Any] = {}
    if config.worker.backend == "qwen3vl":
        backend_kwargs = {
            "model_path": config.worker.qwen3vl.model_path,
            "device_map": config.worker.qwen3vl.device_map,
        }
    elif config.worker.backend == "remote_api":
        backend_kwargs = {
            "url": config.worker.remote_api.api_url,
            "api_key": config.worker.remote_api.api_key,
            "headers": config.worker.remote_api.headers,
            "timeout_sec": config.worker.remote_api.timeout_sec,
        }

    backend = create_backend(config.worker.backend, **backend_kwargs)
    print(f"[VQA Worker] Using backend: {backend.name}")
    backend.warmup()

    # ---- Prompt registry ----
    registry = get_default_prompts()

    print(f"[VQA Worker] Connecting to {server_url}")

    max_connection_retries = 30
    connection_retry_count = 0

    try:
        while True:
            try:
                # ---- Fetch job ----
                try:
                    r = requests.get(f"{server_url}/get_job", timeout=60)
                    connection_retry_count = 0
                except requests.exceptions.RequestException:
                    connection_retry_count += 1
                    if connection_retry_count >= max_connection_retries:
                        print(f"[VQA Worker] Failed to connect after {max_connection_retries} retries. Exiting.")
                        break
                    print(f"[VQA Worker] Waiting for server at {server_url}... ({connection_retry_count}/{max_connection_retries})")
                    time.sleep(2)
                    continue

                if r.status_code != 200:
                    time.sleep(0.5)
                    continue

                resp = r.json()
                if resp.get("status") == "empty":
                    time.sleep(0.5)
                    continue

                job = resp.get("data")
                if job is None:
                    print("[VQA Worker] Invalid job data received")
                    time.sleep(1)
                    continue

                task_id = job.get("task_id", "unknown")
                question_types: List[str] = job.get("question_types", [])

                # ---- Decode images ----
                images_b64 = job.get("images", [])
                images: List[np.ndarray] = []
                for b64 in images_b64:
                    img = decode_b64_to_numpy(b64)
                    if img is not None:
                        images.append(img)
                    else:
                        images.append(np.zeros((224, 224, 3), dtype=np.uint8))

                # ---- Per-type VLM calls ----
                by_type: Dict[str, Dict[str, Any]] = {}
                t0 = time.time()

                for qtype in question_types:
                    type_result: Dict[str, Any] = {}

                    for attempt in range(MAX_LOCAL_RETRIES):
                        raw_result: Any = None
                        try:
                            prompt = registry.build_single_type_prompt(qtype, len(images))
                            raw_result = backend.infer(images, prompt)

                            if isinstance(raw_result, dict):
                                if "qas" in raw_result and isinstance(raw_result["qas"], list):
                                    type_result = raw_result
                                elif "text" in raw_result:
                                    type_result = _parse_vqa_response(raw_result["text"])
                                else:
                                    type_result = {}
                            elif isinstance(raw_result, str):
                                type_result = _parse_vqa_response(raw_result)
                            else:
                                type_result = {}
                        except Exception as e:
                            print(f"[VQA] Inference error ({qtype}): {e}")
                            type_result = {}

                        if type_result.get("qas"):
                            break  # success
                        if attempt < MAX_LOCAL_RETRIES - 1:
                            print(f"[VQA] {task_id} [{qtype}] empty (attempt {attempt + 1}), retrying...")

                    if type_result.get("qas"):
                        by_type[qtype] = type_result
                    else:
                        _log_parse_failure(task_id, qtype, raw_result)

                elapsed = time.time() - t0

                # ---- Summary ----
                total_qas = sum(len(v.get("qas", [])) for v in by_type.values())
                ok_types = list(by_type.keys())
                fail_types = [qt for qt in question_types if qt not in by_type]

                if by_type:
                    print(
                        f"[VQA Done] {task_id} "
                        f"ok={ok_types} fail={fail_types} "
                        f"qas={total_qas} {elapsed:.1f}s"
                    )
                else:
                    print(f"[VQA Fail] {task_id} All types failed -> trigger server retry")

                # ---- Submit ----
                vlm_json = {"by_type": by_type} if by_type else {}
                requests.post(
                    f"{server_url}/submit_result",
                    json={
                        "task_id": task_id,
                        "vlm_json": vlm_json,
                        "meta": job["meta"],
                        "latency_s": elapsed,
                    },
                    timeout=30,
                )

            except KeyboardInterrupt:
                print("[VQA Worker] Stopping...")
                break
            except Exception as e:
                print(f"[VQA Error] Loop crashed: {e}")
                traceback.print_exc()
                time.sleep(1)

    finally:
        backend.cleanup()


def _parse_vqa_response(text: str) -> Dict[str, Any]:
    """Parse VQA response from text."""
    if not text:
        return {}

    text = text.strip().replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "qas" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON in text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return {}
