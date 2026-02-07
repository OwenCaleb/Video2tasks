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


def _split_qas_by_type(
    qas: List[Dict[str, Any]], expected_types: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Split a flat qas list into per-type buckets.

    Each QA must have a 'type' field.  QAs whose type is not in
    *expected_types* are assigned to the first expected type.
    """
    by_type: Dict[str, List[Dict[str, Any]]] = {qt: [] for qt in expected_types}
    for qa in qas:
        qt = qa.get("type", "")
        if qt in by_type:
            by_type[qt].append(qa)
        elif expected_types:
            by_type[expected_types[0]].append(qa)
    return {qt: {"qas": items} for qt, items in by_type.items() if items}


def run_vqa_worker(config: Config) -> None:
    """Run the VQA worker loop.

    Uses a single combined VLM call covering all question types, then
    splits the returned QAs by their ``type`` field into per-type
    buckets so the output structure (``{"by_type": {...}}``) stays the
    same as before.
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

                # ---- Single combined VLM call ----
                by_type: Dict[str, Dict[str, Any]] = {}
                t0 = time.time()
                raw_result: Any = None

                for attempt in range(MAX_LOCAL_RETRIES):
                    try:
                        prompt = registry.build_combined_prompt(question_types, len(images))
                        raw_result = backend.infer(images, prompt)

                        all_qas: List[Dict[str, Any]] = []
                        if isinstance(raw_result, dict):
                            if "qas" in raw_result and isinstance(raw_result["qas"], list):
                                all_qas = raw_result["qas"]
                            elif "text" in raw_result:
                                parsed = _parse_vqa_response(raw_result["text"])
                                all_qas = parsed.get("qas", [])
                            else:
                                all_qas = []
                        elif isinstance(raw_result, str):
                            parsed = _parse_vqa_response(raw_result)
                            all_qas = parsed.get("qas", [])

                        if all_qas:
                            by_type = _split_qas_by_type(all_qas, question_types)
                            break
                    except Exception as e:
                        print(f"[VQA] Inference error: {e}")

                    if attempt < MAX_LOCAL_RETRIES - 1:
                        print(f"[VQA] {task_id} empty (attempt {attempt + 1}), retrying...")

                elapsed = time.time() - t0

                # ---- Summary ----
                if not by_type:
                    _log_parse_failure(task_id, "combined", raw_result)

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
