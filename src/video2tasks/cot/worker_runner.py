"""CoT Worker runner — fetches CoT jobs and calls VLM."""

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
from .prompt import build_cot_prompt

MAX_LOCAL_RETRIES = 2


def _decode_b64_to_numpy(b64_str: str) -> Optional[np.ndarray]:
    if not b64_str:
        return None
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return np.array(img)[:, :, ::-1]  # RGB → BGR
    except Exception:
        return None


def _parse_cot_response(text: str) -> Dict[str, Any]:
    """Extract JSON with 'cot' key from raw VLM text."""
    if not text:
        return {}
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "cot" in data:
            return data
    except json.JSONDecodeError:
        pass
    # Fallback: find JSON object
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


def _log_parse_failure(task_id: str, raw: Any) -> None:
    snippet = str(raw)[:500] if raw else "<empty>"
    print(f"[CoT ParseFail] task={task_id}\n  raw (first 500): {snippet}")


def run_cot_worker(config: Config) -> None:
    """Run the CoT worker loop."""
    server_url = config.worker.server_url

    # ---- Create backend ----
    backend_kwargs: Dict[str, Any] = {}
    if config.worker.backend == "qwen3vl":
        cot_cfg = getattr(config, "cot", None)
        backend_kwargs = {
            "model_path": config.worker.qwen3vl.model_path,
            "device_map": config.worker.qwen3vl.device_map,
            "target_w": cot_cfg.target_width if cot_cfg else 424,
            "target_h": cot_cfg.target_height if cot_cfg else 240,
        }
    elif config.worker.backend == "remote_api":
        backend_kwargs = {
            "url": config.worker.remote_api.api_url,
            "api_key": config.worker.remote_api.api_key,
            "headers": config.worker.remote_api.headers,
            "timeout_sec": config.worker.remote_api.timeout_sec,
        }

    backend = create_backend(config.worker.backend, **backend_kwargs)
    print(f"[CoT Worker] Using backend: {backend.name}")
    backend.warmup()

    print(f"[CoT Worker] Connecting to {server_url}")

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
                        print(f"[CoT Worker] Failed to connect after {max_connection_retries} retries. Exiting.")
                        break
                    print(f"[CoT Worker] Waiting for server... ({connection_retry_count}/{max_connection_retries})")
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
                    time.sleep(1)
                    continue

                task_id = job.get("task_id", "unknown")
                instruction = job.get("instruction", "unknown")

                # ---- Decode images ----
                images_b64 = job.get("images", [])
                images: List[np.ndarray] = []
                for b64 in images_b64:
                    img = _decode_b64_to_numpy(b64)
                    if img is not None:
                        images.append(img)
                    else:
                        images.append(np.zeros((224, 224, 3), dtype=np.uint8))

                # ---- VLM inference ----
                prompt = build_cot_prompt(instruction, len(images))
                vlm_json: Dict[str, Any] = {}
                raw_result: Any = None
                t0 = time.time()

                for attempt in range(MAX_LOCAL_RETRIES):
                    try:
                        raw_result = backend.infer(images, prompt)

                        if isinstance(raw_result, dict):
                            if "cot" in raw_result:
                                vlm_json = raw_result
                            elif "text" in raw_result:
                                vlm_json = _parse_cot_response(raw_result["text"])
                            else:
                                vlm_json = {}
                        elif isinstance(raw_result, str):
                            vlm_json = _parse_cot_response(raw_result)
                        else:
                            vlm_json = {}
                    except Exception as e:
                        print(f"[CoT] Inference error: {e}")
                        vlm_json = {}

                    if vlm_json.get("cot"):
                        break
                    if attempt < MAX_LOCAL_RETRIES - 1:
                        print(f"[CoT] {task_id} empty (attempt {attempt + 1}), retrying...")

                elapsed = time.time() - t0

                if vlm_json.get("cot"):
                    cot_preview = vlm_json["cot"][:80] + "..." if len(vlm_json.get("cot", "")) > 80 else vlm_json.get("cot", "")
                    print(f"[CoT Done] {task_id} {elapsed:.1f}s -> {cot_preview}")
                else:
                    _log_parse_failure(task_id, raw_result)
                    print(f"[CoT Fail] {task_id} -> trigger server retry")

                # ---- Submit ----
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
                print("[CoT Worker] Stopping...")
                break
            except Exception as e:
                print(f"[CoT Error] Loop crashed: {e}")
                traceback.print_exc()
                time.sleep(1)

    finally:
        backend.cleanup()
