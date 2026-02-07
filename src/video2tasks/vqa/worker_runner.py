"""VQA Worker runner implementation."""

import time
import json
import base64
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


def build_vqa_prompt(question_types: List[str], n_images: int = 1) -> str:
    """Build VQA prompt from question types."""
    registry = get_default_prompts()
    return registry.build_combined_prompt(question_types, n_images)


def run_vqa_worker(config: Config) -> None:
    """Run the VQA worker loop."""
    server_url = config.worker.server_url
    
    # Create and warmup backend
    backend_kwargs = {}
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
    
    print(f"[VQA Worker] Connecting to {server_url}")
    
    max_connection_retries = 30
    connection_retry_count = 0
    
    try:
        while True:
            try:
                # Get job
                try:
                    r = requests.get(f"{server_url}/get_job", timeout=60)
                    connection_retry_count = 0
                except requests.exceptions.RequestException as e:
                    connection_retry_count += 1
                    if connection_retry_count >= max_connection_retries:
                        print(f"[VQA Worker] Failed to connect after {max_connection_retries} retries. Exiting.")
                        break
                    print(f"[VQA Worker] Waiting for server at {server_url}... (attempt {connection_retry_count}/{max_connection_retries})")
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
                question_types = job.get("question_types", ["spatial", "attribute", "existence", "count", "manipulation"])
                
                # Decode images
                images_b64 = job.get("images", [])
                images = []
                for b64 in images_b64:
                    img = decode_b64_to_numpy(b64)
                    if img is not None:
                        images.append(img)
                    else:
                        images.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
                # Build VQA prompt
                prompt = build_vqa_prompt(question_types, len(images))
                vlm_json: Dict[str, Any] = {}
                
                for attempt in range(MAX_LOCAL_RETRIES):
                    try:
                        result = backend.infer(images, prompt)
                        
                        # Convert result to VQA format
                        if isinstance(result, dict):
                            if not result:
                                vlm_json = {}
                            elif "qas" in result and isinstance(result.get("qas"), list):
                                vlm_json = result
                            elif "text" in result:
                                # Parse text response
                                vlm_json = _parse_vqa_response(result.get("text", ""))
                            else:
                                print(f"[VQA] {task_id} Missing 'qas' in backend output, triggering retry")
                                vlm_json = {}
                        elif isinstance(result, str):
                            vlm_json = _parse_vqa_response(result)
                        else:
                            vlm_json = {}
                    except Exception as e:
                        print(f"[VQA] Inference failed: {e}")
                        vlm_json = {}
                    
                    if not _is_empty_vlm_json(vlm_json):
                        break
                    
                    print(f"[VQA] {task_id} Empty response (attempt {attempt + 1}/{MAX_LOCAL_RETRIES})")
                
                qas_count = len(vlm_json.get("qas", []))
                if _is_empty_vlm_json(vlm_json):
                    print(f"[VQA Fail] {task_id} Returning empty to trigger server retry")
                else:
                    print(f"[VQA Done] {task_id} ({len(images)}f) -> {qas_count} QAs")
                
                # Submit result
                requests.post(
                    f"{server_url}/submit_result",
                    json={
                        "task_id": task_id,
                        "vlm_json": vlm_json,
                        "meta": job["meta"]
                    },
                    timeout=30
                )
            
            except KeyboardInterrupt:
                print("[VQA Worker] Stopping...")
                break
            except Exception as e:
                print(f"[VQA Error] Loop crashed: {e}")
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
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    
    return {}
