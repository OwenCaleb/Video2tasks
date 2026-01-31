#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# =========================
# Config
# =========================
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8099") 
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/data/wangyinxi/models/Qwen3-VL-32B-Instruct") 
DEVICE_MAP = "balanced"

# =========================
# Prompt
# =========================
def prompt_switch_detection(n_images: int) -> str:
    """
    生成用于检测机器人操作任务切换点的提示词。
    优化目标：降低对单物体复杂操作的过度切分（Over-segmentation）。
    """
    return (
        f"You are a robotic vision analyzer watching a {n_images}-frame video clip of household manipulation tasks.\n"
        f"**Mapping:** Image indices range from 0 to {n_images - 1}.\n\n"
        
        "### Goal\n"
        "Detect **Atomic Task Boundaries** (Switch Points).\n"
        "A 'Switch' occurs strictly when the robot **completes** interaction with one object and **starts** interacting with a DIFFERENT object.\n\n"
        
        "### Core Logic (The 'Distinct Object' Rule)\n"
        "1. **True Switch:** Robot releases Object A (e.g., a cup) and moves to grasp Object B (e.g., a spoon). -> MARK SWITCH.\n"
        "2. **False Switch (IMPORTANT):** If the robot is manipulating different parts of the **SAME** object (e.g., folding sleeves then folding the body of the same shirt), this is **NOT** a switch. Treat it as one continuous task.\n"
        "3. **Visual Similarity:** Be careful with objects of the same color. Only mark a switch if you clearly see the robot **physically separate** from the first item before touching the second.\n\n"
        
        "### Output Format: Strict JSON\n"
        "Your response must be a valid JSON object including a 'thought' field for step-by-step analysis, 'transitions' for the switch indices, and 'instructions' for the task labels.\n\n"
        
        "### Representative Examples\n"
        "**Example 1: Table Setting (True Switch)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-5: Robot places a fork. Frame 6: Hand releases fork and moves to the spoon. Frame 7: Hand grasps spoon. Switch detected at 6.\",\n"
        "  \"transitions\": [6],\n"
        "  \"instructions\": [\"Place the fork\", \"Place the spoon\"]\n"
        "}\n\n"
        
        "**Example 2: Folding Laundry (False Switch - Same Object)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-10: Robot folds the left sleeve of the black shirt. Frames 11-20: Robot folds the body of the **same** black shirt. Although the grasp changed, the object remains the same. The action is continuous.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Fold the black shirt\"]\n"
        "}\n\n"
        
        "**Example 3: Cleaning (Continuous)**\n"
        "{\n"
        "  \"thought\": \"Frames 0-15: Robot is wiping the counter. The motion is repetitive, but it is the same task. No switch.\",\n"
        "  \"transitions\": [],\n"
        "  \"instructions\": [\"Wipe the counter\"]\n"
        "}"
    )

# =========================
# Utils
# =========================
def decode_b64(b64_str: str) -> Optional[Image.Image]:
    try:
        if not b64_str: return None
        return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")
    except: return None

def extract_json(text: str) -> Dict:
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        try:
            s = text.find("{")
            e = text.rfind("}")
            if s != -1 and e != -1:
                return json.loads(text[s:e+1])
        except: pass
    return {}

# =========================
# Model Loader
# =========================
print(f"[Worker] Loading Qwen3VL from {MODEL_PATH}...")
try:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=DEVICE_MAP
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    print("[Worker] Model ready.")
except Exception as e:
    print(f"[Worker] Model Load Error: {e}")
    exit(1)

# =========================
# Main Loop
# =========================
def main():
    print(f"[Worker] Connecting to {SERVER_URL}")
    
    while True:
        try:
            # 1. Get Job
            try:
                r = requests.get(f"{SERVER_URL}/get_job", timeout=60)
            except requests.exceptions.RequestException:
                print(f"[Worker] Waiting for server at {SERVER_URL}...")
                time.sleep(2)
                continue

            if r.status_code != 200 or r.json().get("status") == "empty":
                time.sleep(0.5)
                continue
            
            job = r.json()["data"]
            task_id = job["task_id"]
            
            # 2. Prepare Images
            images_b64 = job.get("images", [])
            pil_images = [decode_b64(b) for b in images_b64]
            pil_images = [img if img else Image.new('RGB', (224, 224)) for img in pil_images]
            
            # 3. Construct Inputs
            prompt = prompt_switch_detection(len(pil_images))
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": img} for img in pil_images] + 
                            [{"type": "text", "text": prompt}]
            }]
            text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs = pil_images
            inputs = processor(text=[text_inputs], images=image_inputs, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 4. Inference with Local Retry
            # 本地尝试重试 1 次，如果还是 Empty，再提交给 Server 让它重新分配
            MAX_LOCAL_RETRIES = 2
            vlm_json = {}
            
            for attempt in range(MAX_LOCAL_RETRIES):
                try:
                    # 提升 Token 上限到 1024，防止物理截断
                    with torch.no_grad():
                        out_ids = model.generate(**inputs, max_new_tokens=1024)
                    
                    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], out_ids)]
                    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    vlm_json = extract_json(output_text)
                    
                    if vlm_json:
                        # 成功解析，跳出循环
                        break
                    else:
                        print(f"[Warn] {task_id} Empty JSON (Attempt {attempt+1}/{MAX_LOCAL_RETRIES}). Raw: {output_text[:50]}...")
                except Exception as e:
                    print(f"[Err] Inference Failed: {e}")

            # 5. Submit (即使是 Empty 也提交，让 Server 触发重试逻辑)
            if vlm_json:
                print(f"[Done] {task_id} ({len(pil_images)}f) -> Cuts: {vlm_json.get('transitions', [])}")
            else:
                print(f"[Fail] {task_id} Returning Empty to trigger Server Retry.")
            
            requests.post(
                f"{SERVER_URL}/submit_result",
                json={
                    "task_id": task_id,
                    "vlm_json": vlm_json, # 如果是 {}，Server 会打回重试
                    "meta": job["meta"]
                },
                timeout=30
            )

        except KeyboardInterrupt:
            print("[Worker] Stopping...")
            break
        except Exception as e:
            print(f"[Error] Loop crashed: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()