"""CLI to print complete prompts for a selected mode/task.

Usage examples:
    python -m video2tasks.prompt.preview_prompt --mode seg --task task00002 --n-images 16
    python -m video2tasks.prompt.preview_prompt --mode cot --config config.yaml
    python -m video2tasks.prompt.preview_prompt --mode vqa --qtype spatial --max-questions 2
"""

import argparse
from pathlib import Path

from ..config import Config
from .loader import build_segment_prompt, build_cot_prompt, create_vqa_prompt_registry


def _load_cfg(config_path: str | None) -> Config | None:
    if not config_path:
        return None
    return Config.from_yaml(Path(config_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview full task prompt")
    parser.add_argument("--mode", choices=["seg", "cot", "vqa"], required=True)
    parser.add_argument("--task", default=None, help="Task id, e.g. task00001/task00002")
    parser.add_argument("--config", default=None, help="Optional config path to read task ids and context")
    parser.add_argument("--n-images", type=int, default=16)

    # CoT-specific
    parser.add_argument("--high-level", default="Put the grapes into the black basket.")
    parser.add_argument("--subtask", default="Put the grapes into the black basket")

    # VQA-specific
    parser.add_argument("--qtype", default="spatial")
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--task-context", default=None)

    args = parser.parse_args()
    cfg = _load_cfg(args.config)

    if args.mode == "seg":
        task_id = args.task or (cfg.prompt.segment_task_id if cfg else "task00001")
        prompt = build_segment_prompt(task_id, args.n_images)
        print(prompt)
        return

    if args.mode == "cot":
        task_id = args.task or (cfg.prompt.cot_task_id if cfg else "task00001")
        high_level = args.high_level
        subtask = args.subtask
        if cfg:
            high_level = cfg.cot.high_level_instruction or high_level
        prompt = build_cot_prompt(task_id, high_level, subtask, args.n_images)
        print(prompt)
        return

    task_id = args.task or (cfg.prompt.vqa_task_id if cfg else "task00001")
    registry = create_vqa_prompt_registry(task_id)
    task_context = args.task_context
    if task_context is None and cfg:
        task_context = cfg.vqa.task_context
    prompt = registry.build_single_type_prompt(
        args.qtype,
        n_images=args.n_images,
        max_questions=args.max_questions,
        task_context=task_context,
    )
    print(prompt)


if __name__ == "__main__":
    main()
