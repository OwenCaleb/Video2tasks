"""Dynamic prompt loaders by task id."""

from importlib import import_module
from typing import Any


def _load_task_module(mode: str, task_id: str, module_name: str):
    path = f"video2tasks.prompt.{mode}.{task_id}.{module_name}"
    try:
        return import_module(path)
    except ModuleNotFoundError as exc:
        if exc.name != path:
            raise
        raise ValueError(
            f"Prompt module not found: {path}. Check task id '{task_id}' for mode '{mode}'."
        ) from exc


def build_segment_prompt(task_id: str, n_images: int) -> str:
    mod = _load_task_module("seg", task_id, "builder")
    return mod.prompt_switch_detection(n_images)


def build_cot_prompt(task_id: str, high_level_instruction: str, subtask: str, n_images: int) -> str:
    mod = _load_task_module("cot", task_id, "builder")
    return mod.build_cot_prompt(high_level_instruction, subtask, n_images)


def create_vqa_prompt_registry(task_id: str) -> Any:
    try:
        mod = _load_task_module("vlm", task_id, "registry")
        return mod.get_default_prompts()
    except ValueError:
        profile_mod = _load_task_module("vlm", task_id, "task_profile")
        from video2tasks.prompt.vlm.adapter import VQATaskAdapter, build_registry

        demos = {}
        try:
            demos_mod = import_module(f"video2tasks.prompt.vlm.{task_id}.demos")
            demos = getattr(demos_mod, "VQA_DEMOS", {})
        except ModuleNotFoundError:
            demos = {}

        adapter = VQATaskAdapter(
            profile=profile_mod.TASK_PROFILE,
            task_context=getattr(profile_mod, "TASK_CONTEXT", ""),
            min_per_type=getattr(profile_mod, "MIN_QUESTIONS_PER_TYPE", 50),
            demos=demos,
        )
        return build_registry(adapter)
