"""Robot Video Segmentor - Configuration management."""

from typing import List, Optional, Union
from pathlib import Path
import json
import os
import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    root: str = Field(..., description="Path to data root directory")
    video_subset: str = Field(
        ..., description="Subset containing per-sample videos (used by segment/cot)"
    )
    frame_subset: str = Field(
        ..., description="Subset containing per-sample extracted frames (used by vqa)"
    )
    data: List[Union[int, str]] = Field(
        default_factory=list,
        description=(
            "Optional sample selector. Supports integer indices over sorted sample list "
            "or explicit sample ids. Example: [0, 1] or ['000000', '000001']."
        ),
    )


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8099, description="Server port")
    max_queue: int = Field(default=32, description="Maximum job queue size")
    inflight_timeout_sec: float = Field(default=300.0, description="Timeout for in-flight jobs")
    max_retries_per_job: int = Field(default=5, description="Maximum retries per job")
    auto_exit_after_all_done: bool = Field(default=False, description="Auto exit when all done")


class Qwen3VLConfig(BaseModel):
    """Qwen3VL-specific configuration."""
    model_path: str = Field(
        default="Qwen/Qwen3-VL-32B-Instruct",
        description="Model path or HuggingFace model name"
    )
    device_map: str = Field(default="balanced", description="Device map strategy")


class RemoteAPIConfig(BaseModel):
    """Remote API backend configuration."""
    api_url: str = Field(default="http://127.0.0.1:8080/infer", description="Remote API URL")
    api_key: str = Field(default="", description="API key for remote API")
    timeout_sec: float = Field(default=60.0, description="Request timeout in seconds")
    headers: dict = Field(default_factory=dict, description="Extra headers for remote API")


class WorkerConfig(BaseModel):
    """Worker configuration."""
    server_url: str = Field(default="http://127.0.0.1:8099", description="Server URL")
    backend: str = Field(default="dummy", description="VLM backend type")
    qwen3vl: Qwen3VLConfig = Field(default_factory=Qwen3VLConfig)
    remote_api: RemoteAPIConfig = Field(default_factory=RemoteAPIConfig)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = ["dummy", "qwen3vl", "remote_api"]
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}, got {v}")
        return v


class WindowingConfig(BaseModel):
    """Video windowing configuration."""
    window_sec: float = Field(default=16.0, description="Window duration in seconds")
    step_sec: float = Field(default=8.0, description="Step size in seconds")
    frames_per_window: int = Field(default=16, description="Frames per window")
    target_width: int = Field(default=720, description="Target frame width")
    target_height: int = Field(default=480, description="Target frame height")
    png_compression: int = Field(default=0, description="PNG compression level (0-9)")


class ProgressConfig(BaseModel):
    """Progress tracking configuration."""
    total_override: int = Field(default=0, description="Override total count (0=auto)")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"level must be one of {allowed}, got {v}")
        return v_upper


class PromptConfig(BaseModel):
    """Prompt task selection configuration."""

    segment_task_id: str = Field(default="task00001", description="Task id for segment prompts")
    cot_task_id: str = Field(default="task00001", description="Task id for CoT prompts")
    vqa_task_id: str = Field(default="task00001", description="Task id for VQA prompts")


class VQAConfig(BaseModel):
    """VQA mode configuration (optional, only used when task_type='vqa')."""
    question_types: List[str] = Field(
        default=["spatial", "attribute", "existence", "count", "manipulation"],
        description="Question types to ask"
    )
    questions_per_type: dict = Field(
        default_factory=dict,
        description="Max questions per type, e.g. {spatial: 4, count: 2}. Omitted types use all slots."
    )
    task_context: str = Field(
        default="",
        description="Task context block (high-level task, object inventory, CanonicalRef mapping) injected into every VQA prompt."
    )
    sample_hz: float = Field(
        default=1.0,
        description="Frame sampling interval. frame_idx = numeric_id * sample_hz"
    )
    target_width: int = Field(
        default=424,
        description="Target frame width for VLM input"
    )
    target_height: int = Field(
        default=240,
        description="Target frame height for VLM input"
    )
    output_format: str = Field(
        default="jsonl",
        description="Output format: jsonl or parquet"
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        allowed = ["jsonl", "parquet"]
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}, got {v}")
        return v

    @field_validator("sample_hz")
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("sample_hz must be > 0")
        return v


class RunConfig(BaseModel):
    """Run/output configuration."""
    base_dir: str = Field(default="./runs", description="Base directory for outputs")
    run_id: str = Field(default="default", description="Run identifier")
    task_type: str = Field(default="segment", description="Task type: segment or vqa")

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        allowed = ["segment", "vqa", "cot"]
        if v not in allowed:
            raise ValueError(f"task_type must be one of {allowed}, got {v}")
        return v


class CoTConfig(BaseModel):
    """CoT mode configuration (optional, only used when task_type='cot').

    Reads segment outputs and generates chain-of-thought reasoning
    for each subtask segment.
    """
    segment_run_id: str = Field(
        default="default",
        description="run_id of the segment stage whose outputs to read"
    )
    high_level_instruction: str = Field(
        default="",
        description="High-level task instruction, e.g. 'Put the toy cars into the brown basket and put the fruit into the black basket'"
    )
    sample_hz: float = Field(
        default=2.0,
        description="Sampling frequency in Hz inside each segment (time-aligned to segment span)"
    )
    min_frames_per_segment: int = Field(
        default=4,
        description="Minimum sampled frames per segment"
    )
    max_frames_per_segment: int = Field(
        default=64,
        description="Maximum sampled frames per segment"
    )
    target_width: int = Field(
        default=424,
        description="Target frame width for VLM input"
    )
    target_height: int = Field(
        default=240,
        description="Target frame height for VLM input"
    )

    @field_validator("sample_hz")
    @classmethod
    def validate_sample_hz(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("cot.sample_hz must be > 0")
        return v

    @field_validator("min_frames_per_segment", "max_frames_per_segment")
    @classmethod
    def validate_positive_frames(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("cot frame limits must be > 0")
        return v


class Config(BaseModel):
    """Main application configuration."""
    datasets: List[DatasetConfig] = Field(default_factory=list)
    run: RunConfig = Field(default_factory=RunConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    vqa: VQAConfig = Field(default_factory=VQAConfig)
    cot: CoTConfig = Field(default_factory=CoTConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    progress: ProgressConfig = Field(default_factory=ProgressConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with env vars if present
        if "DATASETS" in os.environ:
            config.datasets = _parse_datasets_env(os.environ["DATASETS"])
        if "RUN_BASE" in os.environ:
            config.run.base_dir = os.environ["RUN_BASE"]
        if "RUN_ID" in os.environ:
            config.run.run_id = os.environ["RUN_ID"]
        if "PORT" in os.environ:
            config.server.port = int(os.environ["PORT"])
        if "SERVER_URL" in os.environ:
            config.worker.server_url = os.environ["SERVER_URL"]
        if "MODEL_PATH" in os.environ:
            config.worker.qwen3vl.model_path = os.environ["MODEL_PATH"]
        if "BACKEND" in os.environ:
            config.worker.backend = os.environ["BACKEND"]
        if "REMOTE_API_URL" in os.environ:
            config.worker.remote_api.api_url = os.environ["REMOTE_API_URL"]
        if "REMOTE_API_KEY" in os.environ:
            config.worker.remote_api.api_key = os.environ["REMOTE_API_KEY"]
        if "REMOTE_API_TIMEOUT" in os.environ:
            config.worker.remote_api.timeout_sec = float(os.environ["REMOTE_API_TIMEOUT"])
        if "REMOTE_API_HEADERS" in os.environ:
            headers_raw = os.environ["REMOTE_API_HEADERS"]
            headers = json.loads(headers_raw)
            if not isinstance(headers, dict):
                raise ValueError("REMOTE_API_HEADERS must be a JSON object")
            config.worker.remote_api.headers = headers
        
        return config
    
    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "Config":
        """Load configuration with priority: file > env > defaults."""
        if path:
            return cls.from_yaml(path)
        
        # Try to find config.yaml in current directory
        default_path = Path("config.yaml")
        if default_path.exists():
            return cls.from_yaml(default_path)
        
        # Fall back to environment variables
        return cls.from_env()


def _parse_datasets_env(spec: str) -> List[DatasetConfig]:
    """Parse DATASETS environment variable."""
    """
    New format (no backward compatibility):
    DATASETS="<root>:<video_subset>:<frame_subset>[;<root>:<video_subset>:<frame_subset>]"
    """
    configs = []
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    for p in parts:
        fields = [x.strip() for x in p.split(":")]
        if len(fields) != 3 or any(not x for x in fields):
            raise ValueError(
                "Invalid DATASETS item. Expected '<root>:<video_subset>:<frame_subset>'"
            )
        root, video_subset, frame_subset = fields
        configs.append(
            DatasetConfig(
                root=root,
                video_subset=video_subset,
                frame_subset=frame_subset,
            )
        )
    return configs


def select_sample_ids(all_sample_ids: List[str], data_selector: List[Union[int, str]]) -> List[str]:
    """Filter sample ids by selector list.

    Selector items:
    - int: positional index in ``all_sample_ids`` (sorted order)
    - str: exact sample id match
    """
    if not data_selector:
        return all_sample_ids

    selected: List[str] = []
    seen = set()
    for item in data_selector:
        sid: Optional[str] = None
        if isinstance(item, int):
            if 0 <= item < len(all_sample_ids):
                sid = all_sample_ids[item]
        else:
            s = str(item)
            if s in all_sample_ids:
                sid = s

        if sid is not None and sid not in seen:
            seen.add(sid)
            selected.append(sid)

    return selected
