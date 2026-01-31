# Robot Video Segmentor - Open Source Publishing Plan

## TL;DR

> **Quick Summary**: Reorganize the current two-script prototype into a small Python package with YAML config + pluggable VLM backend, then publish as a public MIT-licensed GitHub repo using `gh`.
>
> **Deliverables**:
> - Packaged Python project (`pyproject.toml`) with importable module `robot_video_segmentor`
> - CLI entrypoints: `rvs-server`, `rvs-worker`
> - YAML config: `config.example.yaml`
> - Open-source hygiene: `README.md`, `LICENSE` (MIT), `.gitignore`
> - GitHub repo created + initial push

**Estimated Effort**: Medium
**Parallel Execution**: YES (2 waves)
**Critical Path**: restructure code → config + CLI → docs + packaging → git/gh publish

---

## Context

### Original Request
- Publish the project under the current folder to GitHub as an open source project.
- Before publishing: “整理一下” and do a structured re-organization.

### Known Current Files
- `server_A_pi0_final.py`: FastAPI/Uvicorn server, job queue, OpenCV video window sampling, writes `windows.jsonl` and `segments.json`.
- `worker_B_pi0_final.py`: worker that loads Qwen3-VL (Transformers/Torch), polls server `/get_job`, runs inference, submits `/submit_result`.

### Confirmed Decisions
- Repo name: `robot-video-segmentor`
- License: MIT
- Config: YAML (primary), optional env overrides
- Packaging: `pyproject.toml` + console scripts (`rvs-server`, `rvs-worker`)
- VLM: keep interface pluggable; recommend Qwen3VL 32B but user selects backend/model via config
- GitHub publishing flow: prefer `gh` CLI

---

## Work Objectives

### Core Objective
Make this repository runnable and understandable for external users by separating concerns (server/worker/config/vlm), removing private paths, and providing a documented configuration-driven workflow.

### Concrete Deliverables
- Source layout under `src/robot_video_segmentor/`
- Entry points: `rvs-server`, `rvs-worker`
- `config.example.yaml` covering datasets, run dirs, server port/url, VLM backend selection
- Minimal “smoke” workflow that runs on Windows/CPU without requiring the 32B model (via a lightweight backend)
- Open source basics: README, MIT license, gitignore

### Must NOT Have (Guardrails)
- Do NOT commit datasets, videos, run outputs, or model weights.
- Do NOT hardcode personal `/mnt/...` paths; must be config-driven with safe defaults.
- Do NOT bake in a single VLM vendor; keep backend interface.
- Avoid over-engineering (no unnecessary microservices, no complex plugin registry).

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (currently only scripts)
- **Strategy**: Automated smoke verification + minimal unit tests for pure functions
- **Framework**: `pytest` (small set), plus CLI `--help` and config parse checks

### Automated Verification (agent-executable)
- `python -m pip install -e .` succeeds on Windows and Linux
- `rvs-server --help` and `rvs-worker --help` exit code 0
- `python -m robot_video_segmentor.validate_config --config config.example.yaml` exits 0
- Optional: start server with dummy backend and run one “fake” job end-to-end locally (no GPU)

---

## Execution Strategy

Wave 1 (Start Immediately):
- Task 1: Repository hygiene + ignore rules
- Task 2: Define target package layout + config schema

Wave 2 (After Wave 1):
- Task 3: Move server logic into package + CLI
- Task 4: Move worker logic into package + pluggable VLM backends
- Task 5: Packaging + docs + GitHub publish

---

## TODOs

### 1) Open-source hygiene baseline (repo-safe defaults)

**What to do**:
- Add `.gitignore` to exclude:
  - video/data artifacts: `*.mp4`, `*.avi`, `*.mkv`, `*.zip`
  - run outputs: `runs/`, `samples/`, `*.jsonl`, `segments.json`, `.DONE`
  - model weights/cache: `*.pt`, `*.bin`, `*.safetensors`, `__pycache__/`, `.venv/`, `*.log`
- Add `LICENSE` (MIT).
- Add `README.md` stub (what it does, quickstart, config).

**References**:
- `server_A_pi0_final.py:23` - current default dataset roots (must not ship as hardcoded personal defaults)
- `worker_B_pi0_final.py:19` - current default model path (must be config-driven)

**Acceptance Criteria**:
- `git status` shows only intended source/doc files (no datasets/runs).
- `.gitignore` covers outputs produced by `server_A_pi0_final.py` (`windows.jsonl`, `segments.json`, `.DONE`).

### 2) Package skeleton + YAML config contract

**What to do**:
- Create `pyproject.toml` for package `robot_video_segmentor` (src layout).
- Add YAML config files:
  - `config.example.yaml` (checked-in)
  - `config.yaml` (ignored)
- Define config schema (Pydantic dataclasses/models) and a validator command.
- Decide naming + structure:
  - `robot_video_segmentor/config.py`
  - `robot_video_segmentor/cli/server.py`
  - `robot_video_segmentor/cli/worker.py`
  - `robot_video_segmentor/vlm/base.py`

**Assumptions / Defaults Applied**:
- Add a lightweight `dummy` backend for Windows/CPU smoke runs.
  - This does NOT replace real inference; it just unblocks end-to-end verification.

**Acceptance Criteria**:
- `python -m robot_video_segmentor.validate_config --config config.example.yaml` exits 0.
- Running with missing required fields produces a clear error message and non-zero exit.

### 3) Server refactor into module + CLI

**What to do**:
- Move the server code from `server_A_pi0_final.py` into `robot_video_segmentor/server/app.py`.
- Separate:
  - windowing/extraction utilities (pure functions)
  - FastAPI routes
  - producer loop orchestration
- Replace environment-variable config reads with YAML config load + optional env overrides.

**References (patterns to preserve)**:
- `server_A_pi0_final.py:69` - `parse_datasets()` format and multi-dataset handling
- `server_A_pi0_final.py:203` - `build_windows()` logic and sampling strategy
- `server_A_pi0_final.py:346` - `/get_job` API contract
- `server_A_pi0_final.py:555` - finalize step writing `segments.json` and `.DONE`

**Acceptance Criteria**:
- `rvs-server --config config.example.yaml` starts and binds configured host/port.
- `/get_job` returns `{"status":"empty"}` when no jobs.

### 4) Worker refactor + pluggable VLM backends

**What to do**:
- Extract worker loop into `robot_video_segmentor/worker/runner.py`.
- Define a backend interface, e.g. `VLMBackend.infer(images, prompt) -> dict`.
- Provide at least two backends:
  - `dummy` (no heavy deps): returns stable JSON for smoke runs
  - `qwen3vl` (optional extra): wraps current Transformers/Torch logic
- Make backend selection purely config-driven.
- Keep prompt generation in a dedicated module (so users can edit it).

**References (patterns to preserve)**:
- `worker_B_pi0_final.py:26` - `prompt_switch_detection()` content
- `worker_B_pi0_final.py:79` - `extract_json()` robustness
- `worker_B_pi0_final.py:113` - polling loop and submit shape

**Acceptance Criteria**:
- `rvs-worker --help` exits 0.
- With backend=`dummy`, worker can process at least one job and submit a non-empty `vlm_json`.

### 5) Packaging, docs, and GitHub publish

**What to do**:
- Ensure `pip install -e .` works cross-platform.
- Add dependency grouping:
  - core: fastapi, uvicorn, pydantic, numpy, opencv-python, requests, pyyaml
  - extra: `qwen3vl` includes torch + transformers + pillow
- Expand `README.md`:
  - Architecture diagram (text)
  - Quickstart (dummy backend)
  - GPU setup notes (Linux + CUDA)
  - How to add a custom VLM backend
- Initialize git, make initial commit, create GitHub repo via `gh`, push.

**Acceptance Criteria**:
- `python -m pip install -e .` succeeds on Windows.
- `python -m pip install -e .[qwen3vl]` succeeds on Linux GPU machines (best-effort; doc if torch install differs).
- `gh repo create robot-video-segmentor --public` flow documented and produces a remote with a pushed `main` branch.

---

## Decisions Needed (placeholders)

- GitHub owner: personal account name vs organization name.
- Whether to include CI (GitHub Actions) for lint/tests (recommended, but optional).
