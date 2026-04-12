#!/usr/bin/env bash
set -euo pipefail

# ================== CONFIG ==================
CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV="wallx"

CFG="config.yaml"
RESET_DONE=0

LOGDIR="logs"
PIDDIR="pids"
POLL_SEC=2

# stage selection: any combination from segment,vqa,cot
STAGES_CSV="segment,vqa,cot"

# batch controls
BATCH_ROOT=""
DATASET_GLOB="*_old"
RUN_BASE_PARENT="."
TASKS_FILE="dataset_tasks.yaml"
DRY_RUN=0

# idle detection: MUST be consecutive
IDLE_N=100
IDLE_PAT='GET /get_job HTTP/1.1" 200 OK'
# ===========================================

mkdir -p "$LOGDIR" "$PIDDIR"

usage() {
  cat <<EOF
Usage:
  bash run_serial_gpu1.sh [options]

Options:
  -c, --config PATH          template config path (default: config.yaml)
  --stages CSV               stage combination, e.g. segment,vqa,cot or segment,cot
  --batch-root PATH          batch root; process all datasets matching --dataset-glob
  --dataset-glob GLOB        dataset directory glob in batch mode (default: *_old)
  --run-base-parent PATH     parent dir for dynamic base_dir (default: .)
  --tasks-file PATH          dataset task mapping yaml (default: dataset_tasks.yaml)
  --reset-done 0|1           remove stale .DONE markers before each stage (default: 0)
  --dry-run                  print planned datasets/stages/config, do not execute
  -h, --help                 show this help

Examples:
  bash run_serial_gpu1.sh --stages segment,vqa,cot
  bash run_serial_gpu1.sh --stages segment,cot
  bash run_serial_gpu1.sh --batch-root /path/to/lerobot --dataset-glob "*_old" --stages segment,vqa,cot
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CFG="$2"
      shift 2
      ;;
    --stages)
      STAGES_CSV="$2"
      shift 2
      ;;
    --batch-root)
      BATCH_ROOT="$2"
      shift 2
      ;;
    --dataset-glob)
      DATASET_GLOB="$2"
      shift 2
      ;;
    --run-base-parent)
      RUN_BASE_PARENT="$2"
      shift 2
      ;;
    --tasks-file)
      TASKS_FILE="$2"
      shift 2
      ;;
    --reset-done)
      RESET_DONE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

run_in_env() {
  local cmd="$1"
  bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && $cmd"
}

normalize_stage() {
  local s="$1"
  case "$s" in
    seg|segment)
      echo "segment"
      ;;
    vqa)
      echo "vqa"
      ;;
    cot)
      echo "cot"
      ;;
    *)
      return 1
      ;;
  esac
}

parse_stages() {
  STAGES=()
  IFS=',' read -r -a raw <<< "$STAGES_CSV"
  for item in "${raw[@]}"; do
    local trimmed
    trimmed="$(echo "$item" | xargs)"
    if [ -z "$trimmed" ]; then
      continue
    fi
    local n
    n="$(normalize_stage "$trimmed")" || {
      echo "Invalid stage in --stages: $trimmed"
      exit 1
    }
    STAGES+=("$n")
  done
  if [ "${#STAGES[@]}" -eq 0 ]; then
    echo "--stages resolved to empty list"
    exit 1
  fi
}

cleanup_all() {
  echo
  echo "===== MANUAL CLEANUP ====="
  echo "[cleanup] killing any remaining processes from pid files..."
  local pidfile
  for pidfile in "$PIDDIR"/v2t-*.pid; do
    [ -e "$pidfile" ] || continue
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ]; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  done
  echo "[cleanup] done. see logs in: $LOGDIR/"
}

# 中断时自动清理
trap 'echo; echo "[trap] keyboard interrupt, cleaning up..."; cleanup_all; exit 130' INT TERM

preflight_check() {
  local cfg_path="$1"
  echo
  echo "===== PREFLIGHT ====="
  run_in_env "v2t-validate -c '$cfg_path'"

  CFG_PATH="$cfg_path" bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && python - <<'PY'
from pathlib import Path
import os
from video2tasks.config import Config

cfg = Config.from_yaml(Path(os.environ['CFG_PATH']))
print('[cfg] dataset mappings:')
for i, ds in enumerate(cfg.datasets):
    print(
        f'  - ds[{i}] root={ds.root} video_subset={ds.video_subset} '
        f'frame_subset={ds.frame_subset} data={ds.data}'
    )
print(f'[cfg] run_base={cfg.run.base_dir} run_id={cfg.run.run_id}')
PY"
}

reset_done_markers() {
  local mode="$1"
  local cfg_path="$2"
  if [ "$RESET_DONE" -ne 1 ]; then
    return
  fi

  echo "[${mode}] RESET_DONE=1, removing stale .DONE markers..."
  MODE="$mode" CFG_PATH="$cfg_path" bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && python - <<'PY'
from pathlib import Path
import os
from video2tasks.config import Config

cfg = Config.from_yaml(Path(os.environ['CFG_PATH']))
mode = os.environ['MODE']
removed = 0

for ds in cfg.datasets:
    subset = ds.frame_subset if mode == 'vqa' else ds.video_subset
    out_dir = Path(cfg.run.base_dir) / subset / cfg.run.run_id / mode
    if not out_dir.exists():
        continue
    for marker in out_dir.rglob('.DONE'):
        marker.unlink(missing_ok=True)
        removed += 1

print(f'[reset] mode={mode} removed_done_markers={removed}')
PY"
}

start_pair() {
  local tag="$1"
  local cfg_path="$2"
  local mode="$3"

  echo
  echo "===== START ${tag} (mode=${mode}) ====="

  nohup bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && v2t-server -c '$cfg_path' --mode '$mode'" \
    > "$LOGDIR/v2t-server.${tag}.log" 2>&1 & echo $! > "$PIDDIR/v2t-server.${tag}.pid"
  echo "[${tag}] server pid=$(cat "$PIDDIR/v2t-server.${tag}.pid")"

  nohup bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && v2t-worker -c '$cfg_path' --mode '$mode'" \
    > "$LOGDIR/v2t-worker.${tag}.log" 2>&1 & echo $! > "$PIDDIR/v2t-worker.${tag}.pid"
  echo "[${tag}] worker pid=$(cat "$PIDDIR/v2t-worker.${tag}.pid")"

  echo "[${tag}] env check (paths):"
  run_in_env "which v2t-server && which v2t-worker" || true

  echo "[${tag}] tail logs:"
  echo "  tail -f $LOGDIR/v2t-server.${tag}.log"
  echo "  tail -f $LOGDIR/v2t-worker.${tag}.log"
}

stop_worker() {
  local tag="$1"
  local wpid
  wpid="$(cat "$PIDDIR/v2t-worker.${tag}.pid" 2>/dev/null || true)"
  echo "[${tag}] stopping worker pid=$wpid ..."
  [ -n "$wpid" ] && kill "$wpid" 2>/dev/null || true
  sleep 2
  [ -n "$wpid" ] && kill -0 "$wpid" 2>/dev/null && kill -9 "$wpid" 2>/dev/null || true
  rm -f "$PIDDIR/v2t-worker.${tag}.pid"
  echo "[${tag}] worker stopped."
}

stop_server() {
  local tag="$1"
  local spid
  spid="$(cat "$PIDDIR/v2t-server.${tag}.pid" 2>/dev/null || true)"
  echo "[${tag}] stopping server pid=$spid ..."
  [ -n "$spid" ] && kill "$spid" 2>/dev/null || true
  sleep 2
  [ -n "$spid" ] && kill -0 "$spid" 2>/dev/null && kill -9 "$spid" 2>/dev/null || true
  rm -f "$PIDDIR/v2t-server.${tag}.pid"
  echo "[${tag}] server stopped."
}

wait_idle_by_server_log() {
  local tag="$1"
  local logfile="$LOGDIR/v2t-server.${tag}.log"

  echo
  echo "[${tag}] waiting for ${IDLE_N} consecutive lines matching: ${IDLE_PAT}"
  echo "[${tag}] logfile: $logfile"

  while true; do
    local spid
    spid="$(cat "$PIDDIR/v2t-server.${tag}.pid" 2>/dev/null || true)"
    if [ -n "$spid" ] && ! kill -0 "$spid" 2>/dev/null; then
      echo "[${tag}] ERROR: server pid=$spid is not running, cannot wait idle."
      exit 1
    fi

    if [ -f "$logfile" ]; then
      local ok
      ok=$(tail -n "$IDLE_N" "$logfile" 2>/dev/null | grep -c "$IDLE_PAT" || true)
      echo "[${tag}] idle_check ok=$ok/$IDLE_N"
      if [ "$ok" -eq "$IDLE_N" ]; then
        echo "[${tag}] IDLE reached: last ${IDLE_N} lines are get_job 200 OK."
        break
      fi
    fi

    sleep "$POLL_SEC"
  done
}

make_temp_config_for_dataset() {
  local cfg_template="$1"
  local dataset_root="$2"
  local run_base_parent="$3"
  local tmp_cfg="$4"
  local tasks_file="$5"

  CFG_TEMPLATE="$cfg_template" DATASET_ROOT="$dataset_root" RUN_BASE_PARENT="$run_base_parent" TMP_CFG="$tmp_cfg" TASKS_FILE="$tasks_file" \
    bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && python - <<'PY'
from pathlib import Path
import os
import re
from collections import Counter
import yaml
import json

cfg_path = Path(os.environ['CFG_TEMPLATE'])
dataset_root = Path(os.environ['DATASET_ROOT']).resolve()
run_base_parent = Path(os.environ['RUN_BASE_PARENT']).resolve()
tmp_cfg = Path(os.environ['TMP_CFG']).resolve()
tasks_file = Path(os.environ['TASKS_FILE']).resolve()

with cfg_path.open('r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

if not data.get('datasets'):
    raise ValueError('template config has empty datasets')

dataset_name = dataset_root.name
dataset_no_old = re.sub(r'_old$', '', dataset_name)
run_base = run_base_parent / f'{dataset_no_old}_runs'

data['datasets'][0]['root'] = str(dataset_root)
data['run']['base_dir'] = str(run_base)


def _extract_instruction_from_annotation(path: Path) -> str:
  if not path.exists():
    return ''
  with path.open('r', encoding='utf-8') as f:
    ann = json.load(f)

  # 1) Direct instruction-like fields
  for key in ('instruction', 'high_level_instruction', 'task_instruction'):
    v = ann.get(key)
    if isinstance(v, str) and v.strip():
      return v.strip()

  # 2) Nested meta block
  meta = ann.get('meta')
  if isinstance(meta, dict):
    for key in ('instruction', 'high_level_instruction', 'task_instruction'):
      v = meta.get(key)
      if isinstance(v, str) and v.strip():
        return v.strip()

  # 3) Fallback from episodes[*].subtasks[*].label (majority label)
  labels = []
  episodes = ann.get('episodes', {})
  if isinstance(episodes, dict):
    for ep in episodes.values():
      if not isinstance(ep, dict):
        continue
      subtasks = ep.get('subtasks', [])
      if not isinstance(subtasks, list):
        continue
      for st in subtasks:
        if isinstance(st, dict):
          label = st.get('label')
          if isinstance(label, str) and label.strip():
            labels.append(label.strip())
  if labels:
    return Counter(labels).most_common(1)[0][0]
  return ''


def _build_vqa_task_context(instruction: str, objects: list[str]) -> str:
  lines = [
    'High-level task:',
    instruction or '',
    '',
    'Object inventory (CanonicalRef only):',
  ]
  for obj in objects:
    lines.append(f'- {obj}')
  return '\n'.join(lines).strip()


# IMPORTANT: Keep data root and annotation root strictly separated.
# - data root:   .../<dataset>_old
# - annotation:  .../<dataset>/meta/lerobot_annotations.json (non-_old only)
annotation_path = dataset_root.parent / dataset_no_old / 'meta' / 'lerobot_annotations.json'
instruction = _extract_instruction_from_annotation(annotation_path)
instruction_source = 'annotation_non_old' if instruction else 'template_fallback'

template_instruction = ''
cot_cfg_existing = data.get('cot', {})
if isinstance(cot_cfg_existing, dict):
  v = cot_cfg_existing.get('high_level_instruction', '')
  if isinstance(v, str):
    template_instruction = v.strip()

final_instruction = instruction.strip() if isinstance(instruction, str) and instruction.strip() else template_instruction

task_entry = {}
if tasks_file.exists():
  with tasks_file.open('r', encoding='utf-8') as f:
    task_map = yaml.safe_load(f) or {}
  task_entry = task_map.get(dataset_no_old, {}) or task_map.get(dataset_name, {}) or {}

objects = task_entry.get('objects', []) if isinstance(task_entry, dict) else []
if not isinstance(objects, list):
  objects = []

# dataset-specific prompt ids (if provided)
if isinstance(task_entry, dict):
  prompt_cfg = data.setdefault('prompt', {})
  for k in ('segment_task_id', 'vqa_task_id', 'cot_task_id'):
    v = task_entry.get(k)
    if isinstance(v, str) and v.strip():
      prompt_cfg[k] = v.strip()

# apply high-level instruction (annotation first, then template fallback)
if final_instruction:
  cot_cfg = data.setdefault('cot', {})
  cot_cfg['high_level_instruction'] = final_instruction

# build VQA task_context with instruction + object inventory
if final_instruction and objects:
  vqa_cfg = data.setdefault('vqa', {})
  vqa_cfg['task_context'] = _build_vqa_task_context(final_instruction, [str(x).strip() for x in objects if str(x).strip()])

with tmp_cfg.open('w', encoding='utf-8') as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)

print(f'[tmp_cfg] dataset={dataset_name}')
print(f'[tmp_cfg] root={dataset_root}')
print(f'[tmp_cfg] run.base_dir={run_base}')
print(f'[tmp_cfg] tasks_file={tasks_file}')
print(f'[tmp_cfg] annotation={annotation_path}')
print(f'[tmp_cfg] data_root={dataset_root}')
missing = '[missing]'
print(f'[tmp_cfg] instruction_source={instruction_source}')
print(f'[tmp_cfg] instruction={final_instruction if final_instruction else missing}')
print(f'[tmp_cfg] objects={objects if objects else missing}')
print(f'[tmp_cfg] path={tmp_cfg}')
PY"
}

discover_datasets() {
  local root="$1"
  local glob_pat="$2"
  if [ -z "$root" ]; then
    return 0
  fi
  find "$root" -mindepth 2 -maxdepth 2 -type d -name "$glob_pat" | sort
}

run_pipeline_for_cfg() {
  local cfg_path="$1"
  local ds_tag="$2"

  preflight_check "$cfg_path"

  local mode
  for mode in "${STAGES[@]}"; do
    local tag
    tag="${ds_tag}.${mode}"
    reset_done_markers "$mode" "$cfg_path"
    start_pair "$tag" "$cfg_path" "$mode"
    wait_idle_by_server_log "$tag"
    stop_worker "$tag"
    stop_server "$tag"
  done
}

parse_stages

echo "===== RUN CONFIG ====="
echo "CFG=$CFG"
echo "STAGES=${STAGES[*]}"
echo "BATCH_ROOT=${BATCH_ROOT:-<single-config-mode>}"
echo "DATASET_GLOB=$DATASET_GLOB"
echo "RUN_BASE_PARENT=$RUN_BASE_PARENT"
echo "TASKS_FILE=$TASKS_FILE"
echo "RESET_DONE=$RESET_DONE"
echo "DRY_RUN=$DRY_RUN"

if [ ! -f "$CFG" ]; then
  echo "Config not found: $CFG"
  exit 1
fi

if [ -n "$BATCH_ROOT" ]; then
  mapfile -t DATASET_LIST < <(discover_datasets "$BATCH_ROOT" "$DATASET_GLOB")
  if [ "${#DATASET_LIST[@]}" -eq 0 ]; then
    echo "No datasets found under $BATCH_ROOT with glob $DATASET_GLOB"
    exit 1
  fi

  echo "[batch] found ${#DATASET_LIST[@]} datasets"
  for ds in "${DATASET_LIST[@]}"; do
    echo "  - $ds"
  done

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] no execution"
    exit 0
  fi

  for dataset_root in "${DATASET_LIST[@]}"; do
    dataset_name="$(basename "$dataset_root")"
    dataset_no_old="${dataset_name%_old}"
    ds_tag="$dataset_no_old"
    tmp_cfg="/tmp/v2t.${dataset_no_old}.yaml"

    echo
    echo "================================================================"
    echo "[batch] dataset=$dataset_name"
    echo "================================================================"

    make_temp_config_for_dataset "$CFG" "$dataset_root" "$RUN_BASE_PARENT" "$tmp_cfg" "$TASKS_FILE"
    run_pipeline_for_cfg "$tmp_cfg" "$ds_tag"
  done
else
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] single-config-mode, no execution"
    exit 0
  fi
  run_pipeline_for_cfg "$CFG" "single"
fi

echo
echo "ALL DONE."
