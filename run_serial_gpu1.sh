#!/usr/bin/env bash
set -euo pipefail

# ================== CONFIG ==================
CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV="wallx"

CFG="config.yaml"
COT_MODE="cot"
RESET_DONE=1

LOGDIR="logs"
PIDDIR="pids"
POLL_SEC=2

# idle detection: MUST be consecutive
IDLE_N=100
IDLE_PAT='GET /get_job HTTP/1.1" 200 OK'
# ===========================================

mkdir -p "$LOGDIR" "$PIDDIR"

# 若中断脚本（Ctrl+C），自动清理进程
trap 'echo; echo "[trap] keyboard interrupt, cleaning up..."; cleanup; exit 130' INT TERM

run_in_env() {
  local cmd="$1"
  bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && $cmd"
}

cleanup() {
  echo
  echo "===== MANUAL CLEANUP ====="
  echo "[cleanup] killing any remaining processes..."
  for tag in "segment" "vqa" "cot"; do
    local wpid
    local spid
    wpid="$(cat "$PIDDIR/v2t-worker.${tag}.pid" 2>/dev/null || true)"
    spid="$(cat "$PIDDIR/v2t-server.${tag}.pid" 2>/dev/null || true)"
    
    [ -n "$wpid" ] && kill -9 "$wpid" 2>/dev/null || true
    [ -n "$spid" ] && kill -9 "$spid" 2>/dev/null || true
    
    rm -f "$PIDDIR/v2t-worker.${tag}.pid" "$PIDDIR/v2t-server.${tag}.pid"
  done
  echo "[cleanup] done. see logs in: $LOGDIR/"
}

preflight_check() {
  echo
  echo "===== PREFLIGHT ====="
  run_in_env "v2t-validate -c '$CFG'"

  CFG_PATH="$CFG" bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && python - <<'PY'
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
  if [ "$RESET_DONE" -ne 1 ]; then
    return
  fi

  echo "[${mode}] RESET_DONE=1, removing stale .DONE markers..."
  MODE="$mode" CFG_PATH="$CFG" bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && python - <<'PY'
from pathlib import Path
import os
from video2tasks.config import Config

cfg = Config.from_yaml(Path(os.environ['CFG_PATH']))
mode = os.environ['MODE']
removed = 0

for ds in cfg.datasets:
    # segment/cot use video_subset, vqa uses frame_subset
    if mode == 'vqa':
        subset = ds.frame_subset
    else:
        subset = ds.video_subset
    
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
  local server_cmd="$2"
  local worker_cmd="$3"

  echo
  echo "===== START ${tag} ====="

  nohup bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && $server_cmd" \
    > "$LOGDIR/v2t-server.${tag}.log" 2>&1 & echo $! > "$PIDDIR/v2t-server.${tag}.pid"
  echo "[${tag}] server pid=$(cat "$PIDDIR/v2t-server.${tag}.pid")"

  nohup bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && $worker_cmd" \
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
  echo "[${tag}] server stopped."
}

wait_idle_by_server_log() {
  local tag="$1"
  local logfile="$LOGDIR/v2t-server.${tag}.log"

  echo
  echo "[${tag}] waiting for ${IDLE_N} consecutive lines matching: ${IDLE_PAT}"
  echo "[${tag}] logfile: $logfile"

  while true; do
    # 如果 server 已死，直接报错退出
    local spid
    spid="$(cat "$PIDDIR/v2t-server.${tag}.pid" 2>/dev/null || true)"
    if [ -n "$spid" ] && ! kill -0 "$spid" 2>/dev/null; then
      echo "[${tag}] ERROR: server pid=$spid is not running, cannot wait idle."
      exit 1
    fi

    if [ -f "$logfile" ]; then
      # 必须是“连续”：最后 IDLE_N 行全部匹配
      local ok
      ok=$(tail -n "$IDLE_N" "$logfile" 2>/dev/null | grep -c "$IDLE_PAT" || true)

      # 可选：打印进度（不会太吵）
      echo "[${tag}] idle_check ok=$ok/$IDLE_N"

      if [ "$ok" -eq "$IDLE_N" ]; then
        echo "[${tag}] IDLE reached: last ${IDLE_N} lines are get_job 200 OK."
        break
      fi
    fi

    sleep "$POLL_SEC"
  done
}

# ============== MAIN PIPELINE ==============

preflight_check

# ========== Stage 1: SEGMENT ==========
# Segment mode: detect actions/switches, split videos into clips
reset_done_markers "segment"

start_pair "segment" \
  "v2t-server -c $CFG --mode segment" \
  "v2t-worker -c $CFG --mode segment"

# Wait for segment completion
wait_idle_by_server_log "segment"
stop_worker "segment"
stop_server "segment"

# ========== Stage 2: VQA ==========
# VQA mode: generate questions on extracted frames
reset_done_markers "vqa"

start_pair "vqa" \
  "v2t-server -c $CFG --mode vqa" \
  "v2t-worker -c $CFG --mode vqa"

# Wait for VQA completion
wait_idle_by_server_log "vqa"
stop_worker "vqa"
stop_server "vqa"

# ========== Stage 3: COT ==========
reset_done_markers "cot"

start_pair "cot" \
  "v2t-server -c $CFG --mode $COT_MODE" \
  "v2t-worker -c $CFG --mode $COT_MODE"

# 如果你也希望 cot 用同样规则切换，就保持这一行；
# 如果 cot 是最终阶段不需要切换，可以把 wait+stop 注释掉，手动停。
wait_idle_by_server_log "cot"
stop_worker "cot"
stop_server "cot"

echo
echo "ALL DONE."
