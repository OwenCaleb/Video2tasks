#!/usr/bin/env bash
set -euo pipefail

# ================== CONFIG ==================
CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV="wallx"

VQA_CFG="config.vqa.yaml"
COT_CFG="config.yaml"
COT_MODE="cot"

LOGDIR="logs"
PIDDIR="pids"
POLL_SEC=2

# idle detection: MUST be consecutive
IDLE_N=100
IDLE_PAT='GET /get_job HTTP/1.1" 200 OK'
# ===========================================

mkdir -p "$LOGDIR" "$PIDDIR"

run_in_env() {
  local cmd="$1"
  bash -lc "source '$CONDA_SH' && conda activate '$CONDA_ENV' && $cmd"
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

# ---------------- Stage 1: VQA ----------------
start_pair "vqa" \
  "v2t-server --config $VQA_CFG" \
  "v2t-worker --config $VQA_CFG"

# 关键：不用等 worker 退出，而是等 server log 连续 100 次 get_job 200 OK
wait_idle_by_server_log "vqa"
stop_worker "vqa"
stop_server "vqa"

# ---------------- Stage 2: COT ----------------
start_pair "cot" \
  "v2t-server -c $COT_CFG --mode $COT_MODE" \
  "v2t-worker -c $COT_CFG --mode $COT_MODE"

# 如果你也希望 cot 用同样规则切换，就保持这一行；
# 如果 cot 是最终阶段不需要切换，可以把 wait+stop 注释掉，手动停。
wait_idle_by_server_log "cot"
stop_worker "cot"
stop_server "cot"

echo
echo "ALL DONE."
