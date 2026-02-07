#!/usr/bin/env bash
set -euo pipefail

# ================== CONFIG ==================
CONDA_SH="/opt/conda/etc/profile.d/conda.sh"   # 视你的机器而定；若不对再改
CONDA_ENV="wallx"

VQA_CFG="config.vqa.yaml"
COT_CFG="config.yaml"
COT_MODE="cot"

LOGDIR="logs"
PIDDIR="pids"
POLL_SEC=5
# ===========================================

mkdir -p "$LOGDIR" "$PIDDIR"

run_in_env() {
  # 在指定 conda env 里执行命令
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

wait_worker_done() {
  local tag="$1"
  local wpid
  wpid="$(cat "$PIDDIR/v2t-worker.${tag}.pid")"
  echo
  echo "[${tag}] waiting worker pid=$wpid to finish..."

  while kill -0 "$wpid" 2>/dev/null; do
    sleep "$POLL_SEC"
  done

  echo "[${tag}] worker finished."
}

stop_server() {
  local tag="$1"
  local spid
  spid="$(cat "$PIDDIR/v2t-server.${tag}.pid")" || true
  echo "[${tag}] stopping server pid=$spid ..."
  kill "$spid" 2>/dev/null || true
  sleep 2
  kill -0 "$spid" 2>/dev/null && kill -9 "$spid" 2>/dev/null || true
  echo "[${tag}] server stopped."
}

# ---------------- Stage 1: VQA ----------------
start_pair "vqa" \
  "v2t-server --config $VQA_CFG" \
  "v2t-worker --config $VQA_CFG"

wait_worker_done "vqa"
stop_server "vqa"

# ---------------- Stage 2: COT ----------------
start_pair "cot" \
  "v2t-server -c $COT_CFG --mode $COT_MODE" \
  "v2t-worker -c $COT_CFG --mode $COT_MODE"

wait_worker_done "cot"
stop_server "cot"

echo
echo "ALL DONE."
