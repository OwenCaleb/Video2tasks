#!/usr/bin/env bash
set -euo pipefail

# ================== CONFIG ==================
# 应与 run_serial_gpu1.sh 中的配置保持一致
LOGDIR="logs"
PIDDIR="pids"

# ===========================================

echo "===== CLEANUP GPU1 PROCESSES ====="
echo "[cleanup] config: PIDDIR=$PIDDIR LOGDIR=$LOGDIR"

cleanup_pid_file() {
  local pid_file="$1"
  local proc_name="$2"

  if [ ! -f "$pid_file" ]; then
    return
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  echo "[$proc_name] pid_file=$pid_file pid=$pid"

  if [ -n "$pid" ]; then
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      if kill -0 "$pid" 2>/dev/null; then
        echo "[$proc_name] force killing pid=$pid"
        kill -9 "$pid" 2>/dev/null || true
      fi
    else
      echo "[$proc_name] pid=$pid already stopped"
    fi
  fi

  rm -f "$pid_file"
}

echo
echo "[cleanup] scanning pid files under $PIDDIR"
for f in "$PIDDIR"/v2t-server.*.pid; do
  [ -e "$f" ] || continue
  cleanup_pid_file "$f" "server"
done

for f in "$PIDDIR"/v2t-worker.*.pid; do
  [ -e "$f" ] || continue
  cleanup_pid_file "$f" "worker"
done

# Show remaining pids and logs
echo
echo "===== SUMMARY ====="
if [ -d "$PIDDIR" ]; then
  echo "[cleanup] remaining pid files in $PIDDIR:"
  find "$PIDDIR" -type f || echo "  (none)"
else
  echo "[cleanup] $PIDDIR does not exist"
fi

if [ -d "$LOGDIR" ]; then
  echo "[cleanup] log files in $LOGDIR:"
  ls -lh "$LOGDIR"/ 2>/dev/null || echo "  (none)"
  echo
  echo "[cleanup] to inspect logs, run:"
  echo "  tail -f $LOGDIR/v2t-server.*.log"
  echo "  tail -f $LOGDIR/v2t-worker.*.log"
else
  echo "[cleanup] $LOGDIR does not exist"
fi

echo
echo "===== CLEANUP COMPLETE ====="
