#!/usr/bin/env bash
set -euo pipefail

# ================== CONFIG ==================
# 应与 run_serial_gpu1.sh 中的配置保持一致
LOGDIR="logs"
PIDDIR="pids"
TAGS=("segment" "vqa" "cot")

# ===========================================

echo "===== CLEANUP GPU1 PROCESSES ====="
echo "[cleanup] config: PIDDIR=$PIDDIR LOGDIR=$LOGDIR"

cleanup_tag() {
  local tag="$1"
  
  echo
  echo "[${tag}] cleaning up..."
  
  # Kill worker
  local wpid_file="$PIDDIR/v2t-worker.${tag}.pid"
  if [ -f "$wpid_file" ]; then
    local wpid
    wpid="$(cat "$wpid_file" 2>/dev/null || true)"
    if [ -n "$wpid" ]; then
      echo "[${tag}] stopping worker pid=$wpid ..."
      if kill -0 "$wpid" 2>/dev/null; then
        kill "$wpid" 2>/dev/null || true
        sleep 1
        if kill -0 "$wpid" 2>/dev/null; then
          echo "[${tag}] force killing worker (SIGKILL)..."
          kill -9 "$wpid" 2>/dev/null || true
        fi
        echo "[${tag}] worker killed."
      else
        echo "[${tag}] worker pid=$wpid not running (already dead)."
      fi
    fi
    rm -f "$wpid_file"
    echo "[${tag}] removed $wpid_file"
  fi
  
  # Kill server
  local spid_file="$PIDDIR/v2t-server.${tag}.pid"
  if [ -f "$spid_file" ]; then
    local spid
    spid="$(cat "$spid_file" 2>/dev/null || true)"
    if [ -n "$spid" ]; then
      echo "[${tag}] stopping server pid=$spid ..."
      if kill -0 "$spid" 2>/dev/null; then
        kill "$spid" 2>/dev/null || true
        sleep 1
        if kill -0 "$spid" 2>/dev/null; then
          echo "[${tag}] force killing server (SIGKILL)..."
          kill -9 "$spid" 2>/dev/null || true
        fi
        echo "[${tag}] server killed."
      else
        echo "[${tag}] server pid=$spid not running (already dead)."
      fi
    fi
    rm -f "$spid_file"
    echo "[${tag}] removed $spid_file"
  fi
}

# Cleanup each tag
for tag in "${TAGS[@]}"; do
  cleanup_tag "$tag"
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
