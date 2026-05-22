#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs/monitors"
PID_FILE="$LOG_DIR/current_status_monitor.pid"
LOG_FILE="$LOG_DIR/current_status_monitor.log"
INTERVAL="${INTERVAL:-300}"
ACTION="${1:-status}"

mkdir -p "$LOG_DIR"

read_pid() {
  if [[ -f "$PID_FILE" ]]; then
    tr -d '[:space:]' < "$PID_FILE"
  fi
}

is_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

case "$ACTION" in
  start)
    pid="$(read_pid || true)"
    if is_running "$pid"; then
      echo "current_status monitor already running: pid=$pid"
      exit 0
    fi
    rm -f "$PID_FILE"
    loop='echo "$$" > "$1"; while true; do CURRENT_STATUS_MONITOR=1 python3 scripts/update_current_status.py --interval "$2"; sleep "$2"; done'
    if command -v setsid >/dev/null 2>&1; then
      setsid bash -c "$loop" monitor "$PID_FILE" "$INTERVAL" >> "$LOG_FILE" 2>&1 < /dev/null &
    else
      nohup bash -c "$loop" monitor "$PID_FILE" "$INTERVAL" >> "$LOG_FILE" 2>&1 < /dev/null &
    fi
    for _ in {1..20}; do
      pid="$(read_pid || true)"
      if is_running "$pid"; then
        break
      fi
      sleep 0.2
    done
    if ! is_running "$pid"; then
      echo "failed to start current_status monitor; see $LOG_FILE" >&2
      exit 1
    fi
    echo "started current_status monitor: pid=$pid interval=${INTERVAL}s log=$LOG_FILE"
    ;;
  stop)
    pid="$(read_pid || true)"
    if ! is_running "$pid"; then
      rm -f "$PID_FILE"
      echo "current_status monitor not running"
      exit 0
    fi
    kill "$pid"
    rm -f "$PID_FILE"
    echo "stopped current_status monitor: pid=$pid"
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    pid="$(read_pid || true)"
    if is_running "$pid"; then
      echo "current_status monitor running: pid=$pid interval=${INTERVAL}s log=$LOG_FILE"
      exit 0
    fi
    echo "current_status monitor not running"
    exit 1
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status}" >&2
    exit 2
    ;;
esac
