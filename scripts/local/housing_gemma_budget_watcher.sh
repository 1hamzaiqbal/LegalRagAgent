#!/usr/bin/env bash
# Manage the detached HousingQA Gemma budget watcher.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs/monitors"
LOCK_ROOT="${LOCK_ROOT:-$LOG_DIR/locks}"
WATCH_LOCK_DIR="${WATCH_LOCK_DIR:-$LOCK_ROOT/housing_gemma_budget_watch.lock}"
ACTION="${1:-status}"

mkdir -p "$LOG_DIR" "$LOCK_ROOT"

read_meta() {
  local key="$1"
  local meta="$WATCH_LOCK_DIR/metadata"
  [[ -f "$meta" ]] || return 0
  awk -F= -v wanted="$key" '$1 == wanted {print substr($0, length($1) + 2); exit}' "$meta"
}

is_running() {
  local pid="${1:-}"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

print_metadata() {
  if [[ -f "$WATCH_LOCK_DIR/metadata" ]]; then
    sed 's/^/[watch] /' "$WATCH_LOCK_DIR/metadata"
  fi
}

watcher_pids() {
  ps -eo pid=,args= \
    | awk '$0 ~ /^[[:space:]]*[0-9]+[[:space:]]+bash scripts\/local\/watch_housing_gemma_until_ready\.sh([[:space:]]|$)/ {print $1}'
}

status() {
  if [[ ! -d "$WATCH_LOCK_DIR" ]]; then
    local pids
    pids="$(watcher_pids)"
    if [[ -n "$pids" ]]; then
      echo "Housing Gemma budget watcher process exists without lock: $pids"
      return 4
    fi
    echo "Housing Gemma budget watcher not running; lock absent: $WATCH_LOCK_DIR"
    return 1
  fi
  local pid
  pid="$(read_meta pid || true)"
  if is_running "$pid"; then
    echo "Housing Gemma budget watcher running: pid=$pid lock=$WATCH_LOCK_DIR"
    print_metadata
    return 0
  fi
  echo "Housing Gemma budget watcher lock is stale: pid=${pid:-unknown} lock=$WATCH_LOCK_DIR"
  print_metadata
  return 3
}

start() {
  local pid
  pid="$(read_meta pid || true)"
  if [[ -d "$WATCH_LOCK_DIR" ]] && is_running "$pid"; then
    echo "Housing Gemma budget watcher already running: pid=$pid lock=$WATCH_LOCK_DIR"
    print_metadata
    return 0
  fi
  if [[ -d "$WATCH_LOCK_DIR" ]]; then
    echo "Removing stale Housing Gemma budget watcher lock: $WATCH_LOCK_DIR"
    print_metadata
    rm -rf "$WATCH_LOCK_DIR"
  fi
  local pids
  pids="$(watcher_pids)"
  if [[ -n "$pids" ]]; then
    echo "Refusing to start: watcher process exists without a valid lock: $pids" >&2
    echo "Run '$0 stop' first to terminate untracked watcher processes." >&2
    return 4
  fi

  LAUNCH_ON_READY="${LAUNCH_ON_READY:-0}"
  INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"
  RUN_FINAL_GATE_AFTER_LAUNCH="${RUN_FINAL_GATE_AFTER_LAUNCH:-1}"

  setsid env \
    LAUNCH_ON_READY="$LAUNCH_ON_READY" \
    INTERVAL_SECONDS="$INTERVAL_SECONDS" \
    RUN_FINAL_GATE_AFTER_LAUNCH="$RUN_FINAL_GATE_AFTER_LAUNCH" \
    LOCK_ROOT="$LOCK_ROOT" \
    WATCH_LOCK_DIR="$WATCH_LOCK_DIR" \
    scripts/local/watch_housing_gemma_until_ready.sh \
    >> "$LOG_DIR/housing_gemma_budget_watch_launcher.out" 2>&1 < /dev/null &
  local started_pid="$!"
  for _ in {1..20}; do
    pid="$(read_meta pid || true)"
    if [[ -d "$WATCH_LOCK_DIR" ]] && is_running "$pid"; then
      echo "started Housing Gemma budget watcher: pid=$pid lock=$WATCH_LOCK_DIR"
      print_metadata
      return 0
    fi
    sleep 0.2
  done
  echo "failed to start Housing Gemma budget watcher; launcher pid=$started_pid; see $LOG_DIR/housing_gemma_budget_watch_launcher.out" >&2
  return 1
}

stop() {
  local pids
  if [[ ! -d "$WATCH_LOCK_DIR" ]]; then
    pids="$(watcher_pids)"
    if [[ -n "$pids" ]]; then
      echo "stopping untracked Housing Gemma budget watcher process(es): $pids"
      for pid in $pids; do
        kill -- "-$pid" 2>/dev/null || kill "$pid"
      done
      for _ in {1..30}; do
        pids="$(watcher_pids)"
        [[ -z "$pids" ]] && break
        sleep 0.2
      done
      pids="$(watcher_pids)"
      if [[ -n "$pids" ]]; then
        echo "failed to stop untracked watcher process(es): $pids" >&2
        return 1
      fi
    fi
    echo "Housing Gemma budget watcher not running; lock absent: $WATCH_LOCK_DIR"
    return 0
  fi
  local pid
  pid="$(read_meta pid || true)"
  if is_running "$pid"; then
    kill -- "-$pid" 2>/dev/null || kill "$pid"
    for _ in {1..30}; do
      if ! is_running "$pid"; then
        break
      fi
      sleep 0.2
    done
    if is_running "$pid"; then
      echo "failed to stop Housing Gemma budget watcher pid=$pid" >&2
      return 1
    fi
    echo "stopped Housing Gemma budget watcher: pid=$pid"
  else
    echo "removing stale Housing Gemma budget watcher lock: pid=${pid:-unknown}"
  fi
  rm -rf "$WATCH_LOCK_DIR"
}

case "$ACTION" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    stop
    start
    ;;
  status)
    status
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status}" >&2
    exit 2
    ;;
esac
