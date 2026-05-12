#!/usr/bin/env bash
# supervisor.sh — outer scheduler that keeps auto-kaggle running across crashes,
# context exhaustion, and quota-exhausted sleeps.
#
# Usage:
#   nohup bash supervisor.sh <comp_slug> [--engine claude|codex] \
#                                       [--poll-seconds 1800] \
#                                       [--max-runtime-hours 0] \
#                                       > supervisor.log 2>&1 &
#
# Behavior loop:
#   1. Check $RUN_DIR/STOP        → exit 0 if present.
#   2. Check $RUN_DIR/PAUSE       → sleep $POLL and loop.
#   3. Check $RUN_DIR/wait_until.txt → sleep until that UTC timestamp.
#   4. Check $RUN_DIR/.heartbeat  → if owned by a live pid and ts is fresh, sleep.
#   5. Otherwise, invoke the agent in headless mode with prompt
#      "/auto-kaggle resume <comp_slug>".
#   6. After the invocation returns, sleep $POLL and loop.
#
# Stops when:
#   - $RUN_DIR/STOP appears, OR
#   - --max-runtime-hours is exceeded, OR
#   - the comp deadline passes (read from $RUN_DIR/run.yaml).

set -uo pipefail

COMP_SLUG="${1:-}"
ENGINE="claude"
POLL_SECONDS=1800
MAX_RUNTIME_HOURS=0
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --max-runtime-hours) MAX_RUNTIME_HOURS="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$COMP_SLUG" ]]; then
  echo "usage: supervisor.sh <comp_slug> [--engine claude|codex] [--poll-seconds N] [--max-runtime-hours N]" >&2
  exit 2
fi

RUN_DIR="runs/${COMP_SLUG}"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "[supervisor] run dir does not exist: $RUN_DIR" >&2
  echo "[supervisor] start the first cycle interactively first:" >&2
  echo "    claude -p '/auto-kaggle $COMP_SLUG'" >&2
  exit 2
fi

START_TS_EPOCH=$(date -u +%s)
SUPERVISOR_PID=$$
echo "[supervisor] pid=$SUPERVISOR_PID comp=$COMP_SLUG engine=$ENGINE poll=${POLL_SECONDS}s"
echo "$SUPERVISOR_PID" > "$RUN_DIR/.supervisor.pid"
trap 'rm -f "$RUN_DIR/.supervisor.pid"; echo "[supervisor] exit"; exit 0' EXIT INT TERM

now_utc_epoch() { date -u +%s; }
iso_to_epoch() {
  # Portable ISO-8601 → epoch. Strips trailing Z, treats as UTC.
  local iso="${1%Z}"
  if date -u -d "$iso" +%s >/dev/null 2>&1; then
    date -u -d "$iso" +%s
  else
    # macOS BSD date
    date -u -j -f "%Y-%m-%dT%H:%M:%S" "$iso" +%s
  fi
}

pid_alive() {
  local p="$1"
  [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null
}

run_agent_once() {
  local prompt="/auto-kaggle resume ${COMP_SLUG}"
  case "$ENGINE" in
    claude)
      # Claude Code headless: `claude -p "<prompt>"` runs once, prints output, exits.
      claude -p "$prompt" || echo "[supervisor] claude exit code $?"
      ;;
    codex)
      codex run -- "$prompt" || echo "[supervisor] codex exit code $?"
      ;;
    *)
      echo "[supervisor] unknown engine: $ENGINE" >&2
      exit 2
      ;;
  esac
}

deadline_epoch() {
  # Best-effort read of deadline_utc from run.yaml. Returns empty string on failure.
  local rf="$RUN_DIR/run.yaml"
  [[ -f "$rf" ]] || { echo ""; return; }
  local line
  line=$(grep -E "^deadline_utc:" "$rf" | head -n1 | awk '{print $2}' | tr -d '"')
  [[ -n "$line" ]] && iso_to_epoch "$line" || echo ""
}

while true; do
  # 1. STOP sentinel
  if [[ -f "$RUN_DIR/STOP" ]]; then
    echo "[supervisor] STOP detected, exiting"
    exit 0
  fi

  # 2. max runtime
  if [[ "$MAX_RUNTIME_HOURS" -gt 0 ]]; then
    elapsed=$(( $(now_utc_epoch) - START_TS_EPOCH ))
    if (( elapsed > MAX_RUNTIME_HOURS * 3600 )); then
      echo "[supervisor] max runtime ${MAX_RUNTIME_HOURS}h reached, exiting"
      exit 0
    fi
  fi

  # 3. deadline
  dl=$(deadline_epoch)
  if [[ -n "$dl" ]] && (( $(now_utc_epoch) > dl )); then
    echo "[supervisor] competition deadline passed, exiting"
    exit 0
  fi

  # 4. PAUSE sentinel
  if [[ -f "$RUN_DIR/PAUSE" ]]; then
    echo "[supervisor] PAUSE present, sleeping ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
    continue
  fi

  # 5. wait_until.txt — sleep until then
  WAIT_FILE="$RUN_DIR/stage3_submit/wait_until.txt"
  if [[ -f "$WAIT_FILE" ]]; then
    wu=$(cat "$WAIT_FILE" 2>/dev/null | head -n1 | tr -d '[:space:]')
    if [[ -n "$wu" ]]; then
      wu_epoch=$(iso_to_epoch "$wu")
      now_epoch=$(now_utc_epoch)
      if (( wu_epoch > now_epoch )); then
        delta=$(( wu_epoch - now_epoch ))
        echo "[supervisor] wait_until=$wu, sleeping $delta s"
        sleep "$delta"
        continue
      fi
    fi
  fi

  # 6. Live agent already running? Refuse to start a second.
  HB="$RUN_DIR/.heartbeat"
  if [[ -f "$HB" ]]; then
    hb_pid=$(grep -oE '"pid"[[:space:]]*:[[:space:]]*[0-9]+' "$HB" 2>/dev/null | grep -oE '[0-9]+$' || true)
    hb_ts=$(grep -oE '"ts_utc"[[:space:]]*:[[:space:]]*"[^"]+"' "$HB" 2>/dev/null | sed -E 's/.*"([^"]+)"/\1/' || true)
    if [[ -n "$hb_pid" ]] && pid_alive "$hb_pid"; then
      if [[ -n "$hb_ts" ]]; then
        hb_age=$(( $(now_utc_epoch) - $(iso_to_epoch "$hb_ts") ))
        if (( hb_age < 300 )); then
          echo "[supervisor] live agent pid=$hb_pid (heartbeat ${hb_age}s old), waiting"
          sleep 60
          continue
        fi
      fi
    fi
  fi

  # 7. Invoke the agent once.
  echo "[supervisor] $(date -u +%FT%TZ) invoking $ENGINE"
  run_agent_once

  echo "[supervisor] cycle done, sleeping ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done
