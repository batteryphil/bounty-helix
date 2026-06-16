#!/usr/bin/env bash
# watchdog.sh — auto-restart bounty-helix if it dies
# Usage: nohup ./watchdog.sh >> logs/watchdog.log 2>&1 &

PROJ="/home/phil/.gemini/antigravity/scratch/analysis_project/bounty-helix"
VENV="/home/phil/.gemini/antigravity/scratch/analysis_project/titan_venv/bin/python"
LOG="$PROJ/logs/helix.log"
RESTART_LOG="$PROJ/logs/watchdog.log"
COOLDOWN=90   # seconds to wait before restart (let GPU memory fully clear)
CHECK_INTERVAL=60

echo "[$(date '+%F %T')] Watchdog started"

while true; do
    if ! pgrep -f "titan_venv.*main.py" > /dev/null 2>&1; then
        echo "[$(date '+%F %T')] bounty-helix died — restarting after ${COOLDOWN}s cooldown..."
        sleep $COOLDOWN

        # Free any stale GPU memory
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | tr -d ' ' | xargs -r kill -9 2>/dev/null
        sleep 3

        # Restart
        cd "$PROJ"
        nohup "$VENV" main.py >> "$LOG" 2>&1 &
        NEW_PID=$!
        echo "[$(date '+%F %T')] Restarted — PID $NEW_PID"
    fi
    sleep $CHECK_INTERVAL
done
