#!/bin/bash
# Bounty-Helix Launcher
# Starts the agent and dashboard

PROJ="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJ/venv/bin/python"

# Fall back to system python if no venv
if [ ! -f "$VENV" ]; then
    VENV="python3"
fi

mkdir -p "$PROJ/logs" "$PROJ/data"

echo "🚀 Starting Bounty-Helix..."

nohup "$VENV" "$PROJ/dashboard/dashboard.py" >> "$PROJ/logs/dashboard.log" 2>&1 &
echo "   Dashboard PID: $! — http://localhost:5050"

sleep 1

nohup "$VENV" "$PROJ/main.py" >> "$PROJ/logs/helix.log" 2>&1 &
echo "   Agent PID:     $!"

echo ""
echo "✅ Running. Monitor at http://localhost:5050"
echo "   Logs: tail -f logs/helix.log"
echo ""
echo "To stop: pkill -f 'main.py|dashboard.py'"
