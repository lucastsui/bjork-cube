#!/bin/zsh
# Keep the Björk Cube server up for the public Tailscale Funnel.
# The MLX/Metal backend can hard-abort (e.g. "[metal::malloc] Resource limit
# exceeded") which kills the whole process — uvicorn can't catch that. This loop
# relaunches it. Run under `caffeinate` so the Mac doesn't sleep (sleep -> 502).
#
#   caffeinate -dis nohup ~/Desktop/MRT2_demo/serve_keepalive.sh >/dev/null 2>&1 &
#
# Stop it:  pkill -f serve_keepalive ; pkill -f "uvicorn server:app" ; pkill caffeinate
cd ~/Desktop/MRT2_demo || exit 1
source ~/code/playground/.venv/bin/activate
set -a; [ -f .env ] && source .env; set +a
LOG=/tmp/mrt2_server.log
while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting uvicorn" >> "$LOG"
  uvicorn server:app --host 127.0.0.1 --port 8000 >> "$LOG" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] uvicorn exited (code $?) — restarting in 2s" >> "$LOG"
  sleep 2
done
