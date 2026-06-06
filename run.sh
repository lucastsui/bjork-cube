#!/bin/zsh
# Launch the MRT2 demo webapp.
# Uses the playground venv where magenta-rt[mlx] + fastapi are installed.
cd "$(dirname "$0")"
source ~/code/playground/.venv/bin/activate
# Navigation ("take me somewhere…") uses Claude. The key lives in .env (gitignored,
# chmod 600); edit that file to change it. Falls back to an exported env var.
[ -f .env ] && source .env
[ -z "$ANTHROPIC_API_KEY" ] && echo "note: ANTHROPIC_API_KEY not set — pin/drift work, but 'go →' navigation needs it."
echo "Starting MRT2 demo at http://127.0.0.1:8000  (Ctrl-C to stop)"
exec uvicorn server:app --host 127.0.0.1 --port 8000
