#!/bin/zsh
# Launch the Björk Cube webapp.
# Prefer a project-local .venv (off-the-shelf clones: python -m venv .venv &&
# pip install -r requirements.txt); fall back to the playground venv.
cd "$(dirname "$0")"
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  source ~/code/playground/.venv/bin/activate
fi
# Model weights + code are in-project; server.py sets MAGENTA_HOME=./magenta_home.
# Navigation ("take me somewhere…") uses Claude. The key lives in .env (gitignored,
# chmod 600); edit that file to change it. Falls back to an exported env var.
[ -f .env ] && source .env
[ -z "$ANTHROPIC_API_KEY" ] && echo "note: ANTHROPIC_API_KEY not set — pin/drift work, but 'go →' navigation needs it."
echo "Starting MRT2 demo at http://127.0.0.1:8000  (Ctrl-C to stop)"
exec uvicorn server:app --host 127.0.0.1 --port 8000
