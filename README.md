# Björk Cube — status quo / self-handoff

_Last updated: 2026-06-06. This file is a handoff so a fresh context can pick up fast. It documents the non-obvious stuff (architecture decisions, gotchas, current blockers) on top of what the code shows._

## What this is
A **local web app** that drives **Magenta RealTime 2** (MRT2) to generate a **continuous, live-steerable music stream**, wrapped in a full-screen "Google-Maps-style" UI built around a **navigable PCA hypercube of the latent music space**. Originally "MRT2 Demo", renamed **Björk Cube**.

- Folder: `~/Desktop/MRT2_demo/`
- Backend: `server.py` (FastAPI). Frontend: `static/index.html` (one file: HTML + CSS + vanilla JS, no build step).
- Model: `magenta-rt` **MLX backend**, `mrt2_base`, 8-bit. Runs locally on the M4 Max.

## Run it
```bash
~/Desktop/MRT2_demo/run.sh          # sources the venv + .env, starts uvicorn on 127.0.0.1:8000
# open http://127.0.0.1:8000
```
- **venv**: `~/code/playground/.venv` (has `magenta-rt[mlx]`, `fastapi`, `uvicorn`, `anthropic`, `soundfile`, `numpy 2.4.4`).
- **Manual restart** (what I do every time after a code change — the server runs detached and dies when the shell is reaped):
  ```bash
  pkill -f "uvicorn server:app"; sleep 2
  cd ~/Desktop/MRT2_demo && source ~/code/playground/.venv/bin/activate && set -a && source .env && set +a
  (uvicorn server:app --host 127.0.0.1 --port 8000 >/tmp/mrt2_server.log 2>&1 &)
  ```
- First **▶ Play** loads ~9 GB of weights into RAM (~a minute); after that it stays warm. `/status` shows `not_loaded` until then.
- Server log: `/tmp/mrt2_server.log`.

## Model weights
- Live at `~/Documents/Magenta/magenta-rt-v2/` (found via the `MAGENTA_HOME` default — no config needed).
- The pip lib loads `checkpoints/mrt2_base.safetensors` (**9.2 GB fp32**), downloaded from HF `google/magenta-realtime-2`. The bundled MRT2 apps' `mrt2_base.mlxfn` is a *different* (quantized) format the pip lib can't load — don't point at it.
- MusicCoCa (text/audio→768 embedding) + SpectroStream (audio codec) resources are in the same tree.

## Backend architecture (server.py) — the load-bearing decisions
- **Lazy model load**: `get_model()` builds `MagentaRT2System(size="mrt2_base", bits=8)` once on first use.
- **TWO MusicCoCa instances**: the generator's internal `mrt._style_model` AND a separate `_embed_mc` (`get_embed_mc()`) for the HTTP embedding endpoints. **Why**: `generate()` calls `_style_model.tokenize()` every step; sharing one tflite model across the gen thread + an embed thread corrupts it and silently drops the stream. Keep them separate.
- **Dedicated single-thread executors**: `_gen_executor` for generation, `_embed_executor` for embedding. **Why**: MLX streams are thread-local — running generation on the default thread pool throws `RuntimeError: no Stream(gpu) in current thread`. All generate() calls must stay on the one gen thread.
- **`_gen_lock`**: only one `/stream` at a time (the browser holds it; a second connection gets "Another stream is active"). When testing from a script, the user's open tab will block you — that's expected, don't fight it.
- **Streaming** (`/stream` WebSocket): loops `generate()`, chains `state`, sends raw float32 stereo PCM. **Adaptive controller**: sizes the free-run chunk + buffer lead from measured generation time, self-paces so latency doesn't grow, and emits `perf` (RTF) + warns when it can't keep up. Live control messages: `{type:"params"|"style"|"seq"|"underrun"|"stop"}`.
- **Audio style cap**: `STYLE_AUDIO_MAX_SEC = 20` — only the first 20 s of an uploaded clip is embedded (keeps embedding fast; a long clip otherwise starves generation).
- **Persisted on disk**: `landmarks.json` (user "pins": description + 768-d embedding), `vocab.json` (104 instrument/genre embeddings).

### Endpoints
| Endpoint | Purpose |
|---|---|
| `GET /`, `GET /status` | page, model state |
| `POST /prepare_style` | weighted mix of text/audio inputs → cached style `token` (+ embedding) |
| `POST /embedding` | same mix, preview only (no caching) |
| `POST /prepare_raw` | a raw 768-d vector → cached `token` (used by PCA sliders, drift, dot-clicks) |
| `POST /pca` | PCA of the preset vectors → `{mean, components, ranges, coords, explained, k}` (k = N−1) |
| `GET/POST/DELETE /landmark…`, `POST /landmark/{id}/play` | pins CRUD + play a pin |
| `POST /navigate` | feeling → **Claude** strict tool use → target embedding |
| `GET /vocab`, `GET /vocab_points` | vocab words; words+embeddings for cube dots |
| `WS /stream` | the live audio stream |
| `POST /generate` | one-shot fixed-length clip (legacy, still works) |

## Claude navigation (the "small LLM" layer)
- `POST /navigate`: sends the user's request + their landmark descriptions + the vocab word list to **`claude-opus-4-8`** (anthropic `AsyncAnthropic`), **strict tool use**, must call exactly one of:
  `go_to_landmark` · `blend_landmarks` · `compose_from_vocab` · `rewrite_to_music`. The server resolves the choice to a 768-d target embedding (Claude never emits coordinates).
- **API key** is in `.env` (`ANTHROPIC_API_KEY=…`, chmod 600, gitignored), sourced by `run.sh`.
- ⚠️ **CURRENT BLOCKER**: the key is valid but the **Anthropic account has $0 API credits**, so `/navigate` returns `400 "Your credit balance is too low…"`. Everything else works. The error is surfaced verbatim in the UI status line. Once credits are added, `go →` works with no code change. (Note: a Claude.ai/Code subscription ≠ API credits.)

## Frontend (static/index.html) — full-screen "map" layout
- **Top bar**: "Björk Cube" + ▶/⏸ + status **dot** + the embedding **wave chart** (`#embChart`, lives here now) + log/perf.
- **Full-screen hypercube** (`#pcaCube`) is the base layer: **drag empty space = rotate** (Shift-drag = 4th dim), **pinch/scroll = zoom**, **double-click = reset zoom**, **drag the bright dot = move the current position**, **hover any dot = its name**, **click a preset/vocab dot = glide there**. Only appears with **≥2 presets** (PCA needs them).
- **Corner overlay cards** (translucent, always visible):
  - **top-left — Style**: list of inputs, each stacked two lines — `[prompt/audio ▾] [input]` then `[weight ──○──] [×]`. Buttons: `+ add input`, `★ save preset`, `I'm feeling lucky` (appends 3–6 random presets from mood+genre+instrument).
  - **bottom-left — Ingredients / Presets / Wander & navigate**: "Ingredients" = the floating instrument/genre **word bubbles** (drag onto a prompt to append it, onto empty Style space to make a new prompt, onto an audio input = nothing). Presets = saved snapshots (cards w/ mini chart; click is NOT wired — pins are the clickable thing). Wander = `slow drift` toggle, `wander speed` slider, `show vocabulary on cube` toggle, `📍 pin` (describe current spot → landmark), `go →` (Claude navigate), landmark chips (**click to play**, × to delete).
  - **bottom-right — Sampling & guidance** (temperature, top_k, cfg·style/notes/drums) + **PCA explorer** sliders (one per principal axis).
  - **bottom-center — Step sequencer** (notes/drums grid): **collapsible, default collapsed**; header arrow shows the next action (▼ = will expand, ▲ = will collapse).
- **Travel**: navigation and dot-clicks **glide** (~3 s, 8 steps) to the target; `slow drift` autonomously wanders (scaled by the wander-speed slider).
- **Layout note**: columns are gone — it's `position:fixed` overlays over a full-screen canvas. Top/bottom cards are capped to half the screen height per side so they don't overlap vertically. On short windows the bottom-center card can overlap the bottom corners (mitigated by collapsing the sequencer).

## Verification habits
- JS: extract `<script>` and `node --check`. Server: `python -c "import ast; ast.parse(open('server.py').read())"`.
- Endpoints: `curl` them. Live stream / pin-play / etc. can't be tested while the browser holds `_gen_lock`.

## Files
```
server.py            FastAPI backend (model, streaming, embeddings, PCA, landmarks, navigate, vocab)
static/index.html    entire UI (HTML+CSS+JS)
run.sh               launcher (sources venv + .env)
.env                 ANTHROPIC_API_KEY (chmod 600, gitignored)
.gitignore           (.env)
landmarks.json       user pins (description + 768-d embedding)   ← persists
vocab.json           104 instrument/genre embeddings              ← persists, rebuilt if deleted
amen_break.mp3, "Magnolian - Indigo (Official Video).mp3", "Suits Maps And Guns.mp3"   audio for style input
uploads/, outputs/   scratch dirs
```

## Open items / next steps
- **Add Anthropic API credits** to unblock `go →` navigation (only blocker).
- **Not a git repo / no remote.** The user asked to commit+push; nothing is initialized here yet. If asked again: `git init` here, then create a GitHub repo (gh is authed as `lucastsui`) and push — confirm name/visibility first. (Don't conflate with `~/Desktop/Music_Hackathon`, a *separate* project — "morphing-groove-map" — which has its own `status_quo.md` and remote.)
- Polish ideas raised but not done: smarter hover-label placement near edges; preset cards clickable-to-play; project the prompt-mix position as its own cube dot; persist presets (currently session-only; pins/vocab persist).
