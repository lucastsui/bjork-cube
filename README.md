# Björk Cube — status quo / self-handoff

_Last updated: 2026-06-06. A handoff so a fresh context can resume fast. Documents the non-obvious stuff (architecture, gotchas, current state) on top of what the code shows._

## What this is
A **local web app** driving **Magenta RealTime 2** (MRT2, `magenta-rt` MLX, `mrt2_base`, 8-bit) for a **continuous, live-steerable music stream**, presented as a **full-screen navigable PCA hypercube** of the latent style space — plus an **embedded groove toolkit** (the Morphing Groove Map) and several steering layers (tempo, danmaku suggestions, landmarks).

- Folder: `~/Desktop/MRT2_demo/`. Backend `server.py` (FastAPI). Frontend `static/index.html` (one file: HTML+CSS+vanilla JS, no build). Standalone Groove Cube page `static/groove_cube.html`.
- GitHub (public): **github.com/lucastsui/bjork-cube**. Model weights: `~/Documents/Magenta/magenta-rt-v2/` (`checkpoints/mrt2_base.safetensors`, 9.2 GB; the `.mlxfn` is a different format the pip lib can't load).

## Run / deploy
```bash
~/Desktop/MRT2_demo/run.sh           # sources venv + .env, uvicorn on 127.0.0.1:8000
```
- venv: `~/code/playground/.venv` (magenta-rt[mlx], fastapi, uvicorn, anthropic, soundfile, numpy).
- **Restart after a server change** (I do this constantly):
  ```bash
  pkill -f "uvicorn server:app"; sleep 2
  cd ~/Desktop/MRT2_demo && source ~/code/playground/.venv/bin/activate && set -a && source .env && set +a
  (uvicorn server:app --host 127.0.0.1 --port 8000 >/tmp/mrt2_server.log 2>&1 &)
  ```
- Frontend-only edits need **no restart** (served from disk) — just reload. Log: `/tmp/mrt2_server.log`.
- **Public deploy:** Tailscale Funnel exposes it at **https://tsuis-macbook-pro.tail2214e5.ts.net** (public, NO auth — anyone can use the model + spend Claude credits). Start: `tailscale funnel --bg 8000`; stop: `tailscale funnel --https=443 off`. Funnel config persists across reboot; **uvicorn does not** (rerun run.sh). Mac must stay awake.

## Current state / blockers
- **Anthropic API credits are LIVE** (as of 2026-06-06) — `/navigate` ("go →") and `/danmaku` both call Claude (`claude-opus-4-8`) for real. (Earlier the account was at $0; that's resolved.)
- **Single stream at a time** (`_gen_lock`): a second `/stream` gets "Another stream is active." Testing from a script is blocked while the browser holds it.

## Backend (server.py) — load-bearing decisions
- **Lazy model load**: `get_model()` builds `MagentaRT2System(size="mrt2_base", bits=8)` once; loaded off the event loop inside the stream's try/finally.
- **TWO MusicCoCa instances**: generator's internal `mrt._style_model` + a separate `_embed_mc` (`get_embed_mc()`) for HTTP embedding. Sharing one tflite across the gen thread + an embed thread corrupts it and drops the stream. Keep separate.
- **Dedicated single-thread executors**: `_gen_executor` (generation) + `_embed_executor` (embedding). MLX streams are thread-local → generation MUST stay on one thread (else "no Stream(gpu) in current thread").
- **Stream lock never leaks**: model load + setup are INSIDE the try whose `finally` releases `_gen_lock` (a past bug leaked it and wedged all future streams → UI showed "Paused"; fixed).
- **Adaptive streaming** (`/stream`): loops generate(), chains state, sends float32 stereo PCM; sizes chunk + buffer lead from measured gen time; emits `perf`. Live control msgs: `params|style|seq|underrun|stop`.
- **STYLE_AUDIO_MAX_SEC=20**: only first 20 s of an uploaded clip is embedded.
- **`/pca` clamps `k = min(n-1, dims)`** (a bug crashed it when #vectors > dims, e.g. 38 grooves × 32 dims).
- **Persisted:** `landmarks.json` (pins: description + 768-d embedding + `polarity` good/bad; gitignored), `vocab.json` (104 instrument/genre embeddings; committed).

### Endpoints
| Endpoint | Purpose |
|---|---|
| `GET /`, `GET /groovecube`, `GET /status` | main page, standalone Groove Cube, model state |
| `POST /prepare_style`, `POST /embedding` | weighted mix of text/audio inputs → token / preview. **Accept `tempo_bpm` + `tempo_weight`**: server loads `tempo_refs/beat_<bpm>bpm.wav`, embeds (cached), mixes at that weight (0 = off). |
| `POST /prepare_raw`, `POST /pca` | raw 768-d → token; PCA of vectors |
| `GET/POST/DELETE /landmark…`, `POST /landmark/{id}/play` | pins CRUD (POST takes `polarity`) |
| `POST /navigate` | feeling → Claude tool-use → target embedding; **soft-repels away from 👎 (bad) landmarks** (`_repel_from_bad`) |
| `POST /danmaku` | short Gen-Z **suggestions** to steer the music (Claude + per-call angle for variety; built-in fallback pool) |
| `GET /vocab`, `GET /vocab_points`, `GET /agr_list` | vocab words/embeddings; list of `agr/` files |
| `WS /stream` | live audio (client auto-reconnects on drop) |
| mounts: `/assets` (groove_dist), `/agr` (agr/), `/static`; routes serve `/groove` + 5 root files (straight_drums.wav, amen.wav, demoSongA/B.wav, groove_library.json) | |

## Frontend (static/index.html) — full-screen layout
- **Top bar:** "Björk Cube" + **"comments"** toggle (danmaku) + ▶/⏸ + status dot + `embChart` (style-embedding wave) + `waveTape` (live output wave) + log/perf.
- **Full-screen hypercube** (`#pcaCube`, base layer): drag empty = rotate (Shift = 4th dim), pinch/scroll = zoom, dblclick = reset, **drag the bright dot = move current position**, hover dots = names, **click a preset/vocab/landmark dot = glide there**. A fading **trail** marks travel. **Cube-dims slider** (PCA card) shows 1..min(presets−1, 12) dims. Needs ≥2 presets to appear.
  - **Grabbing the current dot cancels any in-progress glide/nav and pauses drift** (press-drag overrides travel).
- **Left pane** (`ov-left`, flush to top bar + left/bottom edges): **tempo (bpm)** slider + **tempo weight** slider (0 = off) → mixes a kick-beat ref as an audio prompt; **Style** (mix prompts/audio, each row two-line [type][input]/[weight][×]; audio rows are **drag-and-drop** dropzones); **Ingredients** (draggable word bubbles → Style); **Presets** (save/restore + "I'm feeling lucky"); **Wander & navigate** (drift toggle, **travel speed** slider [scales drift AND glides], **👍/👎 pin** [good/bad landmarks], "go →" Claude nav, landmark chips — 👎 shown red, click to play).
- **Right pane** (`ov-right`, flush right): top half = **Groove map** (iframe `/groove`, the embedded Morphing Groove Map, ⤢ expandable); bottom (scroll) = **Sampling & guidance** (temp/top_k/cfg) + **PCA explorer** (cube-dims + per-PC sliders).
- **Bottom-center:** **Groove instrument** panel (collapsible). A 2D **morph pad** (groove-library PCA, top-2 PCs) + a synth **drum loop** (play/BPM/pattern) + a **live per-16th strip** (violet=late, coral=early, green=velocity, playhead) + groove-library chips + **`.agr` presets** (click → parse the Ableton `.agr` client-side [DecompressionStream+DOMParser], fold to 16 slots, set the bars as an override). All in an **IIFE with its own AudioContext** — isolated from the MRT2 stream.
- **Danmaku** overlay: top, **3 rows**, **behind the side cards** (they occlude), comments are **clickable → fill "go →" and navigate**. Toggle is the top-left "comments".

## Files
```
server.py                 backend
static/index.html         main app (HTML+CSS+JS)
static/groove_cube.html   standalone Groove Cube (/groovecube)
groove_dist/              built Morphing Groove Map (embedded iframe) + its wav/json (committed)
tempo_refs/beat_*.wav     151 kick beats 50-200 BPM (GITIGNORED, ~133MB) — regen: python tools/make_tempo_beats.py
agr/*.agr                 12 Ableton groove files (served at /agr, parsed client-side)
tools/make_tempo_beats.py regenerates tempo_refs
landmarks.json            pins (gitignored)   vocab.json  embedding cache (committed)
.env                      ANTHROPIC_API_KEY (chmod 600, gitignored)
run.sh                    launcher
```
`.gitignore` excludes: `.env`, `__pycache__`, `outputs/`, `uploads/`, `*.mp3`, `landmarks.json`, `*.bak`, `tempo_refs/`, `.DS_Store`. (Large `.mp3` style clips live in the folder, gitignored.)

## Concepts that took iteration to get right
- **Tempo:** MRT2 has **no BPM input** (verified — no tempo/velocity params; note control is a 4-state pianoroll, drums binary). Tempo is steered *softly* by feeding a clean kick-beat reference (`tempo_refs/`) as an audio style prompt; the tempo slider picks the BPM file, the weight slider sets its mix strength. Genre/percussiveness transfers too.
- **.stt / .agr ↔ the bars:** an `.agr` (Ableton MIDI-clip groove) folds onto the 16-slot strip (timing ms + velocity per 16th). MRT2 itself can't apply microtiming (its control grid is 25 fps); the groove engine works on a *fixed beat* (known onsets), which is why the Groove instrument is its own synth loop, not MRT2.
- **Two cubes:** the main cube = MusicCoCa **style** space (drives MRT2). The Groove instrument's pad = a separate **groove** PCA space (timing+velocity of groove files). They are NOT interchangeable.

## Verification habits
- JS: extract `<script>` and `node --check`. Server: `python -c "import ast; ast.parse(open('server.py').read())"`.
- `curl` endpoints. Live stream / pin-play can't be tested while the browser holds `_gen_lock`. Playwright MCP is often locked by the user's open browser.

## Open ideas (not done)
- Tempo strength is now a slider (done). Could add a strength/off for the danmaku-driven navigation.
- Danmaku reacts to the prompt *text*, not the live cube position — could feed the current nearest-vocab words instead.
- `.agr` import only in the inline Groove instrument + the embedded Groove map; not wired to drive MRT2 generation (groove vs style domains).
