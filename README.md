<img width="1728" height="1117" alt="Screenshot 2026-06-09 at 12 06 44 PM" src="https://github.com/user-attachments/assets/1da297cf-56fa-4a5a-8388-7ef4ecc6d173" />

# Bjork-Cube

A local web app that turns **Magenta RealTime 2** (MRT2) into a continuous, live-steerable music stream you navigate as a full-screen PCA hypercube of the model's latent style space — playable with a step sequencer, a groove instrument, a MIDI controller, real-time glitch FX, an autonomous scout, and a phone audience-participation channel.

- **UI title:** "Morphing Music Map" · **browser tab:** "Music Morphing Space" · **repo:** `bjork-cube` (formerly "Björk Cube")
- **GitHub (public):** https://github.com/lucastsui/bjork-cube
- The app is **self-contained** — no external service dependencies, and the **Magenta model code and weights ship inside the project** (`magenta_rt/` + `magenta_home/`), so it runs off the shelf. (An earlier "Groove map" panel that called out to a DGX Spark groove service has also been removed.)

---

## Introduction

At its core the app runs MRT2 (`magenta-rt` on MLX, `mrt2_base`, 8-bit) to generate an endless audio stream and lets you steer it in real time. The model's style space is a 768-dimensional MusicCoCa embedding; the app projects it to a navigable **PCA hypercube** and renders it full-screen. The bright dot is "where you are" in style space; dragging it, clicking other points, or letting the autonomous scout wander all change the music as it plays.

Layered on top of the cube are several ways to perform and shape the stream:

- **Step sequencer** — a note grid (C1–C5) + drum row that conditions MRT2 via `seq` (25 fps onset conditioning) with integer-precise BPM.
- **Groove instrument** — a morph pad over a separate groove (timing+velocity) PCA, a synth drum loop, and `.agr` Ableton-groove presets; can bounce its loop to a WAV and feed it back as a style prompt.
- **MIDI control** — Web MIDI (built for an Arturia MiniLab 3): keys play notes the model follows; faders crossfade prompt weights / travel speed / glitch / A→B; pads jump to nearest/furthest/random points.
- **Real-time glitch FX** — a client-side bitcrusher on the audio stream (slider or MIDI CC1), no server or model change.
- **Autonomous agents + breadcrumbs** — a "Wanderer 2" scout that drifts on its own and drops LLM-named "feeling" landmarks.
- **Phone audience channel** — audience members scan a QR, open `/m` on their phones, and leave "feelings" (persistent breadcrumbs the navigation LLM can steer toward) and "steer the course" comments (danmaku on the big screen), pushed live over SSE.
- **Geometric + LLM navigation** — type a destination: literal cube coordinates (`corner`, `vertex`, `edge`, `center`…) move geometrically; anything else is interpreted musically by Claude.
- **Accessibility layer** — ARIA names on every control, a plain-language "now: …" readout, screen-reader-friendly thumbs, dark scrollbars.

Claude (`claude-opus-4-8`) powers the `/navigate`, `/feeling`, and `/danmaku` endpoints, so an Anthropic API key is required for those features (drift, pinning, and direct cube navigation work without it).

---

## Installation

### Requirements

- **Apple Silicon Mac** — the model runs on MLX / Metal.
- **Python 3.12.**
- An **Anthropic API key** (optional, but needed for "go →" navigation, feelings, and danmaku).

The Magenta **model code is vendored** in `magenta_rt/` (committed) and the **weights live in `magenta_home/`** (`magenta-rt-v2/{checkpoints,models,resources}`, ~13 GB). `server.py` sets `MAGENTA_HOME` to that in-project folder automatically, so there is no dependency on `~/Documents/Magenta`.

### 1. Set up the environment

```bash
cd ~/Desktop/MRT2_demo
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the model's third-party runtime deps (MLX, `ai-edge-litert`, `librosa`, …) plus the web-server deps. `segno` (pure-Python QR for the audience join code) is included; if it were missing, the QR endpoint falls back to rendering the URL as text. The existing `~/code/playground/.venv` already has everything, and `run.sh` uses it automatically when no local `.venv` is present.

### 1a. Weights (fresh clones only)

The ~13 GB of weights are **gitignored** (too large for GitHub), so a fresh `git clone` does **not** include them. On this machine they are already in place at `magenta_home/magenta-rt-v2/`. To populate a fresh clone, copy them from an existing install:

```bash
mkdir -p magenta_home
cp -Rc ~/Documents/Magenta/magenta-rt-v2 magenta_home/   # -c = instant APFS clone, no extra disk
```

(or re-download via the Magenta sample-app tooling, then point `MAGENTA_HOME` at wherever they landed).

### 2. Configure secrets

Create a `.env` (gitignored, `chmod 600`) in the project root:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # for navigation / feelings / danmaku
AUDIENCE_ADMIN_TOKEN=...                    # token to open/close the audience channel
PUBLIC_URL=https://<your-tailscale-host>   # base URL the audience QR encodes
```

### 3. Run locally

```bash
~/Desktop/MRT2_demo/run.sh        # activates the venv, sources .env, starts uvicorn
```

Then open **http://127.0.0.1:8000**. Frontend-only edits need no restart — just reload. After a *server* change, restart with a force kill (see note below):

```bash
pkill -9 -f "uvicorn server:app"
```

### 4. Keep-alive / public deploy (optional)

The MLX/Metal backend can hard-abort (e.g. `[metal::malloc] Resource limit exceeded`), which kills the whole process — uncatchable from Python. `serve_keepalive.sh` relaunches uvicorn whenever it exits; run it under `caffeinate` so the Mac never sleeps:

```bash
nohup caffeinate -dis ~/Desktop/MRT2_demo/serve_keepalive.sh >/dev/null 2>&1 &
# stop: pkill -f serve_keepalive; pkill -9 -f "uvicorn server:app"; pkill caffeinate
```

To expose it publicly, use a **Tailscale Funnel** (`tailscale funnel --bg 8000`) and set `PUBLIC_URL` accordingly. **Note:** the Funnel is public with no auth — anyone who reaches it can drive the model and spend Claude credits, and the audience channel is reachable there.

### Notable gotchas

- **Single stream** — only one `/stream` (WebSocket) can run at a time; a second connection gets "another stream active".
- **Use Chrome on the Mac with the MIDI device** — Web MIDI is per-browser-machine; Safari's is unreliable.
- **Restart with `pkill -9`, not SIGTERM** — an open audience SSE connection (`/live`) makes uvicorn's graceful shutdown hang.
- **Anthropic credits are live** — `/navigate`, `/feeling`, `/danmaku` call `claude-opus-4-8` for real.
- **Never commit secrets** — `.env`, `landmarks.json`, and `audience_marks.json` are gitignored; check `git grep --cached -e 'sk-ant-'` is empty before pushing.

---

## File guide

### Application code

| Path | Function |
|---|---|
| `server.py` | FastAPI backend. Runs MRT2, streams audio over a WebSocket (`/stream`), handles style embedding / PCA / sequencer conditioning, the Claude-powered navigation, the audience subsystem (SSE + persistent marks), and serves the static pages. |
| `static/index.html` | The main app — a single file of HTML + CSS + vanilla JS (no build step). Renders the full-screen hypercube and all the performance layers: sequencer, groove instrument, MIDI, glitch FX, Wanderer 2, navigation, audience overlay, and the accessibility layer. |
| `static/m.html` | The phone audience page served at `/m`: two prompts ("feeling" + "steer the course"), SSE-aware so it reflects whether the channel is open. |
| `static/groove_cube.html` | Standalone groove visualizer served at `/groovecube`; fetches `/groove_library.json` and `/pca` from the same server. |

### Magenta model (vendored in-project)

| Path | Function |
|---|---|
| `magenta_rt/` | Vendored Magenta RealTime 2 package (the model code). Includes `mlx/` (the MLX inference system), `musiccoca.py`, `paths.py`, and `_vendor/` (which bundles `sequence_layers` via an import hook). Committed; shadows any pip-installed copy at runtime. |
| `magenta_home/magenta-rt-v2/` | The model weights — `checkpoints/mrt2_base.safetensors` (9.2 GB, loaded in-process), `resources/` (MusicCoCa TFLite + SpectroStream), `models/` (exported `.mlxfn`). **Gitignored (~13 GB)** — see Installation §1a. `server.py` points `MAGENTA_HOME` here. |
| `requirements.txt` | Pinned Python deps — the vendored model's third-party libraries (MLX, `ai-edge-litert`, …) plus the web-server deps. |

### Run / deploy scripts

| Path | Function |
|---|---|
| `run.sh` | Launches the app: activates the playground venv, sources `.env`, and starts uvicorn on `127.0.0.1:8000`. |
| `serve_keepalive.sh` | Self-healing launcher — relaunches uvicorn whenever the MLX backend hard-aborts. Meant to be run under `caffeinate` for public deploys; logs to `/tmp/mrt2_server.log`. |

### Data & assets

| Path | Function |
|---|---|
| `vocab.json` | Style vocabulary (words + embeddings) used for the "now: …" readout and as context for the navigation LLM. **Committed.** |
| `groove_dist/groove_library.json` | Groove library consumed by the standalone `/groovecube` page. (The rest of the old groove bundle was removed when the embedded Groove map was dropped.) |
| `tempo_refs/beat_*.wav` | 151 kick-beat reference clips (50–200 BPM) used as soft tempo prompts. **Gitignored (~133 MB)** — regenerate with `tools/make_tempo_beats.py`. |
| `agr/*.agr` | Ableton groove files, served at `/agr` and parsed client-side by the groove instrument into 16-slot timing/velocity profiles. |
| `landmarks.json` | Persisted user pins (description + embedding + polarity). **Gitignored.** |
| `audience_marks.json` | Persisted crowd "feelings" drawn on the cube (capped at 500). **Gitignored.** |
| `.env` | Secrets and config (`ANTHROPIC_API_KEY`, `AUDIENCE_ADMIN_TOKEN`, `PUBLIC_URL`). **Gitignored.** |

### Tools

| Path | Function |
|---|---|
| `tools/make_tempo_beats.py` | Regenerates the `tempo_refs/` kick-beat clips. |
| `agr_txt/` | `.agr` inspection utility — `agr_to_txt.py` dumps an Ableton groove file to readable text; see `agr_txt/README.txt`. |

`.gitignore` excludes: `.env`, `__pycache__`, `outputs/`, `uploads/`, `*.mp3`, `landmarks.json`, `*.bak`, `tempo_refs/`, `.DS_Store`, `groove-swing-*.wav`, `*.zip`, `agr_txt/*.agr`, `audience_marks.json`, **`magenta_home/`** (13 GB weights), **`.venv/`**.

---

## Architecture notes

### Backend (`server.py`)

- **Two MusicCoCa instances** — the generator's internal one plus a separate `get_embed_mc()` for HTTP embedding, because sharing one tflite across threads corrupts it. Single-thread executors (`_gen_executor` / `_embed_executor`) keep MLX streams thread-local; the `_gen_lock` setup lives inside the stream try/finally so it never leaks.
- **`/stream` (WebSocket)** loops `generate()`, chaining state and sending float32 stereo PCM with an adaptive chunk/lead. In sequencer mode the lead is tighter and frame-dithered. Live control messages: `params | style | seq | underrun | stop`.
- **`seq` conditioning** — `{steps, fps, notes:[[midi…]], drums:[0/1]}`. A fractional `fps` is error-diffused to integer frames so the average tempo equals an exact BPM on the 25 fps grid.
- **`/navigate` (Claude tool-use)** picks a landmark / blends / rewrites a phrase into a target embedding, softly repelling from 👎 landmarks, and also sees the audience's recent feelings so it can steer "to where the crowd felt euphoric".
- **Audience subsystem** — `/audience/*` endpoints plus an SSE feed (`/live`); marks persist to `audience_marks.json` (lock-guarded, capped at 500). The QR (`/m_qr`) encodes `PUBLIC_URL + "/m"` so it is correct even when the operator opens localhost.

### Key concepts

- **MRT2 has no direct tempo/velocity input.** Tempo is a *soft* steer via a kick-beat audio prompt; the only real timing/pitch control is `seq` (25 fps onset conditioning) from the step sequencer.
- **Style is a 768-d MusicCoCa embedding**; MRT2 conditions on 12 discrete RVQ tokens derived from it.
- **Two distinct spaces** — the main cube is the MusicCoCa *style* space (drives MRT2); the groove instrument's pad is a separate *groove* (timing+velocity) PCA. They are not interchangeable.
- **CFG range is [-1, 7]** for style/notes/drums (7 = max literal, -1 = anti-guidance).
