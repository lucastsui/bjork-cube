# Björk Cube — status quo / self-handoff

_Last updated: 2026-06-07. A handoff so a fresh context resumes fast: the non-obvious stuff (architecture, gotchas, current direction) on top of what the code shows._

## What this is
A **local web app** driving **Magenta RealTime 2** (MRT2, `magenta-rt` MLX, `mrt2_base`, 8-bit) for a **continuous, live-steerable music stream**, presented as a **full-screen navigable PCA hypercube** of the latent style space. On top of the cube there are now several **performance/steering layers**: a step sequencer, a groove instrument, MIDI control, an autonomous "Wanderer 2" scout, LLM feeling-breadcrumbs, geometric navigation, and the embedded Morphing Groove Map.

- Folder `~/Desktop/MRT2_demo/`. Backend `server.py` (FastAPI). Frontend `static/index.html` (one file: HTML+CSS+vanilla JS, no build). Standalone Groove Cube page `static/groove_cube.html`.
- GitHub (public): **github.com/lucastsui/bjork-cube**. Weights: `~/Documents/Magenta/magenta-rt-v2/` (`checkpoints/mrt2_base.safetensors`, 9.2 GB).

## The direction we're heading (read this first)
The trend across recent sessions: **turn the cube from a "navigator" into a playable, performable instrument**, steered many ways at once. Concretely, recent work added —
1. **Physical control** — Web MIDI (MiniLab 3): keys play notes the AI follows, faders crossfade prompt weights / travel speed, pads pause/resume travel + jump to nearest point.
2. **Autonomous agents** — "Wanderer 2", a simulated scout that wanders and leaves a trail to follow (no audio of its own).
3. **Semantic breadcrumbs** — LLM-named "feeling" landmarks dropped while wandering, that fade over time.
4. **Literal geometry** — typing edge/corner/center/random in the go-box moves to actual cube coordinates (vs the LLM's musical interpretation).
5. **UI consolidation + polish** — merged the bottom panels into one dock, toggle switches, flip selectors, tidy Sampling, etc.

**Likely next steps** (not done): wire MIDI to more targets (knobs→cube PC axes; MIDI-learn), make Wanderer 2's marks distinct from your own, let the feeling-trail/danmaku react to the *live* cube position (nearest-vocab) rather than the prompt text, and remove the temporary MIDI monitor before any "real" public push.

## Run / deploy
```bash
~/Desktop/MRT2_demo/run.sh                 # venv + .env, uvicorn on 127.0.0.1:8000
```
- venv `~/code/playground/.venv` (magenta-rt[mlx], fastapi, uvicorn, anthropic, soundfile, numpy).
- **Restart after a server change**: `pkill -f "uvicorn server:app"` (if the keepalive is running it relaunches automatically), else rerun the uvicorn line. Frontend-only edits need **no restart** — just reload. Log: `/tmp/mrt2_server.log`.
- **Self-healing deploy**: the MLX/Metal backend can **hard-abort** (`[metal::malloc] Resource limit exceeded` → `libc++abi: terminating`) which kills the whole process — uncatchable in Python. `serve_keepalive.sh` relaunches uvicorn on exit; run it under `caffeinate` so the Mac doesn't sleep:
  ```bash
  nohup caffeinate -dis ~/Desktop/MRT2_demo/serve_keepalive.sh >/dev/null 2>&1 &
  # stop: pkill -f serve_keepalive; pkill -f "uvicorn server:app"; pkill caffeinate
  ```
- **Public**: Tailscale Funnel → **https://tsuis-macbook-pro.tail2214e5.ts.net** (public, NO auth; anyone can use the model + spend Claude credits). `tailscale funnel --bg 8000`. Funnel persists across reboot; uvicorn does not.

## Current state / blockers
- **Anthropic credits LIVE** — `/navigate`, `/danmaku`, `/feeling` all call `claude-opus-4-8` for real.
- **Single stream** (`_gen_lock`): one `/stream` at a time; a 2nd gets "another stream active". Live audio can't be script-tested while a browser holds it.
- **Web MIDI is per-browser-machine**: it reads MIDI devices on the machine running the browser, so use the app **in Chrome on this Mac** (where the MiniLab 3 is) for MIDI. Safari's Web MIDI is unreliable.
- **TEMP MIDI monitor still in the build** (`#midiMon`, top-right, collapsible) — remove before any clean public release.

## Backend (server.py) — load-bearing decisions
- **Two MusicCoCa instances** (generator's internal + a separate `get_embed_mc()` for HTTP embedding) — sharing one tflite across threads corrupts it. **Single-thread executors** (`_gen_executor` / `_embed_executor`) — MLX streams are thread-local. **`_gen_lock` setup is INSIDE the stream try/finally** so it never leaks.
- **`/stream` (WebSocket)**: loops `generate()`, chains state, sends float32 stereo PCM. Adaptive chunk/lead; in **seq mode** the lead is tighter and frame-dithered. Live control msgs: `params | style | seq | underrun | stop`. **Underrun backoff** now persists (`underrun_margin`) — a past bug overwrote it.
- **`seq` conditioning**: `{steps, fps, notes:[[midi…]], drums:[0/1]}`. `fps` can be **fractional** → server **error-diffuses** it to integer frames so the average tempo equals an exact BPM on the 25 fps grid. `notes[p]=2` = onset (pitches 0-127); drums binary.
- **`/feeling`** (Claude): short mood phrase for a spot, given nearby vocab words + a **random lens** (color/weather/creature/…) so repeated/identical inputs still vary. Fallback pool. Used by the feeling-trail.
- **`/navigate`** (Claude tool-use): musical only — picks a landmark / blends / vocabulary / rewrites to a phrase → target embedding; soft-repels from 👎 landmarks. **No geometry awareness** (that's handled client-side, see geometric nav).
- **`/pca`** clamps `k=min(n-1,dims)`. Persisted: `landmarks.json` (pins: desc+embedding+`polarity`, gitignored), `vocab.json` (committed).

### Endpoints
`GET /`, `/groovecube`, `/status`, `/vocab`, `/vocab_points`, `/agr_list` · `POST /prepare_style` & `/embedding` (accept `tempo_bpm`+`tempo_weight`) · `/prepare_raw`, `/pca` · landmarks `GET/POST /landmark…`, `POST /landmark/{id}/play` · `/navigate`, `/danmaku`, `/feeling` · `WS /stream` · mounts `/assets` (groove_dist), `/agr` (agr/ files), `/static`; routes serve `/groove` + 5 groove root files.

## Frontend (static/index.html) — full-screen layout
- **Top bar**: title · "comments" toggle (danmaku) · ▶/⏸ · dot · `embChart` (style-embed wave) · `waveTape` (live output wave) · `log` (status; **emoji stripped** via the `log()` helper) · `perf`.
- **Full-screen hypercube** (`#pcaCube`): drag empty = rotate (Shift = 4D), pinch/scroll zoom, **drag bright dot = move position** (cancels travel + pauses drift), hover = names, **click any dot = glide there**, fading trail. **The cube dot tracks the current style** (start/restyle/preview all `projectToPCA`), so drift starts where you are.
- **Left pane** (`ov-left`): **tempo (bpm)** slider + **tempo weight** slider (**default 0 = off**) → mixes `tempo_refs/beat_<bpm>bpm.wav`; **Style** (mix prompts/audio; the type chooser is a **flip button** `prompt ⇄ / audio ⇄`, fixed 104px; audio rows are **drag-drop** dropzones, "browse or drop audio", same box height as the prompt input; add=`+`, save=icon-only grey SVG, "I'm feeling lucky" — all on one row); **Ingredients** (word bubbles); **Presets** (borderless delete); **Wander & navigate** (drift toggle, **Wanderer 2** toggle, travel-speed slider, 👍/👎 pin, **go →** box).
- **Right pane** (`ov-right`): Groove map iframe (`/groove`) top + **Sampling & guidance** (temperature, top_k, **cfg·style/notes/drums all in one vertical column**; cfg·notes/drums default 7) + **PCA explorer** (cube-dims slider + per-PC sliders).
- **Bottom dock** (`#dock`): ONE segmented bar — **`[ Step sequencer | Groove instrument ]`** — replaces the old two stacked cards. `#seqPanel`/`#giPanel` overlap above it, `display:none` by default; `dockShow(which)` shows one at a time (`__giRedraw` hook redraws the groove pad/strip on open).
  - **Step sequencer**: drum row on top + note grid (2-oct C-major); **integer-precise BPM** (slider+number, server dithers); conditions MRT2 via `seq` (the *real* native control). cfg·notes/drums live in Sampling.
  - **Groove instrument**: morph pad (groove-space PCA) + synth drum loop + 16-step strip; patterns incl. **Drum & bass**, **UK garage**; **`.agr` presets** (parse Ableton `.agr` client-side → 16-slot timing/velocity); **"use as style prompt"** bounces the loop offline → normalized+saturated WAV → Style audio input. Self-contained IIFE (own AudioContext) — isolated from the stream.
- **Checkboxes are sliding toggle switches** (CSS). **Danmaku** overlay top, 3 rows, behind cards, clickable → "go →".

### MIDI (Web MIDI, no driver) — `onMIDI(ev)` dispatch by channel
- **ch1 notes** → live play: held keys are streamed as a 1-step `seq` (the AI follows what you hold) and **light up the grid** (snapped to scale).
- **Faders (CC)** → `FADER_CC = {82:prompt1, 83:prompt2, 85:prompt3, 17:prompt4}` weights; `CC_SLIDER = {74: travel speed}`.
- **ch10 pads** (edge-triggered, one pulse/press): **note 40 = pause travel**, **41 = resume**, **any other = jump to nearest defined dot**.
- Pause = `travelPaused` gates drift + all glides; resume turns wandering back on.

### Other interaction layers
- **Wanderer 2** (`w2Tick`): a simulated red scout (dot + fading red tail) on its own OU walk; **drops feeling breadcrumbs**; **never calls generation** (pure sim). **Click the red dot → the main position glides to it.**
- **Feeling trail**: while drifting (and via Wanderer 2), sporadically drop a **non-persistent 👍 landmark** named by `/feeling`; **fades out over 20 s** (timestamped, 500 ms pruner); 15 px labels; clickable to glide.
- **Geometric nav** (`geoNavigate` in the go-box, before the LLM): `corner`/`vertex` → random cube vertex; `edge` → push current point to the box surface; `center`/`origin` → mean; `random`/`anywhere` → random point. Anything else → the musical LLM.
- **Default presets**: `loadDefaultPresets()` seeds **7 diverse presets** at launch so the cube/PCA is usable immediately (presets aren't persisted).

## Files
```
server.py                 backend
static/index.html         main app
static/groove_cube.html   standalone /groovecube
groove_dist/              built Morphing Groove Map (committed) + its wav/json
tempo_refs/beat_*.wav     151 kick beats 50-200 BPM (GITIGNORED ~133MB) — regen: python tools/make_tempo_beats.py
agr/*.agr                 12 Ableton groove files (served at /agr, parsed client-side)
agr_txt/                  .agr inspection tool (agr_to_txt.py + README.txt; dumps .agr -> readable XML)
serve_keepalive.sh        self-healing launcher (auto-restarts uvicorn; run under caffeinate)
tools/make_tempo_beats.py regenerates tempo_refs
landmarks.json (gitignored)   vocab.json (committed)   .env (key, gitignored)
```
`.gitignore`: `.env`, `__pycache__`, `outputs/`, `uploads/`, `*.mp3`, `*.zip`, `landmarks.json`, `tempo_refs/`, `agr_txt/*.agr`, `.DS_Store`, `*.bak`.

## Concepts that took iteration
- **MRT2 has no tempo/velocity input** (verified). Tempo is a *soft* steer via a kick-beat audio prompt; the **only real timing/pitch control is `seq`** (note/drum onset conditioning, 25 fps), which the **step sequencer** drives. cfg·notes/cfg·drums (max 7) set how strictly it follows; drums are coarse (binary "drum activity").
- **`.agr`/.stt ↔ bars**: an `.agr` (Ableton MIDI-clip groove) folds to a 16-slot timing+velocity profile — what the Groove instrument shows. The instrument is a **fixed-beat synth loop**, NOT MRT2, because MRT2 can't apply microtiming.
- **Two cubes**: main cube = MusicCoCa **style** space (drives MRT2); the Groove instrument pad = a separate **groove** (timing+velocity) PCA. Not interchangeable.
- **CFG range is [-1, 7]** for all three (7 = max, very literal; -1 = anti-guidance).

## Verification habits
- JS: extract `<script>` and `node --check`. Server: `python -c "import ast; ast.parse(open('server.py').read())"`. `curl` endpoints. Live audio / pin-play can't be tested while the browser holds `_gen_lock`; **Playwright MCP is usually locked by the user's open browser** (so MIDI / live-cube behaviors are verified structurally, not in-page).
- Before any push: `git grep --cached -I -e 'sk-ant-'` must be empty; never commit `.env`.

## Open ideas (not done)
- Remove the temp MIDI monitor for a clean release.
- MIDI: knobs → cube PC axes; a MIDI-learn flow; distinguish Wanderer 2's breadcrumbs from your own.
- Make danmaku / feeling-trail react to the **live nearest-vocab words** at the current cube position, not the prompt text.
- Reduce MLX/Metal crash frequency (periodic cache eviction) so the keepalive fires less.
