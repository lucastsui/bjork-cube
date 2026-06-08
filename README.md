# Morphing Music Map — status quo / self-handoff

_Last updated: 2026-06-08. A handoff so a fresh context resumes fast: the non-obvious stuff (architecture, gotchas, current direction) on top of what the code shows._

_Names: UI title **"Morphing Music Map"**, browser tab **"Music Morphing Space"**, repo still **bjork-cube** (formerly "Björk Cube"). Folder `~/Desktop/MRT2_demo/`._

## What this is
A **local web app** driving **Magenta RealTime 2** (MRT2, `magenta-rt` MLX, `mrt2_base`, 8-bit) for a **continuous, live-steerable music stream**, presented as a **full-screen navigable PCA hypercube** of the latent style space. On top of the cube sit several **performance/steering layers**: a step sequencer, a groove instrument, the embedded Morphing Groove Map, MIDI control, a real-time glitch FX, an autonomous "Wanderer 2" scout, LLM feeling-breadcrumbs, geometric navigation, **a phone audience-participation channel**, and an **accessibility layer**.

- Backend `server.py` (FastAPI). Frontend `static/index.html` (one file: HTML+CSS+vanilla JS, no build). Phone page `static/m.html`. Standalone Groove Cube page `static/groove_cube.html`.
- GitHub (public): **github.com/lucastsui/bjork-cube**. Weights: `~/Documents/Magenta/magenta-rt-v2/` (`checkpoints/mrt2_base.safetensors`, 9.2 GB).

## The direction we're heading (read this first)
The trend across recent sessions: **turn the cube from a "navigator" into a playable, performable, participatory instrument**, steered many ways at once, and make it **legible** (interpret the latent space through crowd labels + accessible UI). Recent work added —
1. **Physical control** — Web MIDI (MiniLab 3): keys play notes the AI follows; faders crossfade prompt weights, travel speed, glitch, and A→B; pads jump to nearest/furthest/random + stop/continue travel.
2. **Real-time glitch** — a client-side bitcrusher on the audio stream (slider + MIDI CC1), no server/model change.
3. **Audience channel** — phones (`/m`) leave "feelings" (persistent cube breadcrumbs the nav-LLM can steer to) and "steer the course" comments (danmaku on the big screen), pushed live via SSE; a QR on the main screen; closed-by-default + per-device rate limit.
4. **Accessibility** — ARIA names on every control, a plain-language "now: …" readout, screen-reader-friendly thumbs, dark scrollbars.
5. **Autonomous agents + semantic breadcrumbs** — "Wanderer 2" scout (white/green dot + trail) and fading LLM-named "feeling" landmarks.
6. **UI consolidation** — one bottom dock now holds **three** segments: Step sequencer · Groove instrument · Groove map.

**Likely next steps** (not done): `prefers-reduced-motion` support (auto-calm motion + cap glitch); a `requirements.txt` capturing the `segno` dep; MIDI-learn / knobs→PC axes; remove the temporary MIDI monitor before a clean public release.

## Run / deploy
```bash
~/Desktop/MRT2_demo/run.sh                 # venv + .env, uvicorn on 127.0.0.1:8000
```
- venv `~/code/playground/.venv` (magenta-rt[mlx], fastapi, uvicorn, anthropic, soundfile, numpy, **segno** for the QR). No `requirements.txt` yet — `segno` was added via `uv add segno`; **add it to any redeploy** (graceful fallback: `/m_qr` renders the URL as text if segno is missing).
- **Restart after a server change**: `pkill -9 -f "uvicorn server:app"` (force kill — see SSE gotcha below; the keepalive then relaunches), else rerun uvicorn. Frontend-only edits need **no restart** — just reload. Log: `/tmp/mrt2_server.log`.
- **Self-healing deploy**: the MLX/Metal backend can **hard-abort** (`[metal::malloc] Resource limit exceeded` → `libc++abi: terminating`) which kills the whole process — uncatchable in Python. `serve_keepalive.sh` relaunches uvicorn on exit; run it under `caffeinate`:
  ```bash
  nohup caffeinate -dis ~/Desktop/MRT2_demo/serve_keepalive.sh >/dev/null 2>&1 &
  # stop: pkill -f serve_keepalive; pkill -9 -f "uvicorn server:app"; pkill caffeinate
  ```
- **Public**: Tailscale Funnel → **https://tsuis-macbook-pro.tail2214e5.ts.net** (public, NO auth; anyone can use the model + spend Claude credits, and the audience channel is reachable here). `tailscale funnel --bg 8000`. Funnel persists across reboot; uvicorn does not. Override the audience base URL with `PUBLIC_URL=…` in `.env`.

## Current state / gotchas
- **Anthropic credits LIVE** — `/navigate`, `/danmaku`, `/feeling` all call `claude-opus-4-8` for real.
- **Single stream** (`_gen_lock`): one `/stream` at a time; a 2nd gets "another stream active". Live audio can't be script-tested while a browser holds it.
- **⚠ SSE breaks graceful shutdown**: an open `/live` (audience SSE) connection makes uvicorn's graceful shutdown **hang** ("Waiting for connections to close"). Always restart with **`pkill -9`** (force), not a polite SIGTERM.
- **Web MIDI is per-browser-machine**: use the app **in Chrome on this Mac** (where the MiniLab 3 is). Safari's Web MIDI is unreliable.
- **TEMP MIDI monitor still in the build** (`#midiMon`, top-right, collapsible) — remove before a clean public release.

## Backend (server.py) — load-bearing decisions
- **Two MusicCoCa instances** (generator's internal + a separate `get_embed_mc()` for HTTP embedding) — sharing one tflite across threads corrupts it. **Single-thread executors** (`_gen_executor` / `_embed_executor`) — MLX streams are thread-local. **`_gen_lock` setup is INSIDE the stream try/finally** so it never leaks.
- **`/stream` (WebSocket)**: loops `generate()`, chains state, sends float32 stereo PCM. Adaptive chunk/lead; in **seq mode** the lead is tighter and frame-dithered. Live control msgs: `params | style | seq | underrun | stop`. **Underrun backoff persists** (`underrun_margin`).
- **`seq` conditioning**: `{steps, fps, notes:[[midi…]], drums:[0/1]}`. `fps` can be **fractional** → server **error-diffuses** it to integer frames so the average tempo equals an exact BPM on the 25 fps grid. `notes[p]=2` = onset (pitches 0-127); drums binary.
- **`/feeling`** (Claude): short mood phrase for a spot, given nearby vocab words + a **random lens** (color/weather/creature/…) so repeated inputs still vary. Fallback pool. Used by the feeling-trail.
- **`/navigate`** (Claude tool-use): musical only — picks a landmark / blends / vocabulary / rewrites to a phrase → target embedding; soft-repels from 👎 landmarks. **Now also sees the audience's feelings** (recent 30, deduped, ids `am…`) so it can steer "to where the crowd felt euphoric". **No geometry awareness** (that's client-side, see geometric nav).
- **`/pca`** clamps `k=min(n-1,dims)`. Persisted: `landmarks.json` (pins: desc+embedding+`polarity`, gitignored), `vocab.json` (committed), **`audience_marks.json`** (persistent crowd feelings, gitignored, cap 500).

### Audience subsystem (phones → main screen)
- **`_aud_open`** (default **False** = closed), **`_aud_subscribers`** (set of asyncio.Queue, one per SSE client), **`_aud_rate`** (per-`device:kind` throttle, 2.5 s), **`_aud_marks`** (persistent, lock-guarded, saved to `audience_marks.json`).
- **`PUBLIC_URL`** (env or the Funnel host) → `_aud_join = PUBLIC_URL + "/m"`; the QR encodes this (not `location.origin`, so it's right even if the operator opens localhost).
- `_aud_post(request, kind, cap)`: closed→403; trims+caps; empty→400; per-device(+question) rate→429; else broadcast `{type,text}` to subscribers. Device id is client-generated in `localStorage('aud_dev')`, sent with each post (per-IP would collapse all phones behind the Funnel NAT).
- **QR** uses **segno** (pure-Python SVG), cached in `_aud_qr_cache`; text-SVG fallback if segno missing.

### Endpoints
`GET /`, `/groovecube`, `/groove` (+5 groove root files), `/status`, `/vocab`, `/vocab_points`, `/agr_list`, `/landmarks` · **audience**: `GET /m`, `GET /m_qr`, `GET /audience/status`, `POST /audience/say` (80), `POST /audience/feel` (40), `POST /audience/open`, `GET /live` (SSE), `POST /audience/mark` + `GET /audience/marks` · `POST /prepare_style` & `/embedding` (accept `tempo_bpm`+`tempo_weight`), `/prepare_raw`, `/pca`, `/generate` · landmarks `POST /landmark`, `DELETE /landmark/{id}`, `POST /landmark/{id}/play` · `/navigate`, `/danmaku`, `/feeling` · `WS /stream` · mounts `/assets` (groove_dist), `/agr`, `/static`.

## Frontend (static/index.html) — full-screen layout
- **Top bar**: title "Morphing Music Map" · **`now: <3 nearest vocab words>`** readout (`#nowWords`, aria-live) · "comments" (danmaku) + **"slow rotation"** toggles · ▶/⏸ · dot · `embChart` · `waveTape` · `log` (emoji stripped via `log()`) · `perf`.
- **Full-screen hypercube** (`#pcaCube`, `role="img"`): drag empty = rotate (Shift = 4D), pinch/scroll zoom, **drag bright dot = move position**, hover = names, **click any dot = glide there**, fading trail. The dot tracks the current style. **"slow rotation"** auto-rotates ~one turn / 2 min. Current/**HACKATHON** marker is **coral**; **Wanderer 2** is **white-with-green-ring** (colors + trails swapped).
- **Left pane** (`ov-left`): **tempo (bpm)** + **tempo weight** (default 0=off) sliders → mixes `tempo_refs/beat_<bpm>bpm.wav`; **Style** (default prompt **"UK garage"**; flip button `prompt ⇄ / audio ⇄`; drag-drop audio dropzones; per-prompt **weight slider max 0→1**; add=`+`, save=grey SVG, "I'm feeling lucky"); **Ingredients**; **Presets**; **Audience** (open/close toggle + `channel closed/OPEN` state, synced via SSE); **Wander & navigate** (drift **on by default**, **Wanderer 2 on by default**, travel-speed slider 0.05–5 / 2-dp, 👍/👎 pin, **go →** box); **Breadcrumb map** (raw monospace list of AI + audience marks with the first 5 embedding values).
- **Right pane** (`ov-right`): big **"Morphing Music Map"** charter heading + tagline · **QR** (`/m_qr`) · **Sampling & guidance** (**glitch** slider first, then temperature, top_k, cfg·style/notes/drums in one column) · **PCA explorer** (cube-dims + per-PC sliders).
- **Bottom dock** (`#dock`): ONE segmented bar — **`[ Step sequencer | Groove instrument | Groove map ]`**. `#seqPanel`/`#giPanel`/`#groovePanel` overlap above it, hidden by default; `dockShow(which)` shows one at a time.
  - **Step sequencer**: drum row at the **bottom** + note grid **C1–C5**; integer-precise BPM (slider+number, server dithers); conditions MRT2 via `seq`.
  - **Groove instrument**: morph pad (groove-space PCA) + synth drum loop + 16-step strip; patterns incl. Drum & bass, **UK garage** (default); **`.agr` presets** (client-side parse → 16-slot timing/velocity); **"use as style prompt"** bounces the loop → WAV → Style audio input. Self-contained IIFE (own AudioContext).
  - **Groove map**: the **Morphing Groove Map** iframe (`/groove`), now rendered at **~normal 680px size** (was a shrunk right-card section); lazy-loaded on first open; ⤢ fullscreen.
- **Checkboxes are sliding toggle switches**. **Danmaku** overlay top, 3 rows, behind cards, clickable → "go →"; **audience comments always show** even when lanes are busy.

### Real-time glitch FX (client-side)
- `makeGlitch(ctx)` = a `ScriptProcessorNode(1024,2,2)` **bitcrusher** (sample-rate decimation + bit-depth reduction), spliced `src → glitchNode → destination`. Driven by the `glitch` 0→1 slider with a **steep `a^0.3` curve** (clearly crunchy by ~25%). `glitchNode` is recreated on play, torn down on stop/reconnect. (ScriptProcessorNode is deprecated-but-pragmatic for one file; AudioWorklet is the modern path.)

### MIDI (Web MIDI, no driver) — `onMIDI(ev)` dispatch by channel
- **ch1 notes** → live play: held keys stream as a 1-step `seq` and light the grid (snapped to scale).
- **Faders/CC**: `FADER_CC = {82:prompt1, 83:prompt2, 85:prompt3}` weights; `CC_SLIDER = {1:"glitch", 74:"driftSpeed"}`; **CC17 = A→B crossfader** (`slideTo`: A = position snapshot, B = the last-clicked / last-traveled destination — any travel sets B).
- **ch10 pads** (edge-triggered): **36 = nearest dot**, **37 = furthest cube vertex**, **38 = random point**, **40 = one-shot STOP** (cancels the current glide/directional travel but **keeps wandering**; does NOT latch, so a later 36/37/38 still travels), **41 = continue along the last A→B direction** (extends past B along the embedding shell).

### Other interaction layers
- **Wanderer 2** (`w2Tick`): a simulated scout (dot + fading trail) on its own OU walk; **drops feeling breadcrumbs**; **never calls generation**. Click it → the main position glides to it.
- **Feeling trail**: while drifting, sporadically drop a **non-persistent 👍 landmark** named by `/feeling`; fades over ~20 s; clickable.
- **Audience marks** (`audMarks`): persistent crowd "feelings" drawn on the cube (no-fade) + listed in the Breadcrumb map; restored via `GET /audience/marks` on load, appended via SSE + `POST /audience/mark`.
- **Geometric nav** (`geoNavigate`, before the LLM): `corner`/`vertex`/`edge`/`center`/`origin`/`random` move to literal cube coordinates; anything else → the musical LLM.
- **Default presets**: `loadDefaultPresets()` seeds 7 diverse presets (not persisted).

### Accessibility layer
- **`applyA11yLabels()`** (idempotent, re-runs every 2.5 s for dynamic controls): names every control — sliders ← visible `.lab`, text inputs ← placeholder, titled buttons/selects ← title.
- Explicit `aria-label`s on the canvas + icon buttons (▶/⏸, +, save, 👍/👎, ⤢); `#log` is `role="status" aria-live`.
- **Thumbs not spoken**: saved-place chips hide 👎 (`aria-hidden`) and expose a `.sr-only` "avoid place:"; pin status uses plain "pinned:/avoiding:".
- **Plain-language readout**: `#nowWords` shows "now: <nearest vocab>".
- **Dark slim scrollbars** (consistent across the viewer's macOS scroll-bar setting).

## Files
```
server.py                 backend (FastAPI; MRT2 + audience + Claude)
static/index.html         main app (full-screen cube + dock + audience controls)
static/m.html             phone audience page (/m): two questions, SSE-aware
static/groove_cube.html   standalone /groovecube
groove_dist/              built Morphing Groove Map (committed) + its wav/json
tempo_refs/beat_*.wav     151 kick beats 50-200 BPM (GITIGNORED ~133MB) — regen: python tools/make_tempo_beats.py
agr/*.agr                 Ableton groove files (served at /agr, parsed client-side)
agr_txt/                  .agr inspection tool (agr_to_txt.py + README.txt)
serve_keepalive.sh        self-healing launcher (run under caffeinate)
tools/make_tempo_beats.py regenerates tempo_refs
landmarks.json (gitignored)  audience_marks.json (gitignored)  vocab.json (committed)  .env (gitignored)
```
`.gitignore`: `.env`, `__pycache__`, `outputs/`, `uploads/`, `*.mp3`, `landmarks.json`, `*.bak`, `tempo_refs/`, `.DS_Store`, `groove-swing-*.wav`, `*.zip`, `agr_txt/*.agr`, `audience_marks.json`.

## Concepts that took iteration
- **MRT2 has no tempo/velocity input** (verified). Tempo is a *soft* steer via a kick-beat audio prompt; the **only real timing/pitch control is `seq`** (25 fps onset conditioning) driven by the step sequencer. cfg·notes/drums (max 7) set strictness; drums are coarse (binary).
- **Style is a 768-d MusicCoCa embedding**; MRT2 conditions on **12 discrete RVQ tokens** tokenized from it (so "12-d" you may have heard = the token count, not the steer dimensionality).
- **`.agr` ↔ bars**: an Ableton MIDI-clip groove folds to a 16-slot timing+velocity profile — what the Groove instrument shows. The instrument is a **fixed-beat synth loop**, NOT MRT2 (MRT2 can't apply microtiming).
- **Two cubes**: main cube = MusicCoCa **style** space (drives MRT2); the Groove instrument pad = a separate **groove** (timing+velocity) PCA. Not interchangeable.
- **CFG range is [-1, 7]** for all three (7 = max literal; -1 = anti-guidance).
- **Furthest point** = the opposite corner of the bounded PCA cube (per-axis, maximize distance from the current coord) — geometrically meaningful in this finite box.

## Verification habits
- JS: extract `<script>` and `node --check`. Server: `python -c "import ast; ast.parse(open('server.py').read())"`. `curl` endpoints (`-o /dev/null -w "%{http_code}"`). Audience round-trip can be checked via `/audience/status`, `/audience/open`, posting to `/audience/feel`, and watching `/live`.
- Live audio / pin-play can't be tested while a browser holds `_gen_lock`; **Playwright MCP is usually locked by the user's open browser**, so MIDI / live-cube / glitch / VoiceOver behaviors are verified structurally, then by the user in-page.
- Before any push: **`git grep --cached -I -e 'sk-ant-'` must be empty**; never commit `.env`, `landmarks.json`, or `audience_marks.json`. (This README's mention of that grep is a benign secret-scan false positive.)

## Open ideas (not done)
- `prefers-reduced-motion`: auto-pause drift/Wanderer 2/rotation/danmaku and cap the glitch for users with that OS setting.
- `requirements.txt` capturing `segno` (and the rest) for clean redeploys.
- Remove the temp MIDI monitor; MIDI-learn / knobs → cube PC axes.
- Make danmaku / feeling-trail react to the **live nearest-vocab words** (the `#nowWords` machinery now exists to reuse).
- Reduce MLX/Metal crash frequency (periodic cache eviction) so the keepalive fires less.
