"""
MRT2 Demo backend.

A tiny FastAPI server that wraps the locally-installed Magenta RealTime 2 (MLX)
model and exposes every generation input we can drive from a web UI:

  - style          : text prompt OR an uploaded audio clip (style reference)
  - temperature    : sampling temperature
  - top_k          : top-k sampling
  - cfg_musiccoca  : classifier-free-guidance strength for the style
  - cfg_notes      : guidance strength for note control
  - cfg_drums      : guidance strength for drum control
  - notes / drums  : optional advanced control arrays (raw JSON pass-through)
  - duration       : length of generated audio (converted to frames; 25 frames = 1s)

The model weights live at ~/Documents/Magenta/magenta-rt-v2 (the same place the
sample apps downloaded them), found automatically via the MAGENTA_HOME default.
"""

import io
import json
import time
import asyncio
import pathlib
import threading
import itertools
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

HERE = pathlib.Path(__file__).parent
FRAMES_PER_SECOND = 25  # 25 frames == 1 second of audio (per the model docs)

app = FastAPI(title="MRT2 Demo")

# ---------------------------------------------------------------------------
# Lazy model loading. Instantiating MagentaRT2System loads ~2.6GB of weights
# and can take a while, so we do it once on first request, guarded by a lock.
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()
_model_status = {"state": "not_loaded", "error": None, "load_seconds": None}


def get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        from magenta_rt.mlx.system import MagentaRT2System
        _model_status["state"] = "loading"
        t0 = time.time()
        try:
            m = MagentaRT2System(size="mrt2_base", bits=8)
        except Exception as e:  # noqa: BLE001
            _model_status["state"] = "error"
            _model_status["error"] = repr(e)
            raise
        _model = m
        _model_status["state"] = "ready"
        _model_status["load_seconds"] = round(time.time() - t0, 1)
        return _model


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text()


@app.get("/status")
def status():
    return JSONResponse(_model_status)


def _parse_int_list(raw: str | None):
    """Parse an optional JSON array of ints; return None if empty/invalid."""
    if not raw or not raw.strip():
        return None
    try:
        val = json.loads(raw)
        if isinstance(val, list) and all(isinstance(x, int) for x in val):
            return val
    except Exception:  # noqa: BLE001
        pass
    return None


@app.post("/generate")
async def generate(
    prompt: str = Form(""),
    duration: float = Form(4.0),
    temperature: float = Form(1.3),
    top_k: int = Form(40),
    cfg_musiccoca: float = Form(3.0),
    cfg_notes: float = Form(1.0),
    cfg_drums: float = Form(1.0),
    notes: str = Form(""),
    drums: str = Form(""),
    seed: int = Form(0),
    use_mapper: bool = Form(False),
    style_audio: UploadFile | None = File(None),
):
    from magenta_rt import audio

    mrt = get_model()

    # ---- Build the style embedding from EITHER an audio clip or text -------
    style_source = None
    if style_audio is not None and style_audio.filename:
        raw = await style_audio.read()
        wav_in = audio.Waveform.from_file(io.BytesIO(raw))
        style = mrt.embed_style(wav_in, use_mapper=use_mapper, seed=seed)
        style_source = f"audio:{style_audio.filename}"
    elif prompt.strip():
        style = mrt.embed_style(prompt.strip(), use_mapper=use_mapper, seed=seed)
        style_source = f"text:{prompt.strip()!r}"
    else:
        return JSONResponse(
            {"error": "Provide either a text prompt or an audio style file."},
            status_code=400,
        )

    frames = max(1, round(duration * FRAMES_PER_SECOND))

    t0 = time.time()
    wav_out, _state = mrt.generate(
        style=style,
        notes=_parse_int_list(notes),
        drums=_parse_int_list(drums),
        cfg_musiccoca=cfg_musiccoca,
        cfg_notes=cfg_notes,
        cfg_drums=cfg_drums,
        temperature=temperature,
        top_k=top_k,
        frames=frames,
    )
    gen_seconds = round(time.time() - t0, 2)

    buf = io.BytesIO()
    wav_out.write(buf, format="WAV")
    buf.seek(0)

    headers = {
        "X-Style-Source": style_source,
        "X-Frames": str(frames),
        "X-Gen-Seconds": str(gen_seconds),
        "X-Sample-Rate": str(getattr(wav_out, "sample_rate", "")),
        "Content-Disposition": 'inline; filename="mrt2_output.wav"',
    }
    return StreamingResponse(buf, media_type="audio/wav", headers=headers)


# ---------------------------------------------------------------------------
# Streaming support
#
# A style embedding is computed once (from text or an uploaded clip) and cached
# under a token. The /stream WebSocket then loops generate() forever, chaining
# `state` chunk-by-chunk for gapless audio, and pushes raw float32 stereo PCM to
# the browser. Sampling params and the active style can be changed live, mid-
# stream, by sending JSON control messages.
# ---------------------------------------------------------------------------
_style_cache: dict[str, object] = {}
_token_counter = itertools.count(1)
_gen_lock = threading.Lock()    # only one active stream/generation at a time
_embed_lock = threading.Lock()  # serialize calls to the endpoint MusicCoCa

# MLX streams are thread-local, so all generation must run on ONE fixed thread —
# otherwise a chunk scheduled on a different pool thread hits "no Stream(gpu) in
# current thread". A dedicated single-thread executor pins generation; embedding
# gets its own thread so a long embed runs concurrently without disturbing it.
_gen_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")
_embed_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")

# Dedicated MusicCoCa for the HTTP embedding endpoints, SEPARATE from the one the
# generator uses internally (mrt._style_model). Generation calls _style_model.tokenize
# every step; sharing one tflite model across the gen thread and an embed thread
# corrupts it and drops the stream. A second instance keeps them isolated.
_embed_mc = None
_embed_mc_lock = threading.Lock()


def get_embed_mc():
    global _embed_mc
    if _embed_mc is not None:
        return _embed_mc
    with _embed_mc_lock:
        if _embed_mc is None:
            from magenta_rt.musiccoca import MusicCoCa
            _embed_mc = MusicCoCa()
        return _embed_mc


# ---------------------------------------------------------------------------
# Landmarks + Claude-driven navigation.
# A landmark = a place the user named ("chill like beach") plus the 768-d
# embedding of that spot. Persisted to landmarks.json (survives restarts).
# Claude (claude-opus-4-8) reads the user's request + the landmark descriptions
# and picks ONE action via strict tool use: go to a pin, blend pins, or rewrite
# the request into a music phrase. The chosen action resolves to a target
# embedding server-side — Claude never emits coordinates.
# ---------------------------------------------------------------------------
LANDMARKS_PATH = HERE / "landmarks.json"
_landmarks: list[dict] = []
_landmarks_lock = threading.Lock()


def _load_landmarks():
    global _landmarks
    try:
        _landmarks = json.loads(LANDMARKS_PATH.read_text())
    except Exception:  # noqa: BLE001
        _landmarks = []
    for lm in _landmarks:
        lm.setdefault("polarity", "good")   # existing pins are positive


def _save_landmarks():
    LANDMARKS_PATH.write_text(json.dumps(_landmarks))


def _next_landmark_id():
    n = 0
    for lm in _landmarks:
        try:
            n = max(n, int(str(lm["id"]).lstrip("lm")))
        except Exception:  # noqa: BLE001
            pass
    return f"lm{n + 1}"


def _landmark_by_id(lid):
    return next((lm for lm in _landmarks if lm["id"] == lid), None)


_load_landmarks()


# ---------------------------------------------------------------------------
# Built-in vocabulary: instruments & genres with precomputed embeddings.
# Persisted to vocab.json so Claude can compose a target from known music words
# (and the server resolves it from cached embeddings). These are reference
# anchors, NOT user landmarks — they never appear in the "pinned places" list.
# ---------------------------------------------------------------------------
VOCAB_INSTRUMENTS = ["piano", "electric guitar", "acoustic guitar", "bass guitar", "double bass", "violin", "cello", "viola", "harp", "flute", "clarinet", "oboe", "bassoon", "saxophone", "trumpet", "trombone", "tuba", "french horn", "drum kit", "808 drums", "hi-hats", "congas", "bongos", "timpani", "marimba", "xylophone", "vibraphone", "organ", "hammond organ", "synthesizer", "synth pad", "moog bass", "rhodes piano", "accordion", "harmonica", "banjo", "mandolin", "ukulele", "sitar", "koto", "erhu", "tabla", "djembe", "steel drums", "kalimba", "theremin", "bagpipes", "fiddle", "harpsichord", "clavinet", "vocoder", "choir", "string section", "brass section",
                     "snare drum", "kick drum", "cymbals", "tambourine", "triangle", "glockenspiel", "tubular bells", "cowbell", "shaker", "hand claps", "timbales", "drum machine",
                     "wurlitzer", "celesta", "mellotron", "melodica", "pipe organ",
                     "piccolo", "english horn", "recorder", "pan flute", "cornet", "flugelhorn",
                     "pedal steel guitar", "slide guitar", "12-string guitar", "nylon guitar", "lute", "dulcimer", "zither"]
VOCAB_GENRES = ["jazz", "blues", "rock", "punk", "heavy metal", "death metal", "pop", "synthpop", "electronic", "EDM", "house", "deep house", "techno", "trance", "dubstep", "drum and bass", "jungle", "ambient", "lo-fi", "hip hop", "trap", "R&B", "soul", "funk", "disco", "reggae", "ska", "dub", "classical", "baroque", "orchestral", "opera", "folk", "country", "bluegrass", "gospel", "latin", "salsa", "bossa nova", "samba", "flamenco", "afrobeat", "k-pop", "indie", "shoegaze", "grunge", "breakbeat", "downtempo", "vaporwave", "cinematic",
                "alternative rock", "hard rock", "progressive rock", "psychedelic rock", "post-rock", "math rock", "emo", "post-punk", "new wave", "surf rock", "rockabilly",
                "swing", "bebop", "big band", "ragtime",
                "synthwave", "IDM", "glitch", "breakcore", "hardstyle", "future bass", "UK garage", "grime", "drill", "industrial", "minimal techno",
                "americana", "motown", "doo-wop",
                "metalcore", "doom metal", "black metal", "thrash metal"]
VOCAB_PATH = HERE / "vocab.json"
_vocab: dict = {}
_vocab_lock = threading.Lock()


def _vocab_targets():
    return [(w, "inst") for w in VOCAB_INSTRUMENTS] + [(w, "genre") for w in VOCAB_GENRES]


def _ensure_vocab_sync():
    """Load vocab.json; compute & persist any missing word embeddings via MusicCoCa."""
    global _vocab
    with _vocab_lock:
        if not _vocab:
            try:
                _vocab = json.loads(VOCAB_PATH.read_text())
            except Exception:  # noqa: BLE001
                _vocab = {}
        missing = [(w, k) for (w, k) in _vocab_targets() if w.lower() not in _vocab]
        if missing:
            mc = get_embed_mc()
            with _embed_lock:
                for w, k in missing:
                    e = np.asarray(mc.embed(w, True, False, 0), dtype=np.float32).ravel().tolist()
                    _vocab[w.lower()] = {"word": w, "kind": k, "embedding": e}
            VOCAB_PATH.write_text(json.dumps(_vocab))
        return _vocab


async def ensure_vocab():
    return await asyncio.get_event_loop().run_in_executor(_embed_executor, _ensure_vocab_sync)


_anthropic_client = None


def get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic
        _anthropic_client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
    return _anthropic_client


NAV_SYSTEM = (
    "You are the navigator for a latent music-space explorer. The user wanders a "
    "space of musical styles and pins 'places' they like, each with their own "
    "description. You also have a built-in vocabulary of instruments and genres "
    "(listed in the message) whose positions are known. Given the user's request, "
    "their saved places, and the vocabulary, choose how to travel by calling exactly "
    "one tool:\n"
    "- go_to_landmark: one saved place clearly matches the request.\n"
    "- blend_landmarks: the request sits between or combines saved places.\n"
    "- compose_from_vocab: the request maps to known genres/instruments — pick the "
    "matching vocabulary words with weights.\n"
    "- rewrite_to_music: novel or abstract requests that fit neither a saved place "
    "nor the vocabulary — rewrite into a concise musical style descriptor.\n"
    "Prefer the user's saved places when they match the vibe; otherwise compose from "
    "the vocabulary; use rewrite only as a last resort."
)

NAV_TOOLS = [
    {
        "name": "go_to_landmark",
        "description": "Travel to one saved place that best matches the request.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "landmark_id": {"type": "string", "description": "id of the saved place"},
                "reasoning": {"type": "string", "description": "one short sentence on why"},
            },
            "required": ["landmark_id", "reasoning"],
            "additionalProperties": False,
        },
    },
    {
        "name": "blend_landmarks",
        "description": "Travel to a weighted blend of saved places (a point between them).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "landmark_id": {"type": "string"},
                            "weight": {"type": "number"},
                        },
                        "required": ["landmark_id", "weight"],
                        "additionalProperties": False,
                    },
                },
                "reasoning": {"type": "string"},
            },
            "required": ["weights", "reasoning"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compose_from_vocab",
        "description": "Compose the target from the built-in vocabulary of instruments and genres listed in the message. Use when the request maps to known genres/instruments; pick matching words with weights.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {"type": "string", "description": "a word from the vocabulary"},
                            "weight": {"type": "number"},
                        },
                        "required": ["word", "weight"],
                        "additionalProperties": False,
                    },
                },
                "reasoning": {"type": "string"},
            },
            "required": ["terms", "reasoning"],
            "additionalProperties": False,
        },
    },
    {
        "name": "rewrite_to_music",
        "description": "When neither a saved place nor the vocabulary fits, rewrite the request into a concise musical style descriptor to ground it directly.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "phrase": {"type": "string", "description": "e.g. 'aggressive heavy metal, distorted guitars, double-kick drums'"},
                "reasoning": {"type": "string"},
            },
            "required": ["phrase", "reasoning"],
            "additionalProperties": False,
        },
    },
]

STREAM_FRAMES = 50  # initial free-run chunk (~2s); adapted at runtime (25 frames == 1s)
CHUNK_MIN, CHUNK_MAX = 12, 75  # adaptive free-run chunk bounds (~0.48s .. 3.0s)
STYLE_AUDIO_MAX_SEC = 20  # cap audio used for a style embedding (keeps embed fast)


async def _embed_mix(form):
    """Embed a list of weighted inputs (text or audio) and blend them.

    `form` is the parsed multipart form. It must contain a `spec` field: a JSON
    list of inputs, each like:
        {"type": "text",  "text": "...",   "weight": 1.0, "seed": 0, "use_mapper": false}
        {"type": "audio", "file": "file0", "weight": 0.5, ...}
    Audio inputs reference an uploaded file by its form field name.

    Returns (mixed_embedding_ndarray, sources_list) or (None, None) if empty.
    The blend is a weighted average  Σ wᵢ·eᵢ / Σ|wᵢ|, rescaled to the weighted
    mean of the input norms so it stays in the embeddings' natural magnitude.
    A single positive-weight input reproduces its embedding exactly.
    """
    from magenta_rt import audio
    try:
        spec = json.loads(form.get("spec") or "[]")
    except Exception:  # noqa: BLE001
        spec = []

    # Phase 1 (async): collect inputs and read any uploaded audio bytes.
    items = []  # (kind, payload, label, weight, seed, use_mapper)
    for inp in spec:
        try:
            w = float(inp.get("weight", 1.0))
        except Exception:  # noqa: BLE001
            w = 1.0
        if w == 0:
            continue
        seed = int(inp.get("seed", 0))
        use_mapper = bool(inp.get("use_mapper", False))
        if inp.get("type") == "audio":
            up = form.get(inp.get("file") or "")
            if up is None or not getattr(up, "filename", None):
                continue
            raw = await up.read()
            items.append(("audio", raw, up.filename, w, seed, use_mapper))
        else:
            text = (inp.get("text") or "").strip()
            if not text:
                continue
            items.append(("text", text, text, w, seed, use_mapper))

    if not items:
        return None, None

    # Phase 2 (thread): decode audio + run MusicCoCa + blend. Done OFF the event
    # loop so the /stream websocket keeps sending while a long clip embeds — the
    # style is only applied once its embedding is ready, never blocking playback.
    def _compute():
        with _embed_lock:
            mc = get_embed_mc()
            vecs, weights, sources = [], [], []
            for kind, payload, label, w, seed, use_mapper in items:
                if kind == "audio":
                    wav = audio.Waveform.from_file(io.BytesIO(payload))
                    maxn = int(STYLE_AUDIO_MAX_SEC * wav.sample_rate)
                    if wav.samples.shape[0] > maxn:          # use only the first N seconds
                        wav = audio.Waveform(wav.samples[:maxn], wav.sample_rate)
                    emb = mc.embed(wav, True, use_mapper, seed)
                    sources.append(f"audio:{label}×{w:g}")
                else:
                    emb = mc.embed(payload, True, use_mapper, seed)
                    sources.append(f"text:{payload!r}×{w:g}")
                vecs.append(np.asarray(emb, dtype=np.float64).ravel())
                weights.append(w)
            V = np.stack(vecs)                              # float64 math avoids overflow
            W = np.asarray(weights, dtype=np.float64)
            denom = float(np.abs(W).sum()) or 1.0
            mix = (W[:, None] * V).sum(axis=0) / denom
            norms = np.linalg.norm(V, axis=1)
            target = float((np.abs(W) * norms).sum() / denom)
            nmix = float(np.linalg.norm(mix))
            if nmix > 1e-8:
                mix = mix / nmix * target
            return mix.astype(np.float32), sources

    return await asyncio.get_event_loop().run_in_executor(_embed_executor, _compute)


def _emb_payload(emb, mc):
    """Return (768-dim vector as list, discrete style tokens as list)."""
    vec = np.asarray(emb, dtype=np.float32).ravel()
    try:
        toks = np.asarray(mc.tokenize(emb)).ravel().astype(int).tolist()
    except Exception:  # noqa: BLE001
        toks = []
    return vec.tolist(), toks


async def _payload_async(emb):
    """Compute the chart payload (vector + tokens) off the event loop."""
    def f():
        with _embed_lock:
            return _emb_payload(emb, get_embed_mc())
    return await asyncio.get_event_loop().run_in_executor(_embed_executor, f)


@app.post("/prepare_style")
async def prepare_style(request: Request):
    """Embed a weighted mix of inputs, cache it, and return a token for /stream."""
    emb, sources = await _embed_mix(await request.form())
    if emb is None:
        return JSONResponse(
            {"error": "Add at least one prompt or audio input with non-zero weight."},
            status_code=400,
        )
    token = f"s{next(_token_counter)}"
    _style_cache[token] = emb
    vec, toks = await _payload_async(emb)
    return {"token": token, "source": " + ".join(sources),
            "embedding": vec, "dim": len(vec), "tokens": toks}


@app.post("/embedding")
async def embedding(request: Request):
    """Compute the mixed style embedding for preview/visualization (no caching)."""
    emb, sources = await _embed_mix(await request.form())
    if emb is None:
        return JSONResponse({"error": "No valid inputs."}, status_code=400)
    vec, toks = await _payload_async(emb)
    return {"source": " + ".join(sources), "embedding": vec, "dim": len(vec), "tokens": toks}


@app.post("/prepare_raw")
async def prepare_raw(request: Request):
    """Cache a raw 768-d style embedding (e.g. from the PCA explorer) and return a token."""
    data = await request.json()
    vec = np.asarray(data.get("embedding", []), dtype=np.float32).ravel()
    if vec.size == 0:
        return JSONResponse({"error": "Empty embedding."}, status_code=400)
    token = f"s{next(_token_counter)}"
    _style_cache[token] = vec
    v, toks = await _payload_async(vec)
    return {"token": token, "source": "pca", "embedding": v, "dim": len(v), "tokens": toks}


@app.post("/pca")
async def pca(request: Request):
    """Principal-component analysis of the stored preset embeddings.

    For N preset vectors, the centered data has rank <= N-1, so we return the
    N-1 highest-variance directions. The client reconstructs an embedding as
    mean + Σ slider_k · component_k.
    """
    data = await request.json()
    X = np.asarray(data.get("vectors", []), dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return {"k": 0, "components": [], "mean": [], "ranges": [], "explained": []}

    mean = X.mean(axis=0)
    Xc = X - mean
    _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = X.shape[0]
    k = n - 1                       # number of principal directions
    comps = Vt[:k]                  # (k, 768) orthonormal directions
    coords = Xc @ comps.T          # (n, k) where each preset sits on each axis

    ranges = []
    for j in range(k):
        col = coords[:, j]
        lo, hi = float(col.min()), float(col.max())
        pad = 0.25 * (hi - lo if hi > lo else 1.0)
        ranges.append([lo - pad, hi + pad])

    total = float((S ** 2).sum()) or 1.0
    explained = [float(s ** 2 / total) for s in S[:k]]

    return {
        "k": int(k),
        "n": int(n),
        "mean": mean.astype(np.float32).tolist(),
        "components": comps.astype(np.float32).tolist(),
        "coords": coords.astype(np.float32).tolist(),
        "ranges": ranges,
        "explained": explained,
    }


@app.get("/landmarks")
def list_landmarks():
    return {"landmarks": [{"id": lm["id"], "description": lm["description"],
                           "polarity": lm.get("polarity", "good"),
                           "embedding": lm["embedding"]} for lm in _landmarks]}


@app.get("/vocab")
async def vocab():
    """Ensure the built-in vocabulary embeddings exist (build+persist on first call)."""
    await ensure_vocab()
    return {"count": len(_vocab), "instruments": VOCAB_INSTRUMENTS, "genres": VOCAB_GENRES}


@app.get("/vocab_points")
async def vocab_points():
    """Vocabulary words with their embeddings, for plotting as dots on the cube."""
    await ensure_vocab()
    return {"points": [{"word": v["word"], "kind": v["kind"], "embedding": v["embedding"]}
                       for v in _vocab.values()]}


@app.post("/landmark")
async def add_landmark(request: Request):
    """Pin the current position with a user description ('chill like beach')."""
    data = await request.json()
    desc = (data.get("description") or "").strip()
    emb = data.get("embedding") or []
    if not desc:
        return JSONResponse({"error": "Describe the place first."}, status_code=400)
    if not isinstance(emb, list) or len(emb) < 8:
        return JSONResponse(
            {"error": "No current position to pin — set a style or press play first."},
            status_code=400,
        )
    pol = "bad" if (data.get("polarity") == "bad") else "good"
    lm = {"id": _next_landmark_id(), "description": desc, "polarity": pol,
          "embedding": [float(x) for x in emb], "created": time.time()}
    with _landmarks_lock:
        _landmarks.append(lm)
        _save_landmarks()
    return {"id": lm["id"], "polarity": pol, "count": len(_landmarks)}


@app.delete("/landmark/{lid}")
def delete_landmark(lid: str):
    global _landmarks
    with _landmarks_lock:
        _landmarks = [lm for lm in _landmarks if lm["id"] != lid]
        _save_landmarks()
    return {"count": len(_landmarks)}


@app.post("/landmark/{lid}/play")
async def play_landmark(lid: str):
    """Cache a saved landmark's embedding as a style and return a token to play it."""
    lm = _landmark_by_id(lid)
    if not lm:
        return JSONResponse({"error": "Unknown landmark."}, status_code=404)
    emb = np.asarray(lm["embedding"], dtype=np.float32)
    token = f"s{next(_token_counter)}"
    _style_cache[token] = emb
    vec, toks = await _payload_async(emb)
    return {"token": token, "source": f'landmark: "{lm["description"]}"',
            "embedding": vec, "dim": len(vec), "tokens": toks}


def _repel_from_bad(emb, bad_lms, strength=0.2):
    """Soft nudge: push a target embedding away from 'avoid' landmarks.

    Scale-free and bounded — total displacement is at most `strength` of the
    vector's norm, weighted by cosine closeness (only repels when the target
    actually points toward a bad place), then renormalized to the original norm.
    """
    if not bad_lms:
        return emb
    v = np.asarray(emb, dtype=np.float64).ravel()
    n0 = float(np.linalg.norm(v)) or 1.0
    vn = v / n0
    push = np.zeros_like(v)
    for lm in bad_lms:
        b = np.asarray(lm["embedding"], dtype=np.float64).ravel()
        if b.shape != v.shape:
            continue
        bn = b / (float(np.linalg.norm(b)) or 1.0)
        s = float(np.dot(vn, bn))            # cosine similarity to the bad place
        if s <= 0:
            continue                          # already pointing away -> no push
        d = v - b
        dn = float(np.linalg.norm(d)) or 1e-9
        push += (d / dn) * (s * s)            # closer (higher cosine) -> stronger
    pn = float(np.linalg.norm(push))
    if pn > 0:
        v = v + strength * n0 * (push / pn)   # bounded nudge
        v = v * (n0 / (float(np.linalg.norm(v)) or 1.0))   # keep original magnitude
    return v.astype(np.float32)


async def _resolve_nav(action, inp):
    """Turn Claude's navigation decision into a 768-d target embedding."""
    if action == "go_to_landmark":
        lm = _landmark_by_id(inp.get("landmark_id"))
        if lm:
            return np.asarray(lm["embedding"], dtype=np.float32), f'landmark: "{lm["description"]}"'
        return None, None
    if action == "blend_landmarks":
        vecs, weights, labels = [], [], []
        for w in inp.get("weights", []):
            lm = _landmark_by_id(w.get("landmark_id"))
            if not lm:
                continue
            vecs.append(np.asarray(lm["embedding"], dtype=np.float64).ravel())
            weights.append(float(w.get("weight", 1.0)))
            labels.append(lm["description"])
        if not vecs:
            return None, None
        V = np.stack(vecs)
        W = np.asarray(weights, dtype=np.float64)
        denom = float(np.abs(W).sum()) or 1.0
        mix = (W[:, None] * V).sum(axis=0) / denom
        norms = np.linalg.norm(V, axis=1)
        target = float((np.abs(W) * norms).sum() / denom)
        nmix = float(np.linalg.norm(mix))
        if nmix > 1e-8:
            mix = mix / nmix * target
        return mix.astype(np.float32), "blend: " + " + ".join(labels)
    if action == "compose_from_vocab":
        vecs, weights, labels = [], [], []
        for t in inp.get("terms", []):
            item = _vocab.get(str(t.get("word", "")).strip().lower())
            if not item:
                continue
            vecs.append(np.asarray(item["embedding"], dtype=np.float64).ravel())
            weights.append(float(t.get("weight", 1.0)))
            labels.append(item["word"])
        if not vecs:
            return None, None
        V = np.stack(vecs)
        W = np.asarray(weights, dtype=np.float64)
        denom = float(np.abs(W).sum()) or 1.0
        mix = (W[:, None] * V).sum(axis=0) / denom
        norms = np.linalg.norm(V, axis=1)
        target = float((np.abs(W) * norms).sum() / denom)
        nmix = float(np.linalg.norm(mix))
        if nmix > 1e-8:
            mix = mix / nmix * target
        return mix.astype(np.float32), "vocab: " + " + ".join(labels)
    if action == "rewrite_to_music":
        phrase = (inp.get("phrase") or "").strip()
        if not phrase:
            return None, None

        def f():
            with _embed_lock:
                e = get_embed_mc().embed(phrase, True, False, 0)
                return np.asarray(e, dtype=np.float32).ravel()
        emb = await asyncio.get_event_loop().run_in_executor(_embed_executor, f)
        return emb, f"rewrite: “{phrase}”"
    return None, None


@app.post("/navigate")
async def navigate(request: Request):
    """Feeling -> Claude picks an action over the landmark atlas -> target embedding."""
    data = await request.json()
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "Type where you want to go."}, status_code=400)

    await ensure_vocab()
    good_lms = [lm for lm in _landmarks if lm.get("polarity", "good") != "bad"]
    bad_lms = [lm for lm in _landmarks if lm.get("polarity") == "bad"]
    atlas = "\n".join(f'- {lm["id"]}: "{lm["description"]}"' for lm in good_lms) or "(none saved yet)"
    avoid_txt = "\n".join(f'- "{lm["description"]}"' for lm in bad_lms)
    vocab_txt = "instruments: " + ", ".join(VOCAB_INSTRUMENTS) + "\ngenres: " + ", ".join(VOCAB_GENRES)
    user_msg = (f"Saved places:\n{atlas}\n\n"
                + (f"AVOID these places — do NOT travel to or toward them:\n{avoid_txt}\n\n" if bad_lms else "")
                + f"Vocabulary (compose_from_vocab must use only these words):\n{vocab_txt}\n\n"
                f'Request: "{prompt}"')
    try:
        client = get_anthropic()
        msg = await client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            system=NAV_SYSTEM,
            tools=NAV_TOOLS,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:  # noqa: BLE001
        detail = getattr(e, "message", None) or str(e)
        return JSONResponse({"error": f"Claude API: {detail}"}, status_code=502)

    tu = next((b for b in msg.content if getattr(b, "type", None) == "tool_use"), None)
    if tu is None:
        return JSONResponse({"error": "No navigation decision returned."}, status_code=502)

    emb, source = await _resolve_nav(tu.name, dict(tu.input))
    if emb is None:
        return JSONResponse({"error": f"Could not resolve action '{tu.name}'."}, status_code=422)

    emb = _repel_from_bad(emb, bad_lms)   # soft nudge away from 'avoid' landmarks
    token = f"s{next(_token_counter)}"
    _style_cache[token] = emb
    vec, toks = await _payload_async(emb)
    reasoning = tu.input.get("reasoning", "") if isinstance(tu.input, dict) else ""
    return {"token": token, "action": tu.name, "reasoning": reasoning, "source": source,
            "embedding": vec, "dim": len(vec), "tokens": toks}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        init = await ws.receive_json()
    except Exception:  # noqa: BLE001
        await ws.close()
        return

    style = _style_cache.get(init.get("token"))
    if style is None:
        await ws.send_json({"type": "error", "msg": "Unknown or missing style token."})
        await ws.close()
        return

    # Live-updatable control block, seeded from the init message.
    # `seq` is the step sequencer: {steps, fps, notes: [[midi,...] per step],
    #  drums: [0/1 per step]} or None for free-running generation.
    ctl = {
        "style": style,
        "stop": False,
        "temperature": float(init.get("temperature", 1.3)),
        "top_k": int(init.get("top_k", 40)),
        "cfg_musiccoca": float(init.get("cfg_musiccoca", 3.0)),
        "cfg_notes": float(init.get("cfg_notes", 1.0)),
        "cfg_drums": float(init.get("cfg_drums", 1.0)),
        "seq": init.get("seq"),
        "underrun": 0,  # incremented by the client when its audio buffer starves
    }

    async def receiver():
        try:
            while True:
                msg = await ws.receive_json()
                t = msg.get("type")
                if t == "stop":
                    ctl["stop"] = True
                    return
                if t == "params":
                    for k in ("temperature", "cfg_musiccoca", "cfg_notes", "cfg_drums"):
                        if k in msg:
                            ctl[k] = float(msg[k])
                    if "top_k" in msg:
                        ctl["top_k"] = int(msg["top_k"])
                if t == "style":
                    new = _style_cache.get(msg.get("token"))
                    if new is not None:
                        ctl["style"] = new
                if t == "seq":
                    ctl["seq"] = msg.get("seq")  # may be None to go free-running
                if t == "underrun":
                    ctl["underrun"] = ctl.get("underrun", 0) + 1
        except (WebSocketDisconnect, Exception):  # noqa: BLE001
            ctl["stop"] = True

    recv_task = asyncio.create_task(receiver())
    loop = asyncio.get_event_loop()

    if not _gen_lock.acquire(blocking=False):
        await ws.send_json({"type": "error", "msg": "Another stream is active."})
        await ws.close()
        recv_task.cancel()
        return

    mrt = get_model()
    num_notes = mrt._num_notes
    drum_tokens = mrt._drum_tokens
    state = None
    step = 0

    # --- Adaptive pacing & chunk sizing -------------------------------------
    # Apply new input as fast as possible (small chunk + small buffer lead)
    # while never starving the audio buffer (no jitter). The server times its
    # own generation and: (a) sizes the free-run chunk so generation stays
    # comfortably faster than real time, (b) keeps a buffer "lead" covering the
    # worst recent generation time, (c) self-paces (sleeps) so it never runs
    # more than that lead ahead — which also stops latency from creeping up.
    chunk = max(CHUNK_MIN, min(CHUNK_MAX, int(init.get("chunk", STREAM_FRAMES))))
    rtf_hist = deque(maxlen=10)   # recent real-time factors (gen / audio)
    gen_hist = deque(maxlen=10)   # recent generation times (seconds)
    target_lead = 0.40            # seconds of audio to keep buffered ahead
    t_start = None                # ~ when playback began (first chunk sent)
    audio_sent = 0.0              # cumulative seconds of audio sent

    try:
        await ws.send_json({"type": "ready", "sample_rate": int(mrt._sample_rate)})
        while not ctl["stop"]:
            seq = ctl["seq"]
            seq_mode = bool(seq and seq.get("steps", 0) > 0)
            if seq_mode:
                n_steps = int(seq["steps"])
                i = step % n_steps
                frames = max(1, int(seq.get("fps", STREAM_FRAMES)))
                active = seq.get("notes", [])
                active_i = active[i] if i < len(active) else []
                notes = [-1] * num_notes
                for p in active_i:
                    if 0 <= int(p) < num_notes:
                        notes[int(p)] = 2  # onset
                drums_list = seq.get("drums", [])
                drum_on = bool(drums_list[i]) if i < len(drums_list) else False
                drums = [1 if drum_on else 0] * drum_tokens
                step += 1
                await ws.send_json({"type": "step", "i": i})
            else:
                notes = None
                drums = None
                frames = chunk
            audio_dur = frames / 25.0

            def _gen(_state):
                return mrt.generate(
                    style=ctl["style"], notes=notes, drums=drums,
                    cfg_musiccoca=ctl["cfg_musiccoca"], cfg_notes=ctl["cfg_notes"],
                    cfg_drums=ctl["cfg_drums"], temperature=ctl["temperature"],
                    top_k=ctl["top_k"], frames=frames, state=_state,
                )

            t0 = time.time()
            wav, state = await loop.run_in_executor(_gen_executor, _gen, state)
            gen_s = time.time() - t0
            rtf = gen_s / audio_dur if audio_dur > 0 else 0.0
            gen_hist.append(gen_s)
            rtf_hist.append(rtf)

            samples = np.asarray(wav.samples, dtype=np.float32)
            if samples.ndim == 1:
                samples = np.stack([samples, samples], axis=-1)
            elif samples.shape[1] == 1:
                samples = np.repeat(samples, 2, axis=1)
            # row-major [nsamp, 2] flattens to interleaved L,R,L,R...
            await ws.send_bytes(samples.reshape(-1).astype("<f4").tobytes())

            if t_start is None:
                t_start = t0
            audio_sent += audio_dur

            # Client reported an actual buffer gap -> back off hard.
            if ctl.get("underrun", 0) > 0:
                ctl["underrun"] = 0
                chunk = min(CHUNK_MAX, chunk + 10)
                target_lead = min(1.5, target_lead + 0.20)

            # Size the buffer lead to cover the worst recent generation time
            # (must be >= one generation, or the buffer drains while we compute).
            target_lead = min(2.5, max(0.12, 1.3 * max(gen_hist)))

            # Adapt free-run chunk: push it DOWN to cut latency while RTF has
            # headroom; push it UP when generation gets too close to real time.
            if not seq_mode and len(rtf_hist) >= 3:
                rmax = max(rtf_hist)
                if rmax > 0.90:
                    chunk = min(CHUNK_MAX, chunk + 6)   # near real-time: bigger = safer & more efficient
                elif rmax < 0.85:
                    chunk = max(CHUNK_MIN, chunk - 4)   # headroom: smaller = faster response

            # Self-pace: never get more than target_lead ahead of real time.
            elapsed = time.time() - t_start
            lead = audio_sent - elapsed
            if lead > target_lead:
                await asyncio.sleep(lead - target_lead)

            await ws.send_json({
                "type": "perf",
                "rtf": round(rtf, 2),
                "gen_ms": round(gen_s * 1000),
                "chunk_ms": round(audio_dur * 1000),
                "lead_ms": round(max(0.0, lead) * 1000),
                "target_ms": round(target_lead * 1000),
                "auto": not seq_mode,
                "warn": rtf >= 0.98 or lead < 0.05,
            })
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        import traceback; traceback.print_exc()
    finally:
        ctl["stop"] = True
        _gen_lock.release()
        recv_task.cancel()
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Groove app: serve the pre-built static bundle from groove_dist/.
# The bundle uses ABSOLUTE ROOT urls — index.html loads /assets/index-*.js and at
# runtime fetches /straight_drums.wav, /amen.wav, /demoSongA.wav, /demoSongB.wav
# and /groove_library.json — so those EXACT paths must be served at the root.
# Routes live in this server's clean namespace (no collision with the API routes).
# ---------------------------------------------------------------------------
GROOVE_DIST = HERE / "groove_dist"
GROOVE_ROOT_FILES = ["straight_drums.wav", "amen.wav", "demoSongA.wav", "demoSongB.wav", "groove_library.json"]


@app.get("/groove")
def groove_index():
    """Iframe entry point: the pre-built groove app's index.html."""
    index = GROOVE_DIST / "index.html"
    if not index.is_file():
        return JSONResponse({"error": "groove app not built (groove_dist/index.html missing)."}, status_code=404)
    return FileResponse(index)


def _make_groove_root_route(fn):
    """Factory: each closure captures its OWN fn (avoids the late-binding loop bug)."""
    def route():
        path = GROOVE_DIST / fn
        if not path.is_file():
            return JSONResponse({"error": f"{fn} not found."}, status_code=404)
        return FileResponse(path)
    return route


for _fn in GROOVE_ROOT_FILES:
    app.add_api_route("/" + _fn, _make_groove_root_route(_fn), methods=["GET"])

# Mount the hashed JS/CSS assets only if present, so an absent groove_dist/ can't crash startup.
if (GROOVE_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=GROOVE_DIST / "assets"), name="groove-assets")

app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
