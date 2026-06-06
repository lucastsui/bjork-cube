#!/usr/bin/env python3
"""Regenerate the clean kick-beat tempo references: beat_50bpm.wav ... beat_200bpm.wav.

These ~151 WAVs (~133 MB) are gitignored — the server reads them from disk and
embeds them as a full-weight tempo audio prompt (driven by the Style "tempo" slider).
Reproduce them anytime with:  python tools/make_tempo_beats.py
(needs numpy + soundfile, both in the project venv).
"""
import os
import numpy as np

try:
    import soundfile as sf
    def write(p, y, sr): sf.write(p, y, sr)
except Exception:  # fallback to the magenta_rt waveform writer
    from magenta_rt import audio as _a
    def write(p, y, sr): _a.Waveform(y[:, None], sr).write(p, format="WAV")

SR = 48000

def kick(dur=0.22):
    """A clean, smooth, punchy kick (pure tone body, tiny fade-in, no noise)."""
    n = int(dur * SR); t = np.arange(n) / SR
    f = 45 + (160 - 45) * np.exp(-t / 0.02)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.09)
    a = int(0.002 * SR); body[:a] *= np.linspace(0, 1, a)
    return body

def render(bpm, seconds=9.6):
    beat = 60 / bpm; bps = int(beat * SR); out = np.zeros(int(seconds * SR) + SR)
    for i in range(int(seconds / beat) + 1):       # one kick per beat, evenly spaced
        k = kick(); s = i * bps
        if s + len(k) <= len(out): out[s:s + len(k)] += k
    out = out[:int(seconds * SR)]; out /= (np.max(np.abs(out)) + 1e-6); out *= 0.95
    return out.astype(np.float32)

if __name__ == "__main__":
    d = os.path.join(os.path.dirname(__file__), "..", "tempo_refs")
    os.makedirs(d, exist_ok=True)
    for bpm in range(50, 201):
        write(os.path.join(d, f"beat_{bpm}bpm.wav"), render(bpm), SR)
    print("wrote 151 tempo beats (50-200 bpm) to", os.path.abspath(d))
