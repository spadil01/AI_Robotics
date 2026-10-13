"""
Generates the game's sound effects in assets/sounds/ from scratch (simple
synthesis with numpy). The crowd sounds (cheer, aww, clap) come from real
recordings instead: see prepare_crowd.py.

Run from the repo root after changing anything here:
    .venv/bin/python table_tennis/tools/make_sounds.py
"""

import wave
from pathlib import Path

import numpy as np

RATE = 44100
OUT = Path(__file__).resolve().parent.parent / "assets" / "sounds"
rng = np.random.default_rng(42)


def t_axis(seconds):
    return np.arange(int(RATE * seconds)) / RATE


def noise(seconds):
    return rng.uniform(-1, 1, int(RATE * seconds))


def band(signal, low, high):
    """Keep only frequencies between low and high (Hz), via FFT."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / RATE)
    spectrum[(freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spectrum, len(signal))


def tone(freq, t, harmonics=(1.0,)):
    """Sum of harmonics; `freq` may be an array (pitch changing over time)."""
    phase = 2 * np.pi * np.cumsum(np.broadcast_to(freq, t.shape)) / RATE
    return sum(a * np.sin((i + 1) * phase) for i, a in enumerate(harmonics))


def fade(signal, start=0.005, end=0.02):
    n0, n1 = int(RATE * start), int(RATE * end)
    signal = signal.copy()
    signal[:n0] *= np.linspace(0, 1, n0)
    signal[-n1:] *= np.linspace(1, 0, n1)
    return signal


def save(name, signal, peak=0.9):
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * peak
    data = (signal * 32767).astype(np.int16)
    OUT.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT / f"{name}.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(data.tobytes())
    print(f"  {name}.wav  ({len(signal) / RATE:.2f} s)")


# --- ball sounds ---------------------------------------------------------------

def hit():
    """Paddle 'tock': a woody knock with a tiny click."""
    t = t_axis(0.14)
    f = 1050 * (1 + 0.15 * np.exp(-t * 60))
    body = tone(f, t, (1.0, 0.0, 0.35)) * np.exp(-t * 45)
    knock = np.sin(2 * np.pi * 260 * t) * np.exp(-t * 55) * 0.6
    click = band(noise(0.14), 2000, 9000) * np.exp(-t * 500) * 0.8
    return fade(body + knock + click, 0.0005)


def bounce():
    """Table 'tik': higher and shorter, with a hollow thump from the table."""
    t = t_axis(0.09)
    ping = tone(1750, t, (1.0, 0.0, 0.25)) * np.exp(-t * 75)
    thump = np.sin(2 * np.pi * 210 * t) * np.exp(-t * 45) * 0.55
    click = band(noise(0.09), 3000, 10000) * np.exp(-t * 700) * 0.5
    return fade(ping + thump + click, 0.0005)


def net():
    t = t_axis(0.22)
    thud = np.sin(2 * np.pi * 130 * t) * np.exp(-t * 22)
    rustle = band(noise(0.22), 300, 2500) * np.exp(-t * 18) * 0.5
    return fade(thud + rustle, 0.001)


def floor():
    t = t_axis(0.08)
    return fade(tone(620, t, (1.0, 0.3)) * np.exp(-t * 65) + band(noise(0.08), 800, 4000) * np.exp(-t * 300) * 0.4,
                0.0005)


# --- jingles and UI ------------------------------------------------------------------

def notes(sequence, note_len, hold=0.3, shape=(1.0, 0.4, 0.15)):
    """Little chime melody: list of frequencies (Hz), each note ringing out."""
    total = note_len * len(sequence) + hold
    t = t_axis(total)
    out = np.zeros_like(t)
    for i, f in enumerate(sequence):
        start = i * note_len
        local = np.clip(t - start, 0, None)
        env = (t >= start) * np.exp(-local * 6)
        out += tone(f, t, shape) * env
    return fade(out, 0.002, 0.1)


C5, E5, G5, C6 = 523.25, 659.25, 783.99, 1046.5
C4, Eb4, G4 = 261.63, 311.13, 392.0


def select():
    t = t_axis(0.14)
    f = np.where(t < 0.06, 880, 1320)
    return fade(tone(f, t, (1.0, 0.2)) * np.exp(-t * 18), 0.002, 0.03)


SOUNDS = {
    "hit": hit, "bounce": bounce, "net": net, "floor": floor,
    "point_won": lambda: notes([C5, E5, G5], 0.09), "point_lost": lambda: notes([G4, Eb4], 0.14),
    "match_won": lambda: notes([C5, E5, G5, C6, G5, C6], 0.11, hold=0.8),
    "match_lost": lambda: notes([G4, Eb4, C4], 0.22, hold=0.8),
    "select": select,
}


def main():
    print(f"Writing sounds to {OUT}")
    for name, make in SOUNDS.items():
        save(name, make())


if __name__ == "__main__":
    main()
