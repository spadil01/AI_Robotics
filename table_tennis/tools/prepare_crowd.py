"""
Turns the real crowd recordings in assets/sounds/recordings/ into the game's
cheer / aww / clap sounds: keeps the best part of each, fades the ends so
nothing cuts off suddenly, converts to the game's format (mono, 44.1 kHz,
16-bit) and evens out the loudness.

Run from the repo root after changing a recording or the cut points below:
    .venv/bin/python table_tennis/tools/prepare_crowd.py
"""

import wave
from pathlib import Path

import numpy as np

SOUNDS = Path(__file__).resolve().parent.parent / "assets" / "sounds"
RECORDINGS = SOUNDS / "recordings"
RATE = 44100
TARGET_RMS = 0.18          # same loudness for every crowd sound
MAX_PEAK = 0.95

# name: (start s, end s, fade-in s, fade-out s), chosen by measuring each recording's loudness
CUTS = {
    "cheer": (0.0, 3.04, 0.15, 0.8),     # loud the whole way, but stops abruptly: fade it out
    "aww": (0.0, 2.0, 0.05, 0.3),        # already rises and dies away on its own
    "clap": (0.35, 4.0, 0.1, 1.4),       # skip the quiet start, keep the loudest ~3.5 s of the 8 s
}


def resample(x, rate):
    """Change the sample rate by keeping only the frequencies the new rate can hold (FFT)."""
    n_new = int(round(len(x) * RATE / rate))
    spectrum = np.fft.rfft(x)[:n_new // 2 + 1]
    return np.fft.irfft(spectrum, n_new) * n_new / len(x)


def load(path):
    with wave.open(str(path)) as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, "<i2") / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)
    elif width == 4:
        x = np.frombuffer(raw, "<i4") / float(1 << 31)
    else:
        raise ValueError(f"{path.name}: unsupported sample size {width} bytes")
    return x.reshape(-1, channels).mean(axis=1), rate      # mix to mono


def main():
    for name, (start, end, fade_in, fade_out) in CUTS.items():
        x, rate = load(RECORDINGS / f"{name}.wav")
        if rate != RATE:
            x = resample(x, rate)
        x = x[int(start * RATE):int(end * RATE)].copy()
        n_in, n_out = int(fade_in * RATE), int(fade_out * RATE)
        x[:n_in] *= np.linspace(0, 1, n_in)
        x[-n_out:] *= np.linspace(1, 0, n_out) ** 2          # curved fade sounds more natural
        loud = np.sort(np.abs(x))[int(len(x) * 0.5):]          # the louder half, so fades don't skew it
        x *= TARGET_RMS / np.sqrt(np.mean(loud ** 2))
        x *= min(1.0, MAX_PEAK / np.max(np.abs(x)))
        with wave.open(str(SOUNDS / f"{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes((x * 32767).astype("<i2").tobytes())
        print(f"  {name}.wav  {len(x) / RATE:.2f} s")


if __name__ == "__main__":
    main()
