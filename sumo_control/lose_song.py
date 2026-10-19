"""The losing song: a "sad trombone" (wah wah wah waaah) made from scratch,
so it needs no internet or music file. Set LOSE_SONG_FILE in config.py to
play your own sound file instead (macOS).

    python sumo_control/lose_song.py     # hear it
"""

import os
import subprocess
import threading

import numpy as np
import sounddevice as sd

from config import LOSE_SONG_FILE

RATE = 44100
# (frequency Hz, seconds): three falling notes, then a long wobbly low one
NOTES = [(392, 0.45), (370, 0.45), (349, 0.45), (330, 1.4)]


def _note(freq, secs, wobble=False):
    t = np.arange(int(RATE * secs)) / RATE
    if wobble:
        freq = freq * (1 + 0.012 * np.sin(2 * np.pi * 6 * t))  # trombone "wah-wah"
    phase = 2 * np.pi * np.cumsum(freq) / RATE if wobble else 2 * np.pi * freq * t
    # a few harmonics so it sounds brassy instead of like a beep
    wave = sum(np.sin(k * phase) / k for k in (1, 2, 3, 4))
    fade = np.minimum(1, np.minimum(t / 0.03, (secs - t) / 0.08))  # no clicks
    return wave * fade


def _song():
    parts = [_note(f, s, wobble=(i == len(NOTES) - 1)) for i, (f, s) in enumerate(NOTES)]
    audio = np.concatenate(parts)
    return (0.25 * audio / np.max(np.abs(audio))).astype(np.float32)


def play(wait=False):
    """Play the losing song. Returns right away unless wait=True."""
    def go():
        try:
            if LOSE_SONG_FILE and os.path.exists(LOSE_SONG_FILE):
                subprocess.run(["afplay", LOSE_SONG_FILE], check=False)
            else:
                sd.play(_song(), RATE)
                sd.wait()
        except Exception as e:  # never let a sound problem crash the robot
            print(f"(couldn't play the losing song: {e})")

    t = threading.Thread(target=go, daemon=True)
    t.start()
    if wait:
        t.join()


if __name__ == "__main__":
    play(wait=True)
