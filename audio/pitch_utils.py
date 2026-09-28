"""Pure signal-processing helpers: audio chunk -> dominant frequency -> command.

No I/O (no pyaudio, no hardware) so these are fully unit-testable with
synthetic sine waves.
"""

import numpy as np
from scipy.signal.windows import hann

from config import (
    FREQ_BANDS,
    MAX_WHISTLE_HZ,
    MIN_WHISTLE_HZ,
    SAMPLE_RATE,
    SILENCE_AMPLITUDE_THRESHOLD,
)


def magnitude_spectrum(samples, sample_rate=SAMPLE_RATE):
    """Return (freqs, magnitude) of `samples`' Hann-windowed FFT magnitude spectrum."""
    windowed = samples * hann(len(samples), sym=False)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    return freqs, spectrum


def signal_rms(samples):
    """Return the root-mean-square amplitude of `samples`.

    This is exactly what `dominant_frequency` gates on internally; exposed
    separately so callers (e.g. a live display) can show the actual number
    next to SILENCE_AMPLITUDE_THRESHOLD to tune it against a real mic/room.
    """
    return float(np.sqrt(np.mean(np.square(samples))))


def dominant_frequency(
    samples,
    sample_rate=SAMPLE_RATE,
    min_amplitude=SILENCE_AMPLITUDE_THRESHOLD,
    min_freq=MIN_WHISTLE_HZ,
    max_freq=MAX_WHISTLE_HZ,
):
    """Return the loudest frequency in `samples`, or None if it's not a whistle.

    `samples` is mono float audio in [-1, 1] (matching spectogram.py's
    convention of int16 samples normalized by np.iinfo(np.int16).max).
    Returns None when the chunk is too quiet (RMS below `min_amplitude`), or
    when the chunk's dominant frequency falls outside [min_freq, max_freq]
    -- e.g. loud low-frequency room noise -- both treated as "no whistle" by
    the caller. The range check is against the true dominant frequency
    rather than restricting the search to the range, so loud out-of-range
    noise can't be mistaken for a quiet in-range whistle.
    """
    if signal_rms(samples) < min_amplitude:
        return None

    freqs, spectrum = magnitude_spectrum(samples, sample_rate)

    peak_freq = float(freqs[np.argmax(spectrum)])
    if peak_freq < min_freq or peak_freq > max_freq:
        return None
    return peak_freq


def frequency_to_command(freq, bands=FREQ_BANDS):
    """Map a frequency (or None for silence) to a command name using `bands`.

    `bands` is a list of (low_hz, high_hz, command) tuples. Falls back to
    "stop" for silence or a frequency outside every band, so an
    unrecognized or missing whistle always defaults to safe behavior.
    """
    if freq is None:
        return "stop"
    for low, high, command in bands:
        if low <= freq < high:
            return command
    return "stop"
