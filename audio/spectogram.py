"""Record audio with PyAudio and plot its spectrogram, optionally labeling musical notes."""

import argparse

import numpy as np
import pyaudio
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def record_audio(duration, sample_rate=SAMPLE_RATE):
    """Record `duration` seconds of mono audio from the default input device."""
    input(f"Press Enter to start recording ({duration} seconds)...")

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print(f"Recording for {duration} seconds...")
    frames = []
    for _ in range(int(sample_rate / CHUNK * duration)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
    print("Done recording.")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
    audio /= np.iinfo(np.int16).max
    return audio


def compute_spectrogram(audio, sample_rate=SAMPLE_RATE, window_size=1024, hop=256):
    """Return (times, freqs, magnitude_db) for the given audio signal."""
    window = hann(window_size)
    sft = ShortTimeFFT(window, hop=hop, fs=sample_rate, scale_to="magnitude")
    magnitude = np.abs(sft.stft(audio))
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))

    times = sft.t(len(audio))
    freqs = sft.f
    return times, freqs, magnitude_db


def frequency_to_note(freq, a4=440.0):
    """Convert a frequency in Hz to the nearest musical note name and octave."""
    if freq <= 0:
        return None
    semitones_from_a4 = round(12 * np.log2(freq / a4))
    midi_number = 69 + int(semitones_from_a4)
    note_index = midi_number % 12
    octave = midi_number // 12 - 1
    return f"{NOTE_NAMES[note_index]}{octave}"


def label_notes(ax, times, freqs, magnitude_db, max_freq, num_labels=15):
    """Annotate the dominant note at evenly spaced time steps, ignoring near-silent frames."""
    freq_mask = freqs <= max_freq
    step = max(1, len(times) // num_labels)
    loudest_overall = magnitude_db.max()

    for i in range(0, len(times), step):
        frame = magnitude_db[freq_mask, i]
        peak_idx = np.argmax(frame)
        peak_db = frame[peak_idx]
        if peak_db < loudest_overall - 40:
            continue

        peak_freq = freqs[freq_mask][peak_idx]
        note = frequency_to_note(peak_freq)
        if note is None:
            continue

        ax.annotate(
            note,
            xy=(times[i], peak_freq),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="white",
            weight="bold",
        )
        ax.plot(times[i], peak_freq, "o", color="white", markersize=3)


def plot_spectrogram(times, freqs, magnitude_db, max_freq, label=False):
    freq_mask = freqs <= max_freq

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(
        times,
        freqs[freq_mask],
        magnitude_db[freq_mask],
        shading="gouraud",
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spectrogram")

    if label:
        label_notes(ax, times, freqs, magnitude_db, max_freq)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0, help="Recording length in seconds")
    parser.add_argument("--max-freq", type=float, default=2000.0, help="Highest frequency to display (Hz)")
    parser.add_argument("--label-notes", action="store_true", help="Annotate detected musical notes on the plot")
    args = parser.parse_args()

    audio = record_audio(args.duration)
    times, freqs, magnitude_db = compute_spectrogram(audio)
    plot_spectrogram(times, freqs, magnitude_db, args.max_freq, label=args.label_notes)


if __name__ == "__main__":
    main()
