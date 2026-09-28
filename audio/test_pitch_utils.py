import numpy as np
import pytest

from pitch_utils import (
    dominant_frequency,
    frequency_to_command,
    magnitude_spectrum,
    signal_rms,
)

SAMPLE_RATE = 44100


def sine_wave(freq, duration=0.1, sample_rate=SAMPLE_RATE, amplitude=0.5):
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("freq", [440.0, 660.0, 880.0, 1500.0])
def test_dominant_frequency_detects_known_tone(freq):
    detected = dominant_frequency(sine_wave(freq))
    assert detected == pytest.approx(freq, abs=20)


def test_dominant_frequency_none_for_silence():
    samples = np.zeros(4096, dtype=np.float32)
    assert dominant_frequency(samples) is None


def test_dominant_frequency_none_for_quiet_noise():
    rng = np.random.default_rng(0)
    samples = rng.normal(scale=0.001, size=4096).astype(np.float32)
    assert dominant_frequency(samples) is None


def test_dominant_frequency_ignores_tone_outside_whistle_range():
    # A loud but low-pitched tone (e.g. room rumble) shouldn't register.
    samples = sine_wave(100.0)
    assert dominant_frequency(samples, min_freq=500, max_freq=2000) is None


def test_dominant_frequency_respects_custom_amplitude_threshold():
    samples = sine_wave(440.0, amplitude=0.05)
    assert dominant_frequency(samples, min_amplitude=0.5) is None


def test_frequency_to_command_maps_band():
    bands = [(500, 750, "backward"), (750, 1000, "forward")]
    assert frequency_to_command(600, bands) == "backward"
    assert frequency_to_command(900, bands) == "forward"


def test_frequency_to_command_defaults_to_stop_outside_bands():
    bands = [(500, 750, "backward")]
    assert frequency_to_command(2000, bands) == "stop"


def test_frequency_to_command_defaults_to_stop_for_silence():
    bands = [(500, 750, "backward")]
    assert frequency_to_command(None, bands) == "stop"


@pytest.mark.parametrize(
    "freq,expected",
    [(600, "backward"), (749.9, "backward"), (750, "forward")],
)
def test_frequency_to_command_band_boundaries(freq, expected):
    bands = [(500, 750, "backward"), (750, 1000, "forward")]
    assert frequency_to_command(freq, bands) == expected


def test_frequency_to_command_uses_config_bands_by_default():
    assert frequency_to_command(700) == "backward"
    assert frequency_to_command(None) == "stop"


def test_frequency_to_command_high_whistle_maps_to_goal():
    assert frequency_to_command(1350) == "goal"


def test_magnitude_spectrum_peak_matches_known_tone():
    freqs, magnitude = magnitude_spectrum(sine_wave(880.0))
    assert freqs[np.argmax(magnitude)] == pytest.approx(880.0, abs=20)


def test_magnitude_spectrum_shapes_match():
    freqs, magnitude = magnitude_spectrum(sine_wave(440.0, duration=0.05))
    assert freqs.shape == magnitude.shape


def test_signal_rms_of_silence_is_zero():
    assert signal_rms(np.zeros(1024, dtype=np.float32)) == 0.0


def test_signal_rms_matches_known_sine_amplitude():
    # RMS of a pure sine of amplitude A is A / sqrt(2).
    samples = sine_wave(440.0, amplitude=1.0)
    assert signal_rms(samples) == pytest.approx(1.0 / np.sqrt(2), abs=1e-3)


def test_signal_rms_scales_linearly_with_amplitude():
    quiet = signal_rms(sine_wave(440.0, amplitude=0.01))
    loud = signal_rms(sine_wave(440.0, amplitude=0.1))
    assert loud == pytest.approx(quiet * 10, rel=1e-3)
