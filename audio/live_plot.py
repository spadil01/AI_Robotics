"""Continuous mic capture with a live waveform, scrolling spectrogram, FFT
magnitude, and detected pitch/command overlay.

No MQTT or light sensor here -- this is purely for validating pitch
detection (including against real room noise). Run directly:

    python audio/live_plot.py

Close the plot window (or Ctrl+C in the terminal) to stop. Pass a connected
`motor` to main() (see whistle_control_robot.py) to also drive a real
DoubleMotor from the same detected command/speed while showing the plot.
"""

import queue

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from audio_stream import AudioStreamer
from command_state import CommandDebouncer, SpeedRamper, resolve_speed
from config import (
    CHUNK,
    DISPLAY_MAX_FREQ_HZ,
    PLOT_REFRESH_MS,
    SAMPLE_RATE,
    SILENCE_AMPLITUDE_THRESHOLD,
    SPECTROGRAM_HISTORY_CHUNKS,
)
from motor_commands import command_to_motor
from pitch_utils import dominant_frequency, frequency_to_command, magnitude_spectrum, signal_rms
from spectogram import frequency_to_note

# Color per command, so the big status panel is readable at a glance.
COMMAND_COLORS = {
    "forward": "tab:green",
    "backward": "tab:orange",
    "left": "tab:blue",
    "right": "tab:purple",
    "stop": "tab:red",
    "goal": "tab:pink",
}


def push_spectrogram_column(image, column):
    """Shift `image` (freq_bins x time) left by one column and insert `column` at the end."""
    shifted = np.roll(image, -1, axis=1)
    shifted[:, -1] = column
    return shifted


def main(motor=None, on_chunk=None, gate=None):
    """Show the live plot. If `motor` is given (a connected DoubleMotor-like
    object), also drive it from the same detected command/speed each chunk.

    If `on_chunk` is given, it's called as `on_chunk(command, speed)` once
    per processed chunk; returning a truthy value closes the plot and
    returns from `main()` (e.g. so a caller can end the run on a stop
    trigger without main() needing to know what that trigger means).

    If `gate` (a threading.Event) is given and not yet set, the plot still
    runs and shows live detection as normal, but the motor isn't driven and
    `on_chunk` isn't called -- a "waiting" stage the caller can flip by
    setting the event (e.g. once an MQTT "start" message arrives) without
    needing to reopen or reconnect anything.
    """
    streamer = AudioStreamer()
    debouncer = CommandDebouncer()
    ramper = SpeedRamper()
    chunk_duration_s = CHUNK / SAMPLE_RATE

    freqs = np.fft.rfftfreq(CHUNK, d=1.0 / SAMPLE_RATE)
    display_mask = freqs <= DISPLAY_MAX_FREQ_HZ
    display_freqs = freqs[display_mask]

    spectrogram_image = np.zeros((len(display_freqs), SPECTROGRAM_HISTORY_CHUNKS), dtype=np.float32)
    last_display_magnitude = np.zeros(len(display_freqs), dtype=np.float32)
    # Lowest/highest frequency detected all session, to help tune FREQ_BANDS
    # in config.py to the whistler's actual range instead of a guess.
    session_freq_range = [None, None]
    # Holds the deferred-close timer (see below) so it isn't garbage
    # collected before it fires -- a purely local reference inside update()
    # can be collected the moment update() returns, silently cancelling it.
    close_timer = None

    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 3, 2])
    ax_command = fig.add_subplot(gs[0])
    ax_spec = fig.add_subplot(gs[1])
    ax_fft = fig.add_subplot(gs[2])
    fig.suptitle("Whistle input (close window or Ctrl+C to stop)")

    ax_command.axis("off")
    command_label = ax_command.text(
        0.5, 0.62, "STOP", transform=ax_command.transAxes,
        ha="center", va="center", fontsize=40, fontweight="bold", color=COMMAND_COLORS["stop"],
    )
    info_label = ax_command.text(
        0.5, 0.15, "", transform=ax_command.transAxes,
        ha="center", va="center", fontsize=11, family="monospace",
    )

    spec_image = ax_spec.imshow(
        spectrogram_image,
        aspect="auto",
        origin="lower",
        extent=[0, SPECTROGRAM_HISTORY_CHUNKS, display_freqs[0], display_freqs[-1]],
        cmap="viridis",
    )
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xticks([])

    (fft_line,) = ax_fft.plot(display_freqs, last_display_magnitude)
    ax_fft.set_xlim(display_freqs[0], display_freqs[-1])
    ax_fft.set_ylabel("Magnitude")
    ax_fft.set_xlabel("Frequency (Hz)")

    def update(_frame):
        nonlocal spectrogram_image, last_display_magnitude, close_timer

        freq = None
        raw_command = "stop"
        command = debouncer.confirmed_command
        speed = 0.0
        note = "-"
        rms = 0.0
        drained = False
        should_stop = False
        started = gate is None or gate.is_set()

        # Drain whatever chunks have arrived since the last redraw so the
        # display doesn't fall behind the mic.
        while True:
            try:
                chunk = streamer.chunks.get_nowait()
            except queue.Empty:
                break
            drained = True

            _, magnitude = magnitude_spectrum(chunk, SAMPLE_RATE)
            last_display_magnitude = magnitude[display_mask]
            spectrogram_image = push_spectrogram_column(spectrogram_image, last_display_magnitude)

            rms = signal_rms(chunk)
            freq = dominant_frequency(chunk)
            raw_command = frequency_to_command(freq)
            note = frequency_to_note(freq) if freq is not None else "-"

            command = debouncer.update(raw_command)
            speed = resolve_speed(command, ramper.update(command, chunk_duration_s))

            if started:
                if motor is not None:
                    command_to_motor(motor, command, speed)

                if on_chunk is not None and on_chunk(command, speed):
                    should_stop = True

            if freq is not None:
                low, high = session_freq_range
                session_freq_range[0] = freq if low is None else min(low, freq)
                session_freq_range[1] = freq if high is None else max(high, freq)

            if should_stop:
                break

        if not drained:
            return command_label, info_label, spec_image, fft_line

        spec_image.set_data(spectrogram_image)
        spec_image.set_clim(vmin=0, vmax=max(float(spectrogram_image.max()), 1e-6))
        fft_line.set_ydata(last_display_magnitude)
        ax_fft.set_ylim(0, max(float(last_display_magnitude.max()), 1e-6) * 1.1)

        if started:
            command_label.set_text(command.upper())
            command_label.set_color(COMMAND_COLORS.get(command, "black"))
        else:
            command_label.set_text("WAITING FOR START")
            command_label.set_color("tab:gray")

        freq_str = f"{freq:.0f} Hz" if freq is not None else "-"
        low, high = session_freq_range
        range_str = f"{low:.0f}-{high:.0f} Hz" if low is not None else "-"
        passes_gate = "yes" if rms >= SILENCE_AMPLITUDE_THRESHOLD else "no -- too quiet"
        info_label.set_text(
            f"freq: {freq_str}   note: {note}   raw: {raw_command}   speed: {speed:.0f}%\n"
            f"session whistle range seen: {range_str}   (tune FREQ_BANDS in config.py)\n"
            f"amplitude (rms): {rms:.4f}   threshold: {SILENCE_AMPLITUDE_THRESHOLD:.4f}   "
            f"registers as sound: {passes_gate}"
        )

        if should_stop:
            # Closing the figure synchronously from inside FuncAnimation's own
            # timer callback crashes matplotlib's internals (it nulls out
            # `event_source` for its own close-event handling, then its
            # caller immediately tries to use that same now-None
            # `event_source`). Deferring the close to a one-shot timer lets
            # it happen just after this callback returns instead. The timer
            # is stashed on the enclosing `close_timer` (not just a local)
            # so it survives after update() returns -- otherwise nothing
            # keeps a reference to it and it can be garbage collected before
            # it ever fires, silently cancelling the close.
            close_timer = fig.canvas.new_timer(interval=1)
            close_timer.single_shot = True
            close_timer.add_callback(plt.close, fig)
            close_timer.start()

        return command_label, info_label, spec_image, fft_line

    with streamer:
        ani = animation.FuncAnimation(
            fig, update, interval=PLOT_REFRESH_MS, blit=False, cache_frame_data=False
        )
        plt.tight_layout()
        try:
            plt.show()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
