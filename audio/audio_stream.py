"""Non-blocking PyAudio input stream that queues mono float32 chunks."""

import queue

import numpy as np
import pyaudio

from config import CHANNELS, CHUNK, INPUT_DEVICE_INDEX, SAMPLE_RATE

FORMAT = pyaudio.paInt16


class AudioStreamer:
    """Opens a non-blocking PyAudio input stream and queues float32 chunks.

    Audio arrives on PyAudio's internal callback thread; chunks are handed
    off through a thread-safe queue so a consumer (a plotting loop, a motor
    control loop) can drain them at its own pace on the main thread.
    """

    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        chunk=CHUNK,
        input_device_index=INPUT_DEVICE_INDEX,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.input_device_index = input_device_index
        self.chunks = queue.Queue()
        self._pa = None
        self._stream = None

    def _on_audio(self, in_data, frame_count, time_info, status):
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        samples /= np.iinfo(np.int16).max
        self.chunks.put(samples)
        return (None, pyaudio.paContinue)

    def start(self):
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=FORMAT,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.chunk,
            stream_callback=self._on_audio,
        )
        self._stream.start_stream()

    def stop(self):
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa is not None:
            self._pa.terminate()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
