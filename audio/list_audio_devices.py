"""Print every audio input device PyAudio can see, with its index.

Run this to find the index to put in config.py's INPUT_DEVICE_INDEX when you
want to record from something other than the system default microphone (a
USB mic, an audio interface, a virtual/loopback device, etc.):

    python audio/list_audio_devices.py
"""

import pyaudio


def list_input_devices():
    pa = pyaudio.PyAudio()
    try:
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if info["maxInputChannels"] > 0:
                print(f"{index}: {info['name']} (max {info['maxInputChannels']} channels)")
    finally:
        pa.terminate()


if __name__ == "__main__":
    list_input_devices()
