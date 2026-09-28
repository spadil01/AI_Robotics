"""Light sensor (LEGO Color Sensor) proximity detection for the "ball" role.

Uses the Color Sensor's `reflection` reading (0-100, rises as something
gets closer) as a proximity signal: once a reading crosses
LIGHT_SENSOR_THRESHOLD, `triggered` is set so the caller can stop the robot.

Run directly to calibrate LIGHT_SENSOR_THRESHOLD against the real
sensor/lighting: prints the live reflection reading so you can watch it
change as an object approaches.

    python audio/light_sensor.py

Ctrl+C to stop.
"""

import threading
import time

import legoeducation as le

from config import LIGHT_SENSOR_CARD_COLOR, LIGHT_SENSOR_CARD_SERIAL, LIGHT_SENSOR_THRESHOLD


def is_object_close(reflection, threshold=LIGHT_SENSOR_THRESHOLD):
    """Return True if `reflection` indicates something is close enough to trigger a stop."""
    return reflection >= threshold


class LightSensorMonitor:
    """Wraps a LEGO Color Sensor and sets `triggered` once something gets close.

    Connects on `start()`, which registers a notification callback that
    parses incoming hardware updates and checks `is_object_close()` against
    each reflection reading.
    """

    def __init__(
        self,
        threshold=LIGHT_SENSOR_THRESHOLD,
        card_color=LIGHT_SENSOR_CARD_COLOR,
        card_serial=LIGHT_SENSOR_CARD_SERIAL,
    ):
        self.threshold = threshold
        self.card_color = card_color
        self.card_serial = card_serial
        self.sensor = le.ColorSensor()
        self.triggered = threading.Event()
        self.last_reflection = None

    def _on_notification(self, data):
        for item in le.device_notification_parser(data):
            if isinstance(item, le.ColorSensorNotification):
                self.last_reflection = item.reflection
                if is_object_close(item.reflection, self.threshold) and not self.triggered.is_set():
                    self.triggered.set()
                    print(
                        f"Light sensor triggered! (reflection={item.reflection} "
                        f">= threshold={self.threshold})"
                    )

    def start(self):
        self.sensor.set_notification_callback(self._on_notification)
        self.sensor.connect(card_color=self.card_color, card_serial=self.card_serial)
        if not self.sensor.connected:
            raise RuntimeError("Error connecting to Color Sensor (light sensor).")

    def stop(self):
        if self.sensor.connected:
            # disconnect() blocks until the BLE stack confirms teardown,
            # which can take a few seconds -- print so this isn't mistaken
            # for a hang.
            print("Disconnecting from light sensor (may take a few seconds)...")
            self.sensor.disconnect()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


def main():
    print("Connecting to Color Sensor (light sensor)...")
    monitor = LightSensorMonitor()
    try:
        with monitor:
            print("Connected. Move an object toward/away from the sensor; Ctrl+C to stop.")
            last_printed = None
            while True:
                reading = monitor.last_reflection
                if reading != last_printed:
                    triggered = is_object_close(reading, monitor.threshold) if reading is not None else False
                    print(f"reflection: {reading}   threshold: {monitor.threshold}   close: {triggered}")
                    last_printed = reading
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
