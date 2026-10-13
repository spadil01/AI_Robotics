"""
Small wrapper around paho-mqtt for publishing and subscribing to an MQTT
broker (defaults to the public test.mosquitto.org broker).
"""

import threading

import paho.mqtt.client as mqtt


class MQTTClient:
    def __init__(self, broker="test.mosquitto.org", port=1883, client_id="", will=None):
        """`will` = (topic, payload, qos, retain): a message the broker sends for us
        if this program disappears without disconnecting (crash, killed, etc.)."""
        self.broker = broker
        self.port = port
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        if will:
            self._client.will_set(*will)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._callbacks = {}  # topic -> callback(topic, payload)
        self._connected = threading.Event()

    def connect(self, timeout=5, keepalive=60):
        """keepalive (s): how often to check in with the broker; a missing program is
        noticed (and its `will` sent) after about 1.5x this long."""
        self._client.connect(self.broker, self.port, keepalive=keepalive)
        self._client.loop_start()  # handle network traffic on a background thread
        self._connected.wait(timeout)  # block until the broker confirms the connection

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()
        self._connected.clear()

    def publish(self, topic, message, qos=0, retain=False):
        self._client.publish(topic, message, qos=qos, retain=retain)

    def subscribe(self, topic, callback, qos=0, raw=False):
        """Call `callback(topic, payload)` whenever a message arrives on `topic`.
        The payload is decoded text, or the original bytes if `raw` is True
        (for binary data such as images)."""
        self._callbacks[topic] = (callback, raw)
        self._client.subscribe(topic, qos=qos)

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self._connected.set()

    def _on_message(self, client, userdata, msg):
        entry = self._callbacks.get(msg.topic)
        if entry:
            callback, raw = entry
            callback(msg.topic, msg.payload if raw else msg.payload.decode())

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
