"""Whistle-controlled LEGO Double Motor -- full robot script.

Prompts for the robot's role, connects the Double Motor, then opens
live_plot.py's live waveform/spectrogram/FFT/command display right away --
already running, but gated in a "waiting for start" stage until "start"
arrives on START_TOPIC (config.py), all without blocking: MQTT runs on its
own background thread, so the plot stays live and responsive the whole
time. Once started, whistled commands drive the motor as before, until one
of the role's stop triggers fires and a song plays:

  "ball" role has two distinct triggers, each with its own song and MQTT
  publish, since they mean different things:
    - light sensor detects something close (light_sensor.py) -- the ball
      got stopped by the goalie -- publishes "Ball stopped" to BALL_TOPIC
      and opens BALL_STOPPED_SONG_URL.
    - a very high-pitched whistle is confirmed as the "goal" command
      (FREQ_BANDS in config.py) -- the ball went in the net -- publishes
      "Goal!" to BALL_TOPIC and opens BALL_GOAL_SONG_URL.

  "goalie" role waits on GOALIE_TOPIC for GOALIE_TRIGGER_MESSAGE -- the
  goalie made the stop -- and opens GOALIE_SONG_URL.

Either way: stop the motor, stop listening, end. Run directly:

    python audio/whistle_control_robot.py

Close the plot window or Ctrl+C to stop early without a trigger -- the
motor is stopped and disconnected either way, but no song plays and
nothing is published, since that wasn't a real game event.
"""

import os
import sys
import threading
import webbrowser

import legoeducation as le

# mqttlib.py lives at the repo root, one level up from audio/ -- add it to
# sys.path rather than duplicating or relocating it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mqttlib import MQTTClient  # noqa: E402

import live_plot  # noqa: E402
from config import (  # noqa: E402
    BALL_GOAL_SONG_URL,
    BALL_STOPPED_SONG_URL,
    BALL_TOPIC,
    GOALIE_SONG_URL,
    GOALIE_TOPIC,
    GOALIE_TRIGGER_MESSAGE,
    MOTOR_CARD_COLOR,
    MOTOR_CARD_SERIAL,
    START_TOPIC,
)
from light_sensor import LightSensorMonitor  # noqa: E402


def _message_matches(payload, expected):
    """Case-insensitive, whitespace-trimmed comparison for MQTT message payloads."""
    return payload.strip().lower() == expected.strip().lower()


def prompt_role():
    """Ask on the console whether this robot is the "ball" or the "goalie"."""
    while True:
        answer = input('Is this robot the "ball" or the "goalie"? ').strip().lower()
        if answer in ("ball", "goalie"):
            return answer
        print('Please type "ball" or "goalie".')


def subscribe_for_start(mqtt_client):
    """Subscribe to START_TOPIC; return a threading.Event set once "start" arrives.

    Doesn't block -- the subscription and the eventual message both happen
    on mqtt_client's own background thread, so the caller can go on to open
    the live plot immediately and let it show a "waiting" stage instead.
    """
    started = threading.Event()

    def on_message(_topic, payload):
        if _message_matches(payload, "start"):
            started.set()

    mqtt_client.subscribe(START_TOPIC, on_message)
    print(f'Waiting for "start" on {START_TOPIC}...')
    return started


def run_ball(motor, mqtt_client, started):
    """Drive the motor from whistled commands (once started) until the light
    sensor (ball stopped) or a "goal" whistle triggers a stop, then publish
    and play the matching song.
    """
    trigger = None

    with LightSensorMonitor() as light_monitor:

        def on_chunk(command, _speed):
            nonlocal trigger
            if light_monitor.triggered.is_set():
                trigger = "light_sensor"
                return True
            if command == "goal":
                trigger = "whistle"
                return True
            return False

        print(
            "Whistle to drive the robot once started. A very high whistle "
            "(goal!), or something getting close to the light sensor "
            "(stopped), ends the run."
        )
        live_plot.main(motor=motor, on_chunk=on_chunk, gate=started)

        # Publish/play *before* the light sensor disconnects (below, once
        # this `with` block exits) -- BLE disconnect is a synchronous call
        # that can take a few seconds, and that delay shouldn't sit in front
        # of the feedback that actually matters (the song, the MQTT message).
        if trigger == "whistle":
            print("Goal whistle triggered -- publishing and playing goal song.")
            mqtt_client.publish(BALL_TOPIC, "Goal!")
            webbrowser.open(BALL_GOAL_SONG_URL)
        elif trigger == "light_sensor":
            print("Light sensor triggered -- publishing and playing stopped song.")
            mqtt_client.publish(BALL_TOPIC, "Ball stopped")
            webbrowser.open(BALL_STOPPED_SONG_URL)
        else:
            print("Ended without a trigger (window closed or Ctrl+C) -- no song played.")


def run_goalie(motor, mqtt_client, started):
    """Drive the motor from whistled commands (once started) until
    GOALIE_TRIGGER_MESSAGE arrives on GOALIE_TOPIC, then play the song.
    """
    triggered = threading.Event()

    def on_message(_topic, payload):
        if _message_matches(payload, GOALIE_TRIGGER_MESSAGE):
            triggered.set()

    mqtt_client.subscribe(GOALIE_TOPIC, on_message)

    def on_chunk(_command, _speed):
        return triggered.is_set()

    print(
        f'Whistle to drive the robot once started. Waiting for '
        f'"{GOALIE_TRIGGER_MESSAGE}" on {GOALIE_TOPIC} to end the run.'
    )
    live_plot.main(motor=motor, on_chunk=on_chunk, gate=started)

    if triggered.is_set():
        print("Goalie trigger received -- playing song.")
        webbrowser.open(GOALIE_SONG_URL)
    else:
        print("Ended without a trigger (window closed or Ctrl+C) -- no song played.")


def main():
    role = prompt_role()

    motor = le.DoubleMotor()

    try:
        print("Connecting to Double Motor...")
        motor.connect(card_color=MOTOR_CARD_COLOR, card_serial=MOTOR_CARD_SERIAL)
        if not motor.connected:
            print("Error connecting to Double Motor.")
            return

        with MQTTClient() as mqtt_client:
            started = subscribe_for_start(mqtt_client)

            if role == "ball":
                run_ball(motor, mqtt_client, started)
            else:
                run_goalie(motor, mqtt_client, started)

    finally:
        print("Cleaning up...")
        if motor.connected:
            try:
                motor.movement_stop()
            except Exception as exc:
                print(f"Error stopping motor: {exc}")
            # disconnect() blocks until the BLE stack confirms teardown,
            # which can take a few seconds -- print so this isn't mistaken
            # for a hang (see the light sensor's disconnect for the same).
            print("Disconnecting from motor (may take a few seconds)...")
            motor.disconnect()


if __name__ == "__main__":
    main()
