"""Settings for the laptop half of the sumo robot (sumo_control.py)."""

import legoeducation as le

# All three LEGO parts (Double Motor, Color Sensor, Controller) are paired
# with the same Connection Card.
CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0997"

# How long to look for each LEGO part over Bluetooth before giving up.
BLE_SCAN_TIMEOUT_S = 15

# The MQTT broker (mosquitto) runs on this laptop; sumo_control.py starts it
# if it isn't already running. The UNO Q connects to this laptop's IP.
BROKER = "localhost"
PORT = 1883
LEGO_TOPIC = "sumo/lego"      # laptop -> UNO Q: light sensor colour + joysticks
DRIVE_TOPIC = "sumo/drive"    # UNO Q -> laptop: wheel speeds to apply
STATUS_TOPIC = "sumo/status"  # UNO Q -> laptop: what the robot is doing

SEND_HZ = 30                  # LEGO readings go out on every change, and at least this often
SENSOR_NOTIFY_MS = 15         # how often the LEGO parts report their readings (15 = fastest)
DRIVE_STALE_S = 0.5           # no wheel command from the UNO Q this long -> stop
PART_STALE_S = 0.3            # a LEGO part silent this long counts as lost

# Blue-tape reflex: brake the instant the light sensor sees the edge, until the
# UNO Q's escape command arrives (or REFLEX_MAX_S passes). The edge rule must
# match EDGE_* in ArduinoApps/sumo-robot/python/behavior_tree.py.
EDGE_COLORS = {3}             # LEGO colour 3 = Blue
EDGE_HUE = (170, 250)         # blue-ish hues ...
EDGE_MIN_SAT = 8              # ... that aren't basically grey/white
REFLEX_MAX_S = 0.3

# Ramp (Single Motor). Its position when sumo_control.py starts counts as
# "down"; "up" turns it RAMP_UP_DEG from there (negative = the other way).
# Tune with: python sumo_control/sumo_control.py --ramp
RAMP_UP_DEG = 100
RAMP_SPEED = 50               # % speed when moving the ramp

# Played when the UNO Q says the robot went out of the ring. None = the
# built-in "sad trombone"; or a path to your own .mp3/.wav (macOS).
LOSE_SONG_FILE = None

# Flip these if a wheel spins the wrong way (robot should go forward when
# both sticks are pushed up).
INVERT_LEFT = False
INVERT_RIGHT = False
