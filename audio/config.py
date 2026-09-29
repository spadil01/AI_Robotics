"""Global tunables for the whistle-controlled robot.

Everything here is meant to be tuned in one place -- especially anything
that depends on room acoustics, the whistler, or the mic -- instead of being
hunted down inside the modules that use it. More constants get added here as
later pieces of the project (MQTT topics, motor speeds, light sensor
threshold, songs) are built.
"""

import legoeducation as le

# Which role this robot is playing ("ball" or "goalie") is asked
# interactively when whistle_control_robot.py starts, rather than set here.

# MQTT topics -- change these as assignments change. The robot only starts
# driving/listening for stop triggers after "start" arrives on START_TOPIC.
# The ball publishes to BALL_TOPIC when it stops (light sensor) or scores
# (goal whistle); the goalie waits on GOALIE_TOPIC for GOALIE_TRIGGER_MESSAGE.
START_TOPIC = "ME193/Rogers/"
BALL_TOPIC = "ME193/Rogers/ballstatus"
GOALIE_TOPIC = "ME193/Rogers/goalstatus"

# The exact message payload (case-insensitive, whitespace-trimmed) the
# goalie waits for on GOALIE_TOPIC before stopping and playing its song.
# Placeholder -- confirm this matches whatever message is actually sent.
GOALIE_TRIGGER_MESSAGE = "stop"

# What the "ball" role publishes to BALL_TOPIC when it scores (goal
# whistle) or gets stopped (light sensor).
BALL_GOAL_MESSAGE = "Goal"
BALL_STOPPED_MESSAGE = "Ball stopped"

# The goalie also listens on GOALIE_LOSS_TOPIC for GOALIE_LOSS_MESSAGE --
# the ball scored -- and plays GOALIE_LOSS_SONG_URL instead. Defaults match
# what this code's own "ball" role publishes on a goal; change both to
# whatever's agreed with the other team.
GOALIE_LOSS_TOPIC = BALL_TOPIC
GOALIE_LOSS_MESSAGE = BALL_GOAL_MESSAGE

# The Connection Card attached to the Double Motor being driven. Update
# these to match the card actually on the robot -- see hand_control_robot.py
# for how these are used with legoeducation's connect().
MOTOR_CARD_COLOR = le.LEGO_COLOR_RED
MOTOR_CARD_SERIAL = "0994"

# The Connection Card attached to the Color Sensor used as the "ball"
# role's light/proximity sensor. Update these to match the card actually on
# the robot -- placeholder serial below, this hasn't been connected yet.
LIGHT_SENSOR_CARD_COLOR = le.LEGO_COLOR_RED
LIGHT_SENSOR_CARD_SERIAL = "0994"

# How long (seconds) to scan over Bluetooth for the motor / light sensor
# before giving up. Without a limit, legoeducation's connect() scans
# forever if the hardware is off, out of range, or the card color/serial
# above don't match -- which looks like the program is stuck loading.
BLE_SCAN_TIMEOUT_S = 15

# Reflected light intensity (0-100) at or above which something is
# considered close enough to trigger a stop -- reflection rises as an
# object gets closer to the sensor. This is a placeholder; run
# light_sensor.py directly and watch the printed reflection value while
# moving an object toward/away from the actual sensor to pick a real
# threshold for this sensor/lighting/robot.
LIGHT_SENSOR_THRESHOLD = 50

# Which PyAudio input device to record from. None uses the system's default
# input device (usually the built-in mic). To use something else -- a USB
# mic, an audio interface, a virtual/loopback device -- run
# list_audio_devices.py to print each available device's index and name,
# then set this to that device's index.
INPUT_DEVICE_INDEX = 0

SAMPLE_RATE = 44100

# A chunk's RMS amplitude below this is treated as silence (no whistle),
# which maps to the "stop" command as a fail-safe default.
#
# Measured on this mic/room: quiet-room noise floor peaks around 0.003 RMS;
# a deliberately loud whistle at the lowest usable pitch stays above 0.006
# RMS. Set roughly midway between those so there's margin against both
# false triggers from room noise and false silence from a slightly softer
# whistle. This assumes the whistler consistently blows hard at the low end
# of the range -- re-measure with live_plot.py's amplitude readout if that
# stops being reliable.
SILENCE_AMPLITUDE_THRESHOLD = 0.0045

# Frequencies outside this range are ignored even if loud, to reject low
# rumble/room noise and high-frequency hiss that isn't a whistle.
MIN_WHISTLE_HZ = 400.0
MAX_WHISTLE_HZ = 2200.0

# Ordered (low_hz, high_hz, command) bands. A detected frequency in
# [low_hz, high_hz) maps to `command`; anything unmatched -- including
# silence and the small gaps deliberately left between bands -- falls back
# to "stop", so boundary wobble between two commands lands on "stop" rather
# than flickering between them.
#
# Fitted to a measured whistle range of 764-1270 Hz (padded to 750-1280 Hz),
# shifted down 50 Hz, then extended another 50 Hz at the bottom to a
# 650-1230 Hz working range. "left" was widened up to 1100 Hz (was eating
# into "right"'s lower range), pushing "right" to start after it with the
# same ~15 Hz gap while keeping the 1230 Hz upper limit. Re-run
# live_plot.py's "session whistle range seen" readout and adjust these to
# match if the whistler or mic changes.
#
# The lowest and second-lowest bands were swapped to "forward"/"backward"
# (rather than retuning the pitch ranges themselves) since the lowest one
# was easier to hit reliably and that's now the more-used "forward".
#
# "goal" is not a movement command -- it's a deliberately very high, separate
# whistle that (for the "ball" role) manually triggers the same
# stop-and-play-song flow as the light sensor, to celebrate scoring a goal
# without waiting for the sensor. Originally 1600-2000 Hz, lowered to start
# at 1300 Hz since 2000 Hz wasn't reachable, then widened back up to 2000 Hz
# at the top (so 1300-2000 Hz overall) to make it easier to land reliably --
# still a 70 Hz gap above "right" (1230 Hz) so it can't be whistled by
# accident. Re-verify with live_plot.py that this is comfortably reachable.
FREQ_BANDS = [
    (650.0, 784.0, "forward"),
    (799.0, 933.0, "backward"),
    (948.0, 1100.0, "left"),
    (1115.0, 1230.0, "right"),
    (1300.0, 2000.0, "goal"),
]

# Samples per audio chunk read from the mic. Smaller = lower latency but
# coarser frequency resolution (resolution is SAMPLE_RATE / CHUNK Hz). Each
# chunk also costs one full CONSISTENT_FRAMES_TO_CONFIRM cycle of debounce
# latency (chunk_duration * CONSISTENT_FRAMES_TO_CONFIRM before a command
# can be confirmed at all), so this is a direct trade-off against
# responsiveness. 4096 was needed while FREQ_BANDS were only ~50 Hz wide;
# now that they're ~134 Hz wide, 2048 (~22 Hz resolution, still ~6
# resolvable bins per band) roughly halves that latency. Drop to 1024
# (~43 Hz resolution, ~3 bins/band) for an even snappier response if this
# still feels slow and jitter doesn't return.
CHUNK = 2048
CHANNELS = 1

# How much history the live spectrogram plot keeps on screen, in number of
# audio chunks.
SPECTROGRAM_HISTORY_CHUNKS = 200

# Highest frequency shown on the spectrogram/FFT plots. Set a bit above
# MAX_WHISTLE_HZ so out-of-band noise is still visible on screen.
DISPLAY_MAX_FREQ_HZ = 3000.0

# Matplotlib animation refresh interval, in milliseconds.
PLOT_REFRESH_MS = 30

# How many consecutive chunks must agree on a raw command guess before it
# becomes the confirmed command. Higher = less jitter from momentary
# amplitude/pitch wobble (see the "jitters between a command and stop"
# tuning discussion), but slower to react to a genuine change.
CONSISTENT_FRAMES_TO_CONFIRM = 4

# Only "forward"/"backward" ramp with sustain time -- speed (0-100) ramps
# from MIN_THROTTLE_SPEED up to MAX_THROTTLE_SPEED the longer a confirmed
# command is sustained, reaching MAX_THROTTLE_SPEED after RAMP_DURATION_S
# seconds of continuously whistling the same command. Resets back to
# MIN_THROTTLE_SPEED whenever the confirmed command changes. "stop" always
# means speed 0. Capped below 100 so top speed stays controllable.
MIN_THROTTLE_SPEED = 25.0
MAX_THROTTLE_SPEED = 70.0
RAMP_DURATION_S = 5.0

# "left"/"right" (turning) always drive at this fixed speed instead of
# ramping, so turning stays precise regardless of how long it's sustained.
TURN_SPEED = 30.0

# Links opened (via webbrowser.open(), in the default browser) for
# each of the three song-triggering scenarios. Requires internet access at
# the venue. Point these at whichever clips are actually chosen -- consider
# a link with autoplay enabled (e.g. a YouTube URL with "?autoplay=1"
# appended) so it starts without needing a manual click.
#
# Played when the ball's light sensor is triggered -- the ball got stopped
# by the goalie.
BALL_STOPPED_SONG_URL = "https://www.youtube.com/watch?v=m9zhgDsd4P4&autoplay=1"
# Played when the ball's high "goal" whistle is confirmed -- the ball went
# in the net.
BALL_GOAL_SONG_URL = "https://www.youtube.com/watch?v=TGtWWb9emYI&list=RDTGtWWb9emYI&start_radio=1&autoplay=1"
# Played when the goalie's MQTT trigger fires -- the goalie made the stop.
GOALIE_SONG_URL = "https://youtu.be/w5tWYmIOWGk?list=RDw5tWYmIOWGk&t=45&autoplay=1"
# Played when GOALIE_LOSS_MESSAGE arrives -- the ball scored on the goalie.
# Reuses the ball's stopped (death) song until a different one is picked.
GOALIE_LOSS_SONG_URL = BALL_STOPPED_SONG_URL