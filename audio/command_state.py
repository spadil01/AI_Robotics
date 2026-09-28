"""Turn noisy per-chunk command guesses into a stable command and a ramped speed.

No I/O here either -- both classes are driven purely by the values the
caller feeds them (a raw command string, and elapsed time), so they're fully
unit-testable without a mic, MQTT, or a motor.
"""

from config import (
    CONSISTENT_FRAMES_TO_CONFIRM,
    MAX_THROTTLE_SPEED,
    MIN_THROTTLE_SPEED,
    RAMP_DURATION_S,
    TURN_SPEED,
)

# Commands that turn in place rather than drive forward/backward -- these
# use a fixed speed instead of SpeedRamper's sustain-based ramp, so turning
# stays precise regardless of how long it's held.
TURN_COMMANDS = ("left", "right")


class CommandDebouncer:
    """Stabilizes noisy per-chunk command guesses into a confirmed command.

    Requires `required_consistent_frames` consecutive matching raw guesses
    before the confirmed command changes, so a single noisy chunk (a brief
    amplitude dip, a pitch wobble into an inter-band gap) doesn't flip the
    robot's behavior. Any raw guess that doesn't match the frame currently
    being counted resets the count, so only a *sustained* change wins.
    """

    def __init__(self, required_consistent_frames=CONSISTENT_FRAMES_TO_CONFIRM, initial_command="stop"):
        self.required_consistent_frames = required_consistent_frames
        self.confirmed_command = initial_command
        self._candidate_command = initial_command
        self._candidate_count = 0

    def update(self, raw_command):
        """Feed one raw per-chunk guess; return the (possibly unchanged) confirmed command."""
        if raw_command == self._candidate_command:
            self._candidate_count += 1
        else:
            self._candidate_command = raw_command
            self._candidate_count = 1

        if self._candidate_count >= self.required_consistent_frames:
            self.confirmed_command = self._candidate_command

        return self.confirmed_command

    def reset(self, command="stop"):
        """Force the confirmed command back to `command`, discarding any in-progress candidate."""
        self.confirmed_command = command
        self._candidate_command = command
        self._candidate_count = 0


class SpeedRamper:
    """Maps how long a command has been continuously confirmed into a 0-100 speed.

    Speed ramps linearly from `min_speed` up to `max_speed` over
    `ramp_duration_s` seconds of the same command being sustained, and
    resets back to `min_speed` whenever the command changes (including
    transitions to/from "stop"). "stop" always reports speed 0 regardless
    of how long it's been sustained.
    """

    def __init__(self, min_speed=MIN_THROTTLE_SPEED, max_speed=MAX_THROTTLE_SPEED, ramp_duration_s=RAMP_DURATION_S):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.ramp_duration_s = ramp_duration_s
        self._current_command = "stop"
        self._sustained_s = 0.0

    def update(self, command, dt):
        """Advance by `dt` seconds under `command`; return the current speed."""
        if command != self._current_command:
            self._current_command = command
            self._sustained_s = 0.0
        else:
            self._sustained_s += dt

        if command == "stop":
            return 0.0

        if self.ramp_duration_s <= 0:
            return self.max_speed

        fraction = min(1.0, self._sustained_s / self.ramp_duration_s)
        return self.min_speed + fraction * (self.max_speed - self.min_speed)

    def reset(self):
        self._current_command = "stop"
        self._sustained_s = 0.0


def resolve_speed(command, ramped_speed, turn_speed=TURN_SPEED):
    """Return the speed to actually drive at for `command`.

    "left"/"right" always use the fixed `turn_speed` for precise turning,
    regardless of how long they've been sustained; everything else
    (including "stop", already 0) uses `ramped_speed` from SpeedRamper.
    """
    if command in TURN_COMMANDS:
        return turn_speed
    return ramped_speed
