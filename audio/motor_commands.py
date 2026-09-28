"""Maps a confirmed whistle command and ramped speed to LEGO DoubleMotor calls.

Takes `motor` as a parameter (duck-typed) instead of connecting to hardware
itself, so this mapping logic is unit-testable with a fake motor -- no BLE
connection or physical DoubleMotor required.
"""


def command_to_motor(motor, command, speed):
    """Drive `motor` according to `command` at `speed` (0-100).

    "forward"/"backward" drive both wheels the same direction; "left"/
    "right" pivot in place by driving the wheels opposite directions.
    Anything else (including "stop", and any unrecognized command) stops
    the motor as a fail-safe default.
    """
    speed = int(speed)
    if command == "forward":
        motor.movement_move_tank(speed_left=speed, speed_right=speed, blocking=False)
    elif command == "backward":
        motor.movement_move_tank(speed_left=-speed, speed_right=-speed, blocking=False)
    elif command == "left":
        motor.movement_move_tank(speed_left=-speed, speed_right=speed, blocking=False)
    elif command == "right":
        motor.movement_move_tank(speed_left=speed, speed_right=-speed, blocking=False)
    else:
        motor.movement_stop(blocking=False)
