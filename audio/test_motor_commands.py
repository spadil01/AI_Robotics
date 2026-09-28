from motor_commands import command_to_motor


class FakeMotor:
    """Records calls instead of talking to real hardware."""

    def __init__(self):
        self.calls = []

    def movement_move_tank(self, speed_left, speed_right, blocking=True):
        self.calls.append(("tank", speed_left, speed_right, blocking))

    def movement_stop(self, blocking=True):
        self.calls.append(("stop", blocking))


def test_forward_drives_both_wheels_equally_forward():
    motor = FakeMotor()
    command_to_motor(motor, "forward", 60)
    assert motor.calls == [("tank", 60, 60, False)]


def test_backward_drives_both_wheels_equally_backward():
    motor = FakeMotor()
    command_to_motor(motor, "backward", 60)
    assert motor.calls == [("tank", -60, -60, False)]


def test_left_pivots_wheels_opposite_directions():
    motor = FakeMotor()
    command_to_motor(motor, "left", 40)
    assert motor.calls == [("tank", -40, 40, False)]


def test_right_pivots_wheels_opposite_directions():
    motor = FakeMotor()
    command_to_motor(motor, "right", 40)
    assert motor.calls == [("tank", 40, -40, False)]


def test_stop_calls_movement_stop():
    motor = FakeMotor()
    command_to_motor(motor, "stop", 0)
    assert motor.calls == [("stop", False)]


def test_unrecognized_command_stops_as_fail_safe():
    motor = FakeMotor()
    command_to_motor(motor, "not_a_real_command", 50)
    assert motor.calls == [("stop", False)]


def test_speed_is_cast_to_int():
    motor = FakeMotor()
    command_to_motor(motor, "forward", 42.7)
    assert motor.calls == [("tank", 42, 42, False)]
