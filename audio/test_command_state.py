import pytest

from command_state import CommandDebouncer, SpeedRamper, resolve_speed


# ---- CommandDebouncer ----


def test_debouncer_starts_at_stop_by_default():
    debouncer = CommandDebouncer(required_consistent_frames=4)
    assert debouncer.confirmed_command == "stop"


def test_debouncer_holds_old_command_until_required_frames_reached():
    debouncer = CommandDebouncer(required_consistent_frames=4, initial_command="stop")
    assert debouncer.update("forward") == "stop"
    assert debouncer.update("forward") == "stop"
    assert debouncer.update("forward") == "stop"
    # 4th consecutive matching frame is the one that confirms the switch.
    assert debouncer.update("forward") == "forward"


def test_debouncer_ignores_a_single_noisy_frame():
    debouncer = CommandDebouncer(required_consistent_frames=4, initial_command="forward")
    for _ in range(10):
        assert debouncer.update("forward") == "forward"
    # A single stray "stop" (e.g. a momentary amplitude dip) shouldn't flip it.
    assert debouncer.update("stop") == "forward"
    assert debouncer.update("forward") == "forward"


def test_debouncer_restarts_count_when_candidate_changes_mid_streak():
    debouncer = CommandDebouncer(required_consistent_frames=4, initial_command="stop")
    debouncer.update("forward")
    debouncer.update("forward")
    debouncer.update("left")  # different candidate: restarts the streak
    debouncer.update("forward")
    debouncer.update("forward")
    debouncer.update("forward")
    # Only 3 consecutive "forward" frames since the reset -- not enough yet.
    assert debouncer.confirmed_command == "stop"
    assert debouncer.update("forward") == "forward"


def test_debouncer_with_one_required_frame_confirms_immediately():
    debouncer = CommandDebouncer(required_consistent_frames=1, initial_command="stop")
    assert debouncer.update("right") == "right"


def test_debouncer_reset_clears_in_progress_candidate():
    debouncer = CommandDebouncer(required_consistent_frames=4, initial_command="stop")
    debouncer.update("forward")
    debouncer.update("forward")
    debouncer.reset()
    assert debouncer.confirmed_command == "stop"
    debouncer.update("forward")
    debouncer.update("forward")
    debouncer.update("forward")
    assert debouncer.confirmed_command == "stop"  # streak was reset, needs 4 fresh frames
    assert debouncer.update("forward") == "forward"


# ---- SpeedRamper ----


def test_speed_ramper_new_command_starts_at_min_speed():
    ramper = SpeedRamper(min_speed=25.0, max_speed=100.0, ramp_duration_s=4.0)
    assert ramper.update("forward", dt=0.0) == pytest.approx(25.0)


def test_speed_ramper_reaches_max_speed_after_ramp_duration():
    ramper = SpeedRamper(min_speed=25.0, max_speed=100.0, ramp_duration_s=4.0)
    ramper.update("forward", dt=0.0)
    ramper.update("forward", dt=2.0)
    speed = ramper.update("forward", dt=2.0)
    assert speed == pytest.approx(100.0)


def test_speed_ramper_midway_through_ramp_is_between_min_and_max():
    ramper = SpeedRamper(min_speed=20.0, max_speed=100.0, ramp_duration_s=4.0)
    ramper.update("forward", dt=0.0)
    speed = ramper.update("forward", dt=2.0)  # halfway through the ramp
    assert speed == pytest.approx(60.0)  # 20 + 0.5 * (100 - 20)


def test_speed_ramper_clamps_at_max_speed_past_ramp_duration():
    ramper = SpeedRamper(min_speed=25.0, max_speed=100.0, ramp_duration_s=4.0)
    ramper.update("forward", dt=0.0)
    speed = ramper.update("forward", dt=100.0)  # way past the ramp duration
    assert speed == pytest.approx(100.0)


def test_speed_ramper_resets_when_command_changes():
    ramper = SpeedRamper(min_speed=25.0, max_speed=100.0, ramp_duration_s=4.0)
    ramper.update("forward", dt=0.0)
    ramper.update("forward", dt=4.0)  # fully ramped up
    speed = ramper.update("left", dt=0.0)  # switched commands
    assert speed == pytest.approx(25.0)


def test_speed_ramper_stop_is_always_zero_regardless_of_sustain():
    ramper = SpeedRamper(min_speed=25.0, max_speed=100.0, ramp_duration_s=4.0)
    ramper.update("stop", dt=0.0)
    speed = ramper.update("stop", dt=10.0)
    assert speed == 0.0


def test_speed_ramper_reset_returns_to_stop_state():
    ramper = SpeedRamper(min_speed=25.0, max_speed=100.0, ramp_duration_s=4.0)
    ramper.update("forward", dt=4.0)
    ramper.reset()
    assert ramper.update("forward", dt=0.0) == pytest.approx(25.0)


# ---- resolve_speed ----


def test_resolve_speed_uses_fixed_turn_speed_for_left():
    assert resolve_speed("left", ramped_speed=99.0, turn_speed=40.0) == 40.0


def test_resolve_speed_uses_fixed_turn_speed_for_right():
    assert resolve_speed("right", ramped_speed=10.0, turn_speed=40.0) == 40.0


def test_resolve_speed_ignores_turn_speed_for_forward_and_backward():
    assert resolve_speed("forward", ramped_speed=55.0, turn_speed=40.0) == 55.0
    assert resolve_speed("backward", ramped_speed=55.0, turn_speed=40.0) == 55.0


def test_resolve_speed_passes_through_zero_for_stop():
    assert resolve_speed("stop", ramped_speed=0.0, turn_speed=40.0) == 0.0
