from light_sensor import is_object_close


def test_is_object_close_true_when_at_threshold():
    assert is_object_close(50, threshold=50) is True


def test_is_object_close_true_when_above_threshold():
    assert is_object_close(80, threshold=50) is True


def test_is_object_close_false_when_below_threshold():
    assert is_object_close(49, threshold=50) is False


def test_is_object_close_false_for_zero_reflection():
    assert is_object_close(0, threshold=50) is False


def test_is_object_close_uses_config_threshold_by_default():
    from config import LIGHT_SENSOR_THRESHOLD

    assert is_object_close(LIGHT_SENSOR_THRESHOLD) is True
    assert is_object_close(0) is False
