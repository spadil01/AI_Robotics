import numpy as np

from live_plot import push_spectrogram_column


def test_push_spectrogram_column_appends_at_the_end():
    image = np.zeros((3, 4), dtype=np.float32)
    column = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = push_spectrogram_column(image, column)
    assert list(result[:, -1]) == [1.0, 2.0, 3.0]


def test_push_spectrogram_column_shifts_old_columns_left():
    image = np.zeros((2, 3), dtype=np.float32)
    image[:, -1] = [9.0, 9.0]
    column = np.array([1.0, 1.0], dtype=np.float32)
    result = push_spectrogram_column(image, column)
    # The column that used to be last (index -1) should now be second-to-last.
    assert list(result[:, -2]) == [9.0, 9.0]


def test_push_spectrogram_column_preserves_shape():
    image = np.zeros((5, 8), dtype=np.float32)
    column = np.ones(5, dtype=np.float32)
    result = push_spectrogram_column(image, column)
    assert result.shape == image.shape
