import pytest

from pathlib import Path

from lxd_io import Dataset
from lxd_io.track import Track


@pytest.fixture
def valid_track() -> Track:
    return Dataset("test/data/valid-dataset-v1.5").get_recording(1).get_track(1)


def test_track_access_properties(valid_track: Track) -> None:
    assert valid_track.id == 1
    assert len(valid_track.meta_data_keys) > 0
    assert len(valid_track.data_keys) > 0


def test_track_get_meta_data(valid_track: Track) -> None:
    # Valid meta_data_keys
    for meta_data_key in valid_track.meta_data_keys:
        assert valid_track.get_meta_data(meta_data_key) is not None

    # Invalid meta_data_key
    with pytest.raises(KeyError):
        valid_track.get_meta_data("invalid_key")


def test_track_get_data(valid_track: Track) -> None:
    # Valid data_keys
    for data_key in valid_track.data_keys:
        assert valid_track.get_data(data_key) is not None

    # Invalid data_key
    with pytest.raises(KeyError):
        valid_track.get_data("invalid_key")


def test_track_get_data_at_frame(valid_track: Track) -> None:
    # Valid frames
    for frame in valid_track._frames:
        for data_key in valid_track.data_keys:
            assert valid_track.get_data_at_frame(data_key, frame) is not None

    # Invalid data_key
    with pytest.raises(KeyError):
        valid_track.get_data_at_frame("invalid_key", 0)

    # Invalid frame
    with pytest.raises(KeyError):
        valid_track.get_data_at_frame("xCenter", -1)


def test_track_get_data_between_frames(valid_track: Track) -> None:
    # Valid frames
    for frame in valid_track._frames:
        for data_key in valid_track.data_keys:
            assert (
                valid_track.get_data_between_frames(data_key, frame, frame) is not None
            )

    # Invalid data_key
    with pytest.raises(KeyError):
        valid_track.get_data_between_frames("invalid_key", 0, 0)

    # Invalid frame
    with pytest.raises(KeyError):
        valid_track.get_data_between_frames("xCenter", -1, -1)

    # Invalid frame
    with pytest.raises(KeyError):
        valid_track.get_data_between_frames("xCenter", 0, -1)

    # Invalid frame
    with pytest.raises(KeyError):
        valid_track.get_data_between_frames("xCenter", -1, 0)

    # Invalid frame
    with pytest.raises(KeyError):
        valid_track.get_data_between_frames("xCenter", 1, 0)


def test_track_get_local_trajectory(valid_track: Track) -> None:
    assert valid_track.get_local_trajectory() is not None


def test_track_get_utm_image_trajectory(valid_track: Track) -> None:
    assert valid_track.get_utm_image_trajectory(0, 0) is not None


def test_track_get_bbox_at_frame(valid_track: Track) -> None:
    # Valid frames
    for frame in valid_track._frames:
        assert valid_track.get_bbox_at_frame(frame) is not None
        assert valid_track.get_bbox_at_frame(frame, visualization_mode=True) is not None

    # Invalid frame
    with pytest.raises(KeyError):
        valid_track.get_bbox_at_frame(-1)
