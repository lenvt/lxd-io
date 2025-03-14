import numpy as np
import pytest

from pathlib import Path

from lxd_io.recording import Recording


@pytest.fixture
def valid_recording() -> Recording:
    return Recording(
        1,
        "test/data/valid-dataset-v1.5/data/01_recordingMeta.csv",
        "test/data/valid-dataset-v1.5/data/01_tracksMeta.csv",
        "test/data/valid-dataset-v1.5/data/01_tracks.csv",
        "test/data/valid-dataset-v1.5/data/01_background.png",
        4.0,
        "valid",
    )


@pytest.fixture
def valid_highd_recording() -> Recording:
    return Recording(
        1,
        "test/data/highD-dataset-v1.0/data/01_recordingMeta.csv",
        "test/data/highD-dataset-v1.0/data/01_tracksMeta.csv",
        "test/data/highD-dataset-v1.0/data/01_tracks.csv",
        "test/data/highD-dataset-v1.0/data/01_highway.png",
        4.0,
        "highd",
    )


@pytest.fixture
def lanelet2_path() -> Path:
    return Path("test/data/valid-dataset-v1.5/maps/lanelet2/1_location.osm")


@pytest.fixture
def opendrive_path() -> Path:
    return Path("test/data/valid-dataset-v1.5/maps/opendrive/1/1_location.xodr")


def test_recording_invalid_recording_meta_csv() -> None:
    with pytest.raises(KeyError):
        Recording(
            1,
            "test/data/invalid-dataset/data/1_recordingMeta.csv",
            "test/data/invalid-dataset/data/1_tracksMeta.csv",
            "test/data/invalid-dataset/data/1_tracks.csv",
            "test/data/invalid-dataset/data/1_background.png",
            0.0,
            "invalid",
        )


def test_recording_access_properties(
    valid_recording: Recording, valid_highd_recording: Recording
) -> None:
    assert valid_recording.id == 1
    assert valid_recording.datset_name == "valid"
    assert len(valid_recording.meta_data_keys) > 0
    assert valid_recording.location_id == 1
    assert len(valid_recording.track_ids) > 0
    assert len(valid_recording.frames) > 0
    assert valid_recording.lanelet2_map_file is None
    assert valid_recording.opendrive_map_file is None

    assert valid_highd_recording.id == 1
    assert valid_highd_recording.datset_name == "highd"
    assert len(valid_highd_recording.meta_data_keys) > 0
    assert valid_highd_recording.location_id == 2
    assert len(valid_highd_recording.track_ids) > 0
    assert len(valid_highd_recording.frames) > 0
    assert valid_highd_recording.lanelet2_map_file is None
    assert valid_highd_recording.opendrive_map_file is None


def test_recording_set_map_paths(
    valid_recording: Recording, lanelet2_path: Path, opendrive_path: Path
) -> None:
    valid_recording.lanelet2_map_file = lanelet2_path
    assert valid_recording.lanelet2_map_file == lanelet2_path

    valid_recording.opendrive_map_file = opendrive_path
    assert valid_recording.opendrive_map_file == opendrive_path


def test_recording_get_meta_data(
    valid_recording: Recording, valid_highd_recording: Recording
) -> None:
    # Valid meta_data_keys
    for meta_data_key in valid_recording.meta_data_keys:
        assert valid_recording.get_meta_data(meta_data_key) is not None
    for meta_data_key in valid_highd_recording.meta_data_keys:
        assert valid_highd_recording.get_meta_data(meta_data_key) is not None

    # Invalid meta_data_key
    with pytest.raises(KeyError):
        valid_recording.get_meta_data("invalid_key")


def test_recording_get_track_ids_at_frame(
    valid_recording: Recording, valid_highd_recording: Recording
) -> None:
    # Valid frames
    for frame in valid_recording.frames:
        assert len(valid_recording.get_track_ids_at_frame(frame)) > 0
    for frame in valid_highd_recording.frames:
        assert len(valid_highd_recording.get_track_ids_at_frame(frame)) > 0

    # Invalid frame
    assert len(valid_recording.get_track_ids_at_frame(-1)) == 0


def test_recording_get_track(
    valid_recording: Recording, valid_highd_recording: Recording
) -> None:
    # Valid track_ids
    for track_id in valid_recording.track_ids:
        assert valid_recording.get_track(track_id) is not None
    for track_id in valid_highd_recording.track_ids:
        assert valid_highd_recording.get_track(track_id) is not None

    # Invalid track_id
    with pytest.raises(KeyError):
        valid_recording.get_track(-1)


def test_recording_get_track_batches(
    valid_recording: Recording, valid_highd_recording: Recording
) -> None:
    # Valid track_batch_size
    # Batches with 1 track
    assert len(valid_recording.get_track_batches(1)) == 2
    # 1 batch of all tracks
    assert len(valid_recording.get_track_batches(-1)) == 1

    # highd
    # Batches with 1 track
    assert len(valid_highd_recording.get_track_batches(1)) == 2
    # 1 batch of all tracks
    assert len(valid_highd_recording.get_track_batches(-1)) == 1

    # Invalid track_batch_size
    assert len(valid_recording.get_track_batches(-2)) == 0


def test_recording_get_background_image(
    valid_recording: Recording, valid_highd_recording: Recording
) -> None:
    assert isinstance(valid_recording.get_background_image(), np.ndarray)
    assert isinstance(valid_highd_recording.get_background_image(), np.ndarray)


def test_recording_plot_track(
    valid_recording: Recording, valid_highd_recording: Recording
) -> None:
    # One track_id
    valid_recording.plot_track(1, "test")
    # Multiple track_ids (combined image)
    valid_recording.plot_track([1, 2], "test")
    # Multiple track_ids (separate images)
    valid_recording.plot_track([1, 2], "test", combine=False)

    # highd
    # One track_id
    valid_highd_recording.plot_track(1, "test")
    # Multiple track_ids (combined image)
    valid_highd_recording.plot_track([1, 2], "test")
    # Multiple track_ids (separate images)
    valid_highd_recording.plot_track([1, 2], "test", combine=False)

    # Invalid track_id
    with pytest.raises(KeyError):
        valid_recording.plot_track(-1, "test")
    with pytest.raises(TypeError):
        valid_recording.plot_track("1", "test")
    with pytest.raises(KeyError):
        valid_recording.plot_track([-1, -2], "test")
    with pytest.raises(KeyError):
        valid_recording.plot_track([-1, -2], "test", combine=False)
