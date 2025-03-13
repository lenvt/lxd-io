import pytest
from pathlib import Path
from lxd_io.dataset import Dataset
  

@pytest.fixture
def invalid_dataset_dir():
    return Path("test/data/invalid-dataset")

@pytest.fixture
def valid_dataset_with_version_dir():
    return Path("test/data/valid-dataset-v1.5")

@pytest.fixture
def valid_dataset_without_version_dir():
    return Path("test/data/valid-dataset")

@pytest.fixture
def valid_dataset_only_name_dir():
    return Path("test/data/valid_dataset")

def test_dataset_initialization(
        invalid_dataset_dir: Path,
        valid_dataset_with_version_dir: Path,
        valid_dataset_without_version_dir: Path,
        valid_dataset_only_name_dir: Path
    ):

    for dataset_dir in (
            invalid_dataset_dir,
            valid_dataset_with_version_dir,
            valid_dataset_without_version_dir,
            valid_dataset_only_name_dir
        ):

        Dataset(dataset_dir)

    with pytest.raises(FileNotFoundError):
        Dataset(Path("/path/to/invalid/dataset"))

def test_dataset_read_dataset_info_from_folder_name(
        invalid_dataset_dir: Path,
        valid_dataset_with_version_dir: Path,
        valid_dataset_without_version_dir: Path,
        valid_dataset_only_name_dir: Path
    ):

    dataset = Dataset(invalid_dataset_dir)
    assert dataset.id == "invalid"
    assert dataset.version == "0.0"

    dataset = Dataset(valid_dataset_with_version_dir)
    assert dataset.id == "valid"
    assert dataset.version == "1.5"

    dataset = Dataset(valid_dataset_without_version_dir)
    assert dataset.id == "valid"
    assert dataset.version == "0.0"

    dataset = Dataset(valid_dataset_only_name_dir)
    assert dataset.id == "valid_dataset"
    assert dataset.version == "0.0"

def test_dataset_load_background_image_scale_factor(
        invalid_dataset_dir: Path,
        valid_dataset_with_version_dir: Path,
        valid_dataset_without_version_dir: Path,
        valid_dataset_only_name_dir: Path
    ):

    for dataset_dir in (
            invalid_dataset_dir,
            valid_dataset_with_version_dir,
            valid_dataset_without_version_dir,
            valid_dataset_only_name_dir
        ):
        
        dataset = Dataset(dataset_dir)
        assert dataset._background_image_scale_factor == 1.0

def test_dataset_explore_data_dir(
        invalid_dataset_dir: Path,
        valid_dataset_with_version_dir: Path,
        valid_dataset_without_version_dir: Path,
        valid_dataset_only_name_dir: Path
    ):

    dataset = Dataset(invalid_dataset_dir)
    assert len(dataset.recording_ids) == 1
    assert len(dataset.location_ids) == 1

    for dataset_dir in (
            valid_dataset_with_version_dir,
            valid_dataset_without_version_dir,
            valid_dataset_only_name_dir
        ):
        dataset = Dataset(dataset_dir)
        assert len(dataset.recording_ids) == 2
        assert len(dataset.location_ids) == 2

def test_dataset_explore_maps_dir(
        invalid_dataset_dir: Path,
        valid_dataset_with_version_dir: Path,
        valid_dataset_without_version_dir: Path,
        valid_dataset_only_name_dir: Path
    ):

    dataset = Dataset(invalid_dataset_dir)
    assert len(dataset._lanelet2_map_files_per_location) == 1
    assert len(dataset._opendrive_map_files_per_location) == 0

    for dataset_dir in (
            valid_dataset_with_version_dir,
            valid_dataset_without_version_dir,
            valid_dataset_only_name_dir
        ):
        dataset = Dataset(dataset_dir)
        assert len(dataset._lanelet2_map_files_per_location) == 1
        assert len(dataset._opendrive_map_files_per_location) == 1

def test_dataset_get_recording(
        invalid_dataset_dir: Path,
        valid_dataset_with_version_dir: Path,
        valid_dataset_without_version_dir: Path,
        valid_dataset_only_name_dir: Path
    ):

    dataset = Dataset(invalid_dataset_dir)
    for recording_id in recording_id:
        with pytest.raises(KeyError):
            dataset.get_recording(recording_id)

    for dataset_dir in (
            valid_dataset_with_version_dir,
            valid_dataset_without_version_dir,
            valid_dataset_only_name_dir
        ):
        dataset = Dataset(dataset_dir)
        for recording_id in dataset.recording_ids:
            dataset.get_recording(recording_id) 

def test_dataset_get_track_batches(
        invalid_dataset_dir: Path,
        valid_dataset_with_version_dir: Path,
        valid_dataset_without_version_dir: Path,
        valid_dataset_only_name_dir: Path
    ):

    dataset = Dataset(invalid_dataset_dir)
    with pytest.raises(KeyError):
        dataset.get_track_batches(10)
    with pytest.raises(KeyError):
        dataset.get_track_batches(10, [1])

    for dataset_dir in (
            valid_dataset_with_version_dir,
            valid_dataset_without_version_dir,
            valid_dataset_only_name_dir
        ):
        dataset = Dataset(dataset_dir)
        for recording_id in dataset.recording_ids:
            for track_id in dataset.get_recording(recording_id).track_ids:
                assert len(dataset.get_track_batches(track_id)) == 2
                assert len(dataset.get_track_batches(track_id, [1])) == 1
