import pytest

from pathlib import Path

from lxd_io import Dataset


@pytest.mark.skip(reason="Dataset conent unavailable in pipeline")
def test_lxd_io(dataset_path: Path) -> None:

    dataset = Dataset(dataset_path)

    print(f"Dataset ID: {dataset.id}")
    print(f"Dataset version: {dataset.version}")

    print(f"Available recordings: {dataset.recording_ids}")
    print(f"Available locations: {dataset.location_ids}")
    print(f"Available recordings at locations: {dataset.recordings_at_location}")

    # Get recording
    print("Get a recording")
    recording = dataset.get_recording(1)

    print("Recording meta data keys")
    print(recording.meta_data_keys)

    print("Recording location ID")
    print(recording.location_id)

    print("lanelet2 map file")
    print(recording.lanelet2_map_file)

    print("OpenDRIVE map file")
    print(recording.opendrive_map_file)

    # Get the list of track ids
    print("Recording track ids")
    print(recording.track_ids)

    # Get the list of frames
    print("Recording frames")
    print(recording.frames)

    # Get track_ids at certain frame
    track_ids_at_frame = recording.get_track_ids_at_frame(500)
    print("Track ids at certain frame")
    print(track_ids_at_frame)

    # Get a track
    print("Get track 25")
    track = recording.get_track(25)

    # Read meta data of track
    print("Track meta data keys")
    print(track.meta_data_keys)

    initial_frame = track.get_meta_data("initialFrame")
    print("Initial frame")
    print(initial_frame)

    # Read data of track
    print("Track data keys")
    print(track.data_keys)

    lon_velocity = track.get_data("lonVelocity")
    print("Lon velocity")
    print(lon_velocity)

    # Read data at a certain frame
    lon_velocity_at_frame = track.get_data_at_frame("lonVelocity", initial_frame + 2)
    print(f"lonVelocity at frame: {initial_frame + 2}")
    print(lon_velocity_at_frame)

    # Plot trajectory on image and save as jpg/png
    print("Plot track onto background image and save as image file")
    plot_dir = Path(__file__).parent
    recording.plot_track(25, plot_dir)
    recording.plot_track(track_ids_at_frame, plot_dir)
    dataset.get_recording(2).plot_track(25, plot_dir)
    dataset.get_recording(10).plot_track(150, plot_dir)
    dataset.get_recording(55).plot_track(200, plot_dir)


if __name__ == "__main__":
    test_lxd_io(Path("/path/to/exiD-dataset-v2.1"))
