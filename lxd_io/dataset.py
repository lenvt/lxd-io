from __future__ import annotations

import json
import re
import pandas as pd

from loguru import logger
from pathlib import Path

from .recording import Recording


class Dataset:
    TRACKS_FILE_SUFFIX = "_tracks.csv"
    TRACKS_META_FILE_SUFFIX = "_tracksMeta.csv"
    RECORDING_META_FILE_SUFFIX = "_recordingMeta.csv"
    BACKGROUND_IMAGE_FILE_SUFFIX = ("_background.png", "_highway.jpg", "_highway.png")

    def __init__(
        self,
        dataset_dir: Path,
        dataset_id: str | None = None,
        major_version_no: int | None = None,
        minor_version_no: int | None = None,
        background_image_scale_factor: float | None = None,
    ) -> None:
        self._dataset_dir = Path(dataset_dir)

        if not self._dataset_dir.exists():
            msg = "Specified dataset dir does not exist."
            raise FileNotFoundError(msg)

        if not self._dataset_dir.is_dir():
            msg = "Specified dataset dir is not a directory."
            raise FileNotFoundError(msg)

        self._dataset_id = dataset_id
        self._major_version_no = major_version_no
        self._minor_version_no = minor_version_no
        if (
            self._dataset_id is None
            or self._major_version_no is None
            or self._minor_version_no is None
        ):
            self._read_dataset_info_from_folder_name()

        logger.debug(
            "Read dataset {}-{} from: {}",
            self._dataset_id,
            self.version,
            self._dataset_dir,
        )

        # Load scale factor
        self._background_image_scale_factor = background_image_scale_factor
        if self._background_image_scale_factor is None:
            self._background_image_scale_factor = (
                self._load_background_image_scale_factor()
            )

        # Data dir
        self._data_dir = self._dataset_dir / "data"
        self._recording_meta_files = {}
        self._tracks_meta_files = {}
        self._tracks_files = {}
        self._background_image_files = {}
        self._recording_ids = []
        self._location_ids = []
        self._recordings_at_location = {}
        self._explore_data_dir()

        # Map dir
        self._maps_dir = self._dataset_dir / "maps"
        self._lanelet2_map_files_per_location = {}
        self._opendrive_map_files_per_location = {}
        self._explore_maps_dir()

    @property
    def id(self) -> str:
        return self._dataset_id

    @property
    def version(self) -> str:
        return f"{self._major_version_no}.{self._minor_version_no}"

    @property
    def recording_ids(self) -> list[int]:
        return self._recording_ids

    @property
    def location_ids(self) -> list[int]:
        return self._location_ids

    @property
    def recordings_at_location(self) -> dict[int, list[int]]:
        return self._recordings_at_location

    def _read_dataset_info_from_folder_name(self) -> None:
        """
        Try to read info from folder name
        Assumption: Format is either
        "<dataset_id>-dataset-v<major_version_nr>.<minor_version_nr>"
        or
        "<dataset_id>-v<major_version_nr>.<minor_version_nr>"
        or
        "<dataset_id>-dataset"
        or
        "<dataset_id>"
        """

        dataset_id = "unknown"
        major_version_no = 0
        minor_version_no = 0

        possible_patterns = {
            "name-dataset-version": "^(.+)-dataset-v([0-9]+).([0-9]+)$",
            "name-version": "^(.+)-v([0-9]+).([0-9]+)$",
            "name-dataset": "^(.+)-dataset$",
        }

        for pattern_name, pattern in possible_patterns.items():  # noqa: B007
            m = re.match(pattern, self._dataset_dir.name)
            if m:
                break

        if m is None:
            dataset_id = self._dataset_dir.name
        else:
            if "name" in pattern_name:
                dataset_id = m.group(1).lower()
            if "version" in pattern_name:
                major_version_no = int(m.group(2))
                minor_version_no = int(m.group(3))

        self._dataset_id = dataset_id
        self._major_version_no = major_version_no
        self._minor_version_no = minor_version_no

    def _load_background_image_scale_factor(self) -> float:
        if self._dataset_id is None:
            logger.warning(
                "Since dataset_id is None, background_image_scale_factor is assumed to be 1.0"
            )
            return 1.0

        scale_factor_json = (
            Path(__file__).parent / "background_image_scale_factors.json"
        )

        with scale_factor_json.open("r") as f:
            scale_factor_dict = json.load(f)

        background_scale_factor = None

        if self._dataset_id in scale_factor_dict:
            version_str = f"{self._major_version_no}.{self._minor_version_no}"
            if version_str in scale_factor_dict[self._dataset_id]:
                background_scale_factor = scale_factor_dict[self._dataset_id][
                    version_str
                ]

        if background_scale_factor is None:
            logger.warning(
                "Dataset id {} could not be found in {}. Scale factor is set to 1.0",
                self._dataset_id,
                scale_factor_json,
            )
            return 1.0

        return background_scale_factor

    def _explore_data_dir(self) -> None:
        logger.debug("Search for recordings in: {}", self._data_dir)

        if not self._data_dir.exists():
            msg = "Data dir does not exist."
            raise FileExistsError(msg)

        if not self._data_dir.is_dir():
            msg = "Data dir is not a directory."
            raise FileExistsError(msg)

        # Find files by recording ids
        def find_files_by_recording_ids(
            data_dir: Path, suffix: str | list[str]
        ) -> dict:
            files_by_recording_ids = {}
            if isinstance(suffix, str):
                suffix = [suffix]
            for sfx in suffix:
                files = data_dir.glob(f"*{sfx}")
                for f in files:
                    recording_id = int(re.match(f"(.*){sfx}", f.name).group(1))
                    files_by_recording_ids[recording_id] = f
            return files_by_recording_ids

        self._recording_meta_files = find_files_by_recording_ids(
            self._data_dir, self.RECORDING_META_FILE_SUFFIX
        )
        self._tracks_meta_files = find_files_by_recording_ids(
            self._data_dir, self.TRACKS_META_FILE_SUFFIX
        )
        self._tracks_files = find_files_by_recording_ids(
            self._data_dir, self.TRACKS_FILE_SUFFIX
        )
        self._background_image_files = find_files_by_recording_ids(
            self._data_dir, self.BACKGROUND_IMAGE_FILE_SUFFIX
        )

        all_potential_recording_ids = sorted(
            set(
                list(self._recording_meta_files.keys())
                + list(self._tracks_meta_files.keys())
                + list(self._tracks_files.keys())
                + list(self._background_image_files.keys())
            )
        )

        # Match files
        self._recording_ids = []
        for recording_id in all_potential_recording_ids:
            if (
                recording_id in self._recording_meta_files
                and recording_id in self._tracks_meta_files
                and recording_id in self._tracks_files
                and recording_id in self._background_image_files
            ):
                self._recording_ids.append(recording_id)

                # Get location id
                recording_meta = pd.read_csv(
                    self._recording_meta_files[recording_id]
                ).to_dict(orient="records")[0]
                location_id = recording_meta["locationId"]

                # Store location id
                if location_id not in self._location_ids:
                    self._location_ids.append(location_id)
                    self._recordings_at_location[location_id] = []
                self._recordings_at_location[location_id].append(recording_id)

            else:
                logger.warning("Incomplete data for recording {}", recording_id)

        # Sort location ids
        self._location_ids = sorted(self._location_ids)
        self._recordings_at_location = {
            key: sorted(self._recordings_at_location[key])
            for key in sorted(self._recordings_at_location.keys())
        }

        logger.debug(
            "Found complete data for {} recordings:\n{}",
            len(self._recording_ids),
            self._recording_ids,
        )

    def _explore_maps_dir(self) -> None:
        logger.debug("Search for maps in: {}", self._maps_dir)

        # Lanelet2
        lanelet2_dir = self._maps_dir / "lanelet2"
        self._find_lanelet2_maps(lanelet2_dir)

        # OpenDRIVE
        opendrive_dir = self._maps_dir / "opendrive"
        self._find_opendrive_maps(opendrive_dir)

    def _find_lanelet2_maps(self, lanelet2_dir: Path) -> None:
        if not lanelet2_dir.exists() or not lanelet2_dir.is_dir():
            logger.warning("Lanelet2 map directory does not exist at {}.", lanelet2_dir)
            return

        potential_lanelet2_maps = list(lanelet2_dir.rglob("*.osm"))

        # Assumption: Format is "X_Y.osm". X=location_id int, Y=location name str
        lanelet2_map_files_per_location = {}
        for f in potential_lanelet2_maps:
            location_id = None

            if f.stem.startswith("location"):
                m = re.match("location.*([0-9]+).osm", f.name)
                if m:
                    location_id = int(m.group(1))
            else:
                m = re.match("([0-9]+).*.osm", f.name)
                if m:
                    location_id = int(m.group(1))

            if location_id is not None:
                lanelet2_map_files_per_location[location_id] = f

        self._lanelet2_map_files_per_location = lanelet2_map_files_per_location

    def _find_opendrive_maps(self, opendrive_dir: Path) -> None:
        if not opendrive_dir.exists() or not opendrive_dir.is_dir():
            logger.warning(
                "OpenDrive map directory does not exist at {}.", opendrive_dir
            )
            return

        potential_opendrive_maps = list(opendrive_dir.rglob("*.xodr"))
        opendrive_map_files_per_location = {}
        for f in potential_opendrive_maps:
            # Assumption:
            # a) Folders with format "X_Y". X=location_id int, Y=location name str.
            # OR
            # b) Folders with format "X". X=location_id int.

            location_id = None

            folder_name = f.parent.name
            if folder_name == opendrive_dir.name:
                continue

            folder_name_pattern = "([0-9]+).*"
            m = re.match(folder_name_pattern, folder_name)
            if m:
                location_id = int(m.group(1))

            if location_id is not None:
                opendrive_map_files_per_location[location_id] = f

        self._opendrive_map_files_per_location = opendrive_map_files_per_location

    def get_recording(self, recording_id: str) -> Recording:
        logger.debug("Get recording {}.", recording_id)

        if recording_id not in self._recording_ids:
            msg = (
                f"Invalid recording ID. Available recordings are: {self._recording_ids}"
            )
            raise KeyError(msg)

        recording = Recording(
            recording_id,
            self._recording_meta_files[recording_id],
            self._tracks_meta_files[recording_id],
            self._tracks_files[recording_id],
            self._background_image_files[recording_id],
            self._background_image_scale_factor,
            f"{self.id}-v{self.version}",
        )

        location_id = recording.location_id
        if location_id in self._lanelet2_map_files_per_location:
            lanelet2_map_file = self._lanelet2_map_files_per_location[location_id]
            recording.lanelet2_map_file = lanelet2_map_file
        if location_id in self._opendrive_map_files_per_location:
            opendrive_map_file = self._opendrive_map_files_per_location[location_id]
            recording.opendrive_map_file = opendrive_map_file

        return recording

    def get_track_batches(
        self, track_batch_size: int, recording_ids: list[int] | None = None
    ) -> list[list[int]]:
        """
        Return a list of track batches (consisting of recording_id, start_track_idx and end_track_idx) for all recordings in the dataset.
        """

        track_batches = []

        for recording_id in self._recording_ids:
            if recording_ids is not None and recording_id not in recording_ids:
                continue
            track_batches.extend(
                self.get_recording(recording_id).get_track_batches(track_batch_size)
            )

        return track_batches
