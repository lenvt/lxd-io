from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from loguru import logger
from pathlib import Path

from .track import Track


class Recording:
    def __init__(
        self,
        recording_id: int,
        recording_meta_file: Path,
        tracks_meta_file: Path,
        tracks_file: Path,
        background_image_file: Path,
        background_image_scale_factor: float = 1.0,
        dataset: str = "unknown-v0.0",
    ) -> None:
        self._recording_id = recording_id
        self._datset_name = dataset
        self._background_image_file = background_image_file
        self._background_image_scale_factor = background_image_scale_factor

        logger.debug("Load csv files for recording {}.", recording_id)

        self._recording_meta_data = pl.read_csv(recording_meta_file).to_dicts()[0]

        self._tracks_meta_data = pl.read_csv(tracks_meta_file)

        self._tracks_file = tracks_file
        self._tracks_data = None

        self._background_image = None

        self._lanelet2_map_file = None
        self._opendrive_map_file = None

        self._track_id_key = "trackId" if "trackId" in self._tracks_meta_data else "id"
        self._is_highd = "trackId" not in self._tracks_meta_data

        self._track_ids = self._tracks_meta_data[self._track_id_key].to_list()

        initial_frame = 0 if not self._is_highd else 1
        try:
            max_frame = int(
                self._recording_meta_data["duration"]
                * self._recording_meta_data["frameRate"]
            )
        except KeyError as e:
            msg = f"{recording_id:02d}_recordingMeta.csv data does not contain key {e}"
            raise KeyError(msg) from e
        self._frames = np.arange(initial_frame, max_frame + 1).tolist()

        self._tracks = {}

    def _read_tracks_file(self, tracks_file: Path) -> pl.DataFrame:
        logger.debug("Load tracks file for recording {}", self._recording_id)

        schema_overrides = None
        if not self._is_highd:
            schema_overrides = {col: pl.Utf8 for col in Track.SEMICOLON_LIST_COLUMNS}
        return pl.read_csv(tracks_file, schema_overrides=schema_overrides)

    @property
    def id(self) -> int:
        return self._recording_id

    @property
    def datset_name(self) -> int:
        return self._datset_name

    @property
    def meta_data_keys(self) -> list[str]:
        return list(self._recording_meta_data)

    @property
    def location_id(self) -> int:
        return self._recording_meta_data["locationId"]

    @property
    def track_ids(self) -> list[int]:
        return self._track_ids

    @property
    def frames(self) -> list[int]:
        return self._frames

    @property
    def lanelet2_map_file(self) -> Path:
        return self._lanelet2_map_file

    @lanelet2_map_file.setter
    def lanelet2_map_file(self, lanelet2_map_file: Path) -> None:
        self._lanelet2_map_file = lanelet2_map_file

    @property
    def opendrive_map_file(self) -> Path:
        return self._opendrive_map_file

    @opendrive_map_file.setter
    def opendrive_map_file(self, opendrive_map_file: Path) -> None:
        self._opendrive_map_file = opendrive_map_file

    def _get_tracks_data(self) -> pl.DataFrame:
        if self._tracks_data is None:
            self._tracks_data = self._read_tracks_file(self._tracks_file)
        return self._tracks_data

    def get_meta_data(self, key: str) -> any:
        if key not in self._recording_meta_data:
            msg = f"Invalid recording meta data key: {key}"
            raise KeyError(msg)

        data = self._recording_meta_data[key]

        return data

    def get_track_ids_at_frame(self, frame: int) -> list[Track]:
        tracks_data = self._get_tracks_data()

        track_ids = (
            tracks_data.filter(pl.col("frame") == frame)
            .select(self._track_id_key)
            .sort(self._track_id_key)
            .to_series()
            .to_list()
        )

        return track_ids

    def get_track(self, track_id: int) -> Track:
        if track_id not in self._track_ids:
            msg = f"Invalid track ID {track_id} for recording {self._recording_id}."
            raise KeyError(msg)

        tracks_data = self._get_tracks_data()

        if track_id not in self._tracks:
            track_meta_data = self._tracks_meta_data.filter(
                pl.col(self._track_id_key) == track_id
            ).to_dicts()[0]
            track_data = tracks_data.filter(pl.col(self._track_id_key) == track_id)
            track = Track(track_id, track_meta_data, track_data)
            self._tracks[track_id] = track

        track = self._tracks[track_id]
        return track

    def get_track_batches(self, track_batch_size: int) -> list[dict]:
        """
        Return a list of track batches (consisting of recording_id, start_track_idx and end_track_idx) for this recording.
        """

        track_batches = []

        n_tracks = len(self._track_ids)

        if track_batch_size == -1:
            n_batches = 1
            track_batch_size = n_tracks
        else:
            n_batches = n_tracks // track_batch_size
            if n_tracks % track_batch_size:
                n_batches += 1

        for i in range(n_batches):
            start_track_idx = i * track_batch_size
            end_track_idx = start_track_idx + track_batch_size - 1

            if end_track_idx > n_tracks:
                end_track_idx = n_tracks - 1

            track_batches.append(
                {
                    "recording_id": self._recording_id,
                    "start_track_idx": start_track_idx,
                    "end_track_idx": end_track_idx,
                }
            )

        return track_batches

    def get_background_image(self) -> np.ndarray:
        if self._background_image is None:
            self._background_image = plt.imread(self._background_image_file)

        return self._background_image

    def plot_track(
        self, track_id: int | list[int], folder: Path, combine: bool = True
    ) -> None:
        if isinstance(track_id, list):
            return self._plot_multiple_tracks(track_id, folder, combine)

        if isinstance(track_id, int):
            return self._plot_single_track(track_id, folder)

        msg = "track_id must be of type int or list[int]"
        raise TypeError(msg)

    def _plot_single_track(self, track_id: int, folder: Path) -> None:
        logger.debug(f"Plot recording {self._recording_id}, track {track_id}")

        track = self.get_track(track_id)
        if self._is_highd:
            background_image_trajectory = track.get_background_image_trajectory(
                0.1, self._background_image_scale_factor
            )
        else:
            background_image_trajectory = track.get_background_image_trajectory(
                self._recording_meta_data["orthoPxToMeter"],
                self._background_image_scale_factor,
            )

        background_image = self.get_background_image()
        plot_file = Path(folder) / f"{self._recording_id}_{track_id}.jpg"

        height, width, _ = background_image.shape
        dpi = 100
        f, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax.imshow(background_image)
        ax.plot(
            background_image_trajectory[:, 0],
            background_image_trajectory[:, 1],
            color="red",
            linewidth=2,
        )
        ax.axis("off")
        f.savefig(plot_file, bbox_inches="tight", pad_inches=0)
        plt.close(f)

        return plot_file

    def _plot_multiple_tracks(
        self, track_ids: list, folder: Path, combine_plots: bool
    ) -> None:
        if combine_plots:
            # Plot all trajectories in one image
            logger.debug(f"Plot recording {self._recording_id}, tracks {track_ids}")

            background_image = self.get_background_image()
            plot_file = (
                Path(folder) / f"{self._recording_id}_{len(track_ids)}_tracks.jpg"
            )

            height, width, _ = background_image.shape
            dpi = 100
            f, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
            ax.imshow(background_image)

            for track_id in track_ids:
                track = self.get_track(track_id)

                if self._is_highd:
                    background_image_trajectory = track.get_background_image_trajectory(
                        0.1, self._background_image_scale_factor
                    )
                else:
                    background_image_trajectory = track.get_background_image_trajectory(
                        self._recording_meta_data["orthoPxToMeter"],
                        self._background_image_scale_factor,
                    )

                ax.plot(
                    background_image_trajectory[:, 0],
                    background_image_trajectory[:, 1],
                    color="red",
                    linewidth=2,
                )

            ax.axis("off")
            f.savefig(plot_file, bbox_inches="tight", pad_inches=0)
            plt.close(f)

            return plot_file

        # Plot all trajectories in separate images
        return [self._plot_single_track(track_id, folder) for track_id in track_ids]
