import numpy as np
import pandas as pd


class Track:

    def __init__(self, track_id: int, track_meta_data: dict, track_data: pd.DataFrame) -> None:

        self._track_id = track_id
        self._track_meta_data = track_meta_data
        self._track_data = track_data

        if "xCenter" in self._track_data:
            self._is_highd = False
        else:
            self._is_highd = True

        self._frames = self._track_data["frame"].tolist()

    @property
    def id(self) -> int:
        return self._track_id

    @property
    def meta_data_keys(self) -> list:
        return list(self._track_meta_data.keys())

    @property
    def data_keys(self) -> list:
        return list(self._track_data.columns)

    def get_meta_data(self, key: str) -> any:

        if key not in self._track_meta_data:
            msg = f"Invalid track meta data key: {key}"
            raise KeyError(msg)

        data = self._track_meta_data[key]

        return data

    def get_data(self, key: str) -> np.ndarray:

        if key not in self._track_data.columns:
            msg = f"Invalid track data key: {key}"
            raise KeyError(msg)

        data = self._track_data[key].to_numpy()

        return data

    def get_data_at_frame(self, key: str, frame: int) -> any:

        if key not in self._track_data.columns:
            msg = f"Invalid track data key: {key}"
            raise KeyError(msg)

        if not (self._frames[0] <= frame <= self._frames[-1]):
            msg = f"Invalid frame: {frame}"
            raise KeyError(msg)

        data = self._track_data.loc[self._track_data["frame"] == frame][key].to_numpy()[0]

        return data

    def get_data_between_frames(self, key: str, frame0: int, frame1: int) -> any:

        if key not in self._track_data.columns:
            msg = f"Invalid track data key: {key}"
            raise KeyError(msg)

        for frame in (frame0, frame1):
            if not (self._frames[0] <= frame <= self._frames[-1]):
                msg = f"Invalid frame: {frame}"
                raise KeyError(msg)

        frames = np.arange(frame0, frame1 + 1)
        data = self._track_data.loc[self._track_data["frame"].isin(frames)].sort_values(by="frame")[key].to_numpy()

        return data

    def get_local_trajectory(self) -> np.ndarray:

        if self._is_highd:
            x = self._track_data["x"] + self._track_data["width"] / 2
            y = self._track_data["y"] + self._track_data["height"] / 2
        else:
            x = self._track_data["xCenter"]
            y = self._track_data["yCenter"]

        trajectory = np.vstack((x, y)).T

        return trajectory

    def get_utm_image_trajectory(self, utm_origin_x: float, utm_origin_y: float) -> np.ndarray:

        if self._is_highd:
            return self.get_local_trajectory()

        x_utm = self._track_data["xCenter"] + utm_origin_x
        y_utm = self._track_data["yCenter"] + utm_origin_y

        utm_trajectory = np.vstack((x_utm, y_utm)).T

        return utm_trajectory

    def get_background_image_trajectory(self, px2m_factor: float, background_image_scale_factor: float = 1.0) -> np.ndarray:

        if self._is_highd:
            x_image = (self._track_data["x"] + self._track_data["width"] / 2) / px2m_factor
            y_image = (self._track_data["y"] + self._track_data["height"] / 2) / px2m_factor
        else:
            x_image = self._track_data["xCenter"] / px2m_factor
            y_image = - self._track_data["yCenter"] / px2m_factor
        image_trajectory = np.vstack((x_image, y_image)).T

        # Scale down factor
        image_trajectory = image_trajectory / background_image_scale_factor

        return image_trajectory

    def get_bbox_at_frame(self, frame: int, visualization_mode: bool = False, ortho_px_to_m: float = 0.1) -> np.ndarray:

        if self._is_highd:
            length = self.get_meta_data("width")
            width = self.get_meta_data("height")
            x_center = self.get_data_at_frame("x", frame) + length / 2
            y_center = self.get_data_at_frame("y", frame) + width / 2
            heading = (self.get_meta_data("drivingDirection") - 2) * 180

        else:
            x_center = self.get_data_at_frame("xCenter", frame)
            y_center = self.get_data_at_frame("yCenter", frame)
            length = self.get_meta_data("length")
            width = self.get_meta_data("width")
            heading = self.get_data_at_frame("heading", frame)

        if visualization_mode:
            x_center = x_center / ortho_px_to_m
            y_center = -y_center / ortho_px_to_m
            length = length / ortho_px_to_m
            width = width / ortho_px_to_m
            heading = heading * (-1)
            if heading < 0:
                heading += 360

        return self._bbox_to_vertices(x_center, y_center, length, width, np.deg2rad(heading))

    @staticmethod
    def _bbox_to_vertices(
            x_center: np.ndarray,
            y_center: np.ndarray,
            length: np.ndarray,
            width: np.ndarray,
            heading: np.ndarray
        ) -> np.ndarray:
        """
        Calculate the corners of a rotated bbox from the position, shape and heading for every timestamp.

        :param x_center: x coordinates of the object center positions [num_timesteps]
        :param y_center: y coordinates of the object center positions [num_timesteps]
        :param length: objects lengths [num_timesteps]
        :param width: object widths [num_timesteps]
        :param heading: object heading (rad) [num_timesteps]
        :return: Numpy array in the shape [num_timesteps, 4 (corners), 2 (dimensions)]
        """
        centroids = np.column_stack([x_center, y_center])

        # Precalculate all components needed for the corner calculation
        half_length = length / 2
        half_width = width / 2
        heading_cos = np.cos(heading)
        heading_sin = np.sin(heading)

        lc = half_length * heading_cos
        ls = half_length * heading_sin
        wc = half_width * heading_cos
        ws = half_width * heading_sin

        # Calculate all four rotated bbox corner positions assuming the object is located at the origin.
        # To do so, rotate the corners at [+/- length/2, +/- width/2] as given by the orientation.
        # Use a vectorized approach using precalculated components for maximum efficiency
        rotated_bbox_vertices = np.empty((centroids.shape[0], 4, 2))

        # Front-right corner
        rotated_bbox_vertices[:, 0, 0] = lc - ws
        rotated_bbox_vertices[:, 0, 1] = ls + wc

        # Rear-right corner
        rotated_bbox_vertices[:, 1, 0] = -lc - ws
        rotated_bbox_vertices[:, 1, 1] = -ls + wc

        # Rear-left corner
        rotated_bbox_vertices[:, 2, 0] = -lc + ws
        rotated_bbox_vertices[:, 2, 1] = -ls - wc

        # Front-left corner
        rotated_bbox_vertices[:, 3, 0] = lc + ws
        rotated_bbox_vertices[:, 3, 1] = ls - wc

        # Move corners of rotated bounding box from the origin to the object's location
        rotated_bbox_vertices = rotated_bbox_vertices + np.expand_dims(centroids, axis=1)

        return rotated_bbox_vertices
