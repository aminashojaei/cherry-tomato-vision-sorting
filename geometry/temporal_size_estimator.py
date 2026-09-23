"""Pixel-space size measurement inside a controlled video zone."""

from __future__ import annotations

from typing import Any

from core.schemas import TomatoTrack


class TemporalShortSideMeasurer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.zone = config["zone"]

    def observe(
        self,
        track: TomatoTrack,
        frame_index: int,
        frame_shape: tuple[int, ...],
    ) -> bool:
        if not self.config.get("enabled", True) or track.size_finalized:
            return False

        height, width = frame_shape[:2]
        center_x = (track.bbox.x1 + track.bbox.x2) / (2 * width)
        center_y = (track.bbox.y1 + track.bbox.y2) / (2 * height)
        inside = float(self.zone["x_min_ratio"]) <= center_x <= float(
            self.zone["x_max_ratio"]
        ) and float(self.zone["y_min_ratio"]) <= center_y <= float(
            self.zone["y_max_ratio"]
        )
        track.in_size_measurement_zone = inside

        if not inside or track.detection_confidence < float(
            self.config["min_detection_confidence"]
        ):
            return False

        margin = int(self.config.get("frame_edge_margin_px", 0))
        touches_edge = (
            track.bbox.x1 <= margin
            or track.bbox.y1 <= margin
            or track.bbox.x2 >= width - margin
            or track.bbox.y2 >= height - margin
        )
        if self.config.get("reject_if_touches_frame_edge", True) and touches_edge:
            return False

        short_side = float(min(track.bbox.width, track.bbox.height))
        track.size_history = [short_side]
        track.size_frames = [frame_index]
        track.size_metric_px = short_side

        thresholds = self.config["thresholds"]
        labels = self.config["labels"]
        if short_side <= float(thresholds["small_max_px"]):
            track.size_class = str(labels["small"])
        elif short_side <= float(thresholds["medium_max_px"]):
            track.size_class = str(labels["medium"])
        else:
            track.size_class = str(labels["large"])

        track.size_finalized = True
        track.size_finalization_frame = frame_index
        return True
