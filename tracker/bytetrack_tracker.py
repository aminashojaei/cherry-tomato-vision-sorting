"""Independent adapter around the actual Ultralytics BYTETracker."""

from __future__ import annotations

import inspect
from argparse import Namespace
from typing import Any

import numpy as np

from core.schemas import BoundingBox, Detection, TrackerObservation


def instantiate_bytetracker(
    tracker_class: type[Any], args: Namespace, frame_rate: float
) -> Any:
    """Handle both current BYTETracker(args) and historical FPS-aware signatures."""
    normalized_fps = max(1, round(frame_rate))
    try:
        parameters = inspect.signature(tracker_class).parameters
    except (TypeError, ValueError):
        return tracker_class(args=args)

    if "frame_rate" in parameters:
        return tracker_class(args=args, frame_rate=normalized_fps)
    if "fps" in parameters:
        return tracker_class(args=args, fps=normalized_fps)
    return tracker_class(args=args)


class ByteTrackTracker:
    def __init__(self, config: dict[str, Any], frame_rate: float) -> None:
        import torch
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._torch = torch
        self._first_seen: dict[int, int] = {}

        args = Namespace(
            track_high_thresh=float(config["high_threshold"]),
            track_low_thresh=float(config["low_threshold"]),
            new_track_thresh=float(config["new_track_threshold"]),
            track_buffer=int(config["lost_track_buffer"]),
            match_thresh=float(config["match_threshold"]),
            fuse_score=bool(config.get("fuse_score", True)),
            mot20=False,
        )
        self._tracker = instantiate_bytetracker(BYTETracker, args, frame_rate)

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame_shape: tuple[int, ...],
    ) -> list[TrackerObservation]:
        from ultralytics.engine.results import Boxes

        rows = [
            [*detection.bbox.as_xyxy(), detection.confidence, float(detection.class_id)]
            for detection in detections
        ]
        matrix = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        boxes = Boxes(self._torch.from_numpy(matrix), orig_shape=frame_shape[:2])
        tracked = self._tracker.update(boxes)

        observations: list[TrackerObservation] = []
        for row in tracked:
            track_id = int(row[4])
            self._first_seen.setdefault(track_id, frame_index)
            observations.append(
                TrackerObservation(
                    track_id=track_id,
                    bbox=BoundingBox(*(float(v) for v in row[:4])),
                    detection_confidence=float(row[5]),
                    age=frame_index - self._first_seen[track_id] + 1,
                    active_status=True,
                    matched_in_current_frame=True,
                )
            )
        return observations
