"""Substitutable boundaries between pipeline components."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from core.schemas import ClassificationResult, Detection, TomatoTrack, TrackerObservation


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class Tracker(Protocol):
    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        frame_shape: tuple[int, ...],
    ) -> list[TrackerObservation]: ...


class Classifier(Protocol):
    def classify(self, crop: np.ndarray) -> ClassificationResult: ...


class SizeEstimator(Protocol):
    def observe(
        self, track: TomatoTrack, frame_index: int, frame_shape: tuple[int, ...]
    ) -> bool: ...
