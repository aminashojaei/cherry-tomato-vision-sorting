"""Typed, implementation-independent pipeline objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_valid(self) -> bool:
        return (
            all(isfinite(v) for v in self.as_xyxy())
            and self.width > 0
            and self.height > 0
        )

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def expand_by_ratio(self, ratio: float) -> BoundingBox:
        pad_x = self.width * ratio
        pad_y = self.height * ratio
        return BoundingBox(
            self.x1 - pad_x, self.y1 - pad_y, self.x2 + pad_x, self.y2 + pad_y
        )

    def clip_to_frame(self, width: int, height: int) -> BoundingBox:
        return BoundingBox(
            max(0.0, min(float(width), self.x1)),
            max(0.0, min(float(height), self.y1)),
            max(0.0, min(float(width), self.x2)),
            max(0.0, min(float(height), self.y2)),
        )

    def outside_ratio(self, width: int, height: int) -> float:
        if self.area <= 0:
            return 1.0
        visible_area = self.clip_to_frame(width, height).area
        return max(0.0, min(1.0, 1.0 - visible_area / self.area))


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int


@dataclass(frozen=True, slots=True)
class TrackerObservation:
    track_id: int
    bbox: BoundingBox
    detection_confidence: float
    age: int
    active_status: bool
    matched_in_current_frame: bool


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    health_probability: float
    health_prediction: int
    health_confidence: float
    calyx_probability: float
    calyx_prediction: int
    calyx_confidence: float


@dataclass(frozen=True, slots=True)
class BBoxObservation:
    frame_index: int
    bbox: BoundingBox
    detection_confidence: float


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    accepted: bool
    reasons: list[str]
    metrics: dict[str, float]


@dataclass(slots=True)
class TomatoTrack:
    track_id: int
    first_frame: int
    last_frame: int
    bbox: BoundingBox
    detection_confidence: float
    age: int = 1
    active_status: bool = True
    matched_in_current_frame: bool = True
    bbox_history: list[BBoxObservation] = field(default_factory=list)
    classified: bool = False
    health_prediction: int | None = None
    health_probability: float | None = None
    health_confidence: float | None = None
    calyx_prediction: int | None = None
    calyx_probability: float | None = None
    calyx_confidence: float | None = None
    size_class: str | None = None
    size_metric_px: float | None = None
    classification_frame: int | None = None
    health_history: list[float] = field(default_factory=list)
    calyx_history: list[float] = field(default_factory=list)
    health_prediction_history: list[int] = field(default_factory=list)
    calyx_prediction_history: list[int] = field(default_factory=list)
    classification_frames: list[int] = field(default_factory=list)
    latency_information: dict[str, float] = field(default_factory=dict)
    quality_gate_failures: list[dict[str, Any]] = field(default_factory=list)
    in_classification_zone: bool = False
    zone_entry_frame: int | None = None
    last_zone_frame: int | None = None
    health_decision_locked: bool = False
    health_decision_reason: str | None = None
    health_decision_frame: int | None = None
    health_positive_count: int = 0
    health_max_positive_run: int = 0
    health_top_k_mean: float | None = None
    size_history: list[float] = field(default_factory=list)
    size_frames: list[int] = field(default_factory=list)
    size_finalized: bool = False
    size_finalization_frame: int | None = None
    in_size_measurement_zone: bool = False
    passed: bool = False
    passed_frame: int | None = None
