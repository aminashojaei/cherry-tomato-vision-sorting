"""Temporal sampling and health-decision aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.schemas import TomatoTrack


@dataclass(frozen=True, slots=True)
class SamplingDecision:
    should_sample: bool
    inside_zone: bool


@dataclass(frozen=True, slots=True)
class TemporalAggregate:
    finalized: bool
    prediction: int | None
    probability: float | None
    confidence: float | None
    reason: str | None
    positive_count: int
    max_positive_run: int
    top_k_mean: float | None


class AsymmetricTemporalPolicy:
    """Lock unhealthy decisions early while requiring more evidence for healthy."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.zone = config["zone"]
        self.sample_stride = int(config["sample_stride_frames"])
        self.min_healthy_samples = int(config["min_samples_for_healthy"])

        evidence = config["unhealthy_evidence"]
        self.consecutive_positive_samples = int(
            evidence["consecutive_positive_samples"]
        )
        self.total_positive_samples = int(evidence["total_positive_samples"])
        self.top_k = int(evidence["top_k"])
        self.top_k_mean_threshold = float(evidence["top_k_mean_threshold"])

    def sampling_decision(
        self,
        track: TomatoTrack,
        frame_index: int,
        frame_shape: tuple[int, ...],
    ) -> SamplingDecision:
        height, width = frame_shape[:2]
        center_x = (track.bbox.x1 + track.bbox.x2) / (2 * width)
        center_y = (track.bbox.y1 + track.bbox.y2) / (2 * height)
        inside = float(self.zone["x_min_ratio"]) <= center_x <= float(
            self.zone["x_max_ratio"]
        ) and float(self.zone["y_min_ratio"]) <= center_y <= float(
            self.zone["y_max_ratio"]
        )

        if inside and track.zone_entry_frame is None:
            track.zone_entry_frame = frame_index
        if inside:
            track.last_zone_frame = frame_index

        track.in_classification_zone = inside
        should_sample = (
            inside
            and not track.health_decision_locked
            and frame_index % self.sample_stride == 0
        )
        return SamplingDecision(should_sample, inside)

    def aggregate(self, track: TomatoTrack) -> TemporalAggregate:
        probabilities = track.health_history
        predictions = track.health_prediction_history
        positive_count = sum(value == 1 for value in predictions)

        current_run = 0
        max_positive_run = 0
        for value in predictions:
            current_run = current_run + 1 if value == 1 else 0
            max_positive_run = max(max_positive_run, current_run)

        top_values = sorted(probabilities, reverse=True)[: self.top_k]
        top_k_mean = (
            sum(top_values) / len(top_values) if len(top_values) == self.top_k else None
        )

        reason = None
        if max_positive_run >= self.consecutive_positive_samples:
            reason = "consecutive_unhealthy_samples"
        elif positive_count >= self.total_positive_samples:
            reason = "total_unhealthy_samples"
        elif top_k_mean is not None and top_k_mean >= self.top_k_mean_threshold:
            reason = "top_k_probability_mean"

        if reason:
            probability = max(probabilities)
            return TemporalAggregate(
                True,
                1,
                probability,
                probability,
                reason,
                positive_count,
                max_positive_run,
                top_k_mean,
            )

        if len(probabilities) >= self.min_healthy_samples:
            probability = sum(probabilities) / len(probabilities)
            return TemporalAggregate(
                True,
                0,
                probability,
                1.0 - probability,
                "minimum_samples_without_unhealthy_evidence",
                positive_count,
                max_positive_run,
                top_k_mean,
            )

        return TemporalAggregate(
            False,
            None,
            None,
            None,
            None,
            positive_count,
            max_positive_run,
            top_k_mean,
        )
