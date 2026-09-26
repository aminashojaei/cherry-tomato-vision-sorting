"""Durable application decisions independent of tracker internals."""

from __future__ import annotations

from typing import Any

from classifier.temporal_policy import TemporalAggregate
from core.schemas import (
    BBoxObservation,
    ClassificationResult,
    TomatoTrack,
    TrackerObservation,
)


class TrackManager:
    def __init__(
        self, save_bbox_history: bool = True, calyx_threshold: float = 0.5
    ) -> None:
        self._tracks: dict[int, TomatoTrack] = {}
        self._save_bbox_history = save_bbox_history
        self._calyx_threshold = calyx_threshold

    def upsert(self, observation: TrackerObservation, frame_index: int) -> TomatoTrack:
        track = self._tracks.get(observation.track_id)
        if track is None:
            track = TomatoTrack(
                track_id=observation.track_id,
                first_frame=frame_index,
                last_frame=frame_index,
                bbox=observation.bbox,
                detection_confidence=observation.detection_confidence,
            )
            self._tracks[observation.track_id] = track

        track.last_frame = frame_index
        track.bbox = observation.bbox
        track.detection_confidence = observation.detection_confidence
        track.age = observation.age
        track.active_status = observation.active_status
        track.matched_in_current_frame = observation.matched_in_current_frame
        track.passed = False

        if self._save_bbox_history:
            track.bbox_history.append(
                BBoxObservation(
                    frame_index, observation.bbox, observation.detection_confidence
                )
            )
        return track

    def get_track(self, track_id: int) -> TomatoTrack:
        return self._tracks[track_id]

    def save_temporal_observation(
        self, track_id: int, result: ClassificationResult, frame_index: int
    ) -> TomatoTrack:
        track = self.get_track(track_id)
        track.health_history.append(result.health_probability)
        track.calyx_history.append(result.calyx_probability)
        track.health_prediction_history.append(result.health_prediction)
        track.calyx_prediction_history.append(result.calyx_prediction)
        track.classification_frames.append(frame_index)
        probability = sum(track.calyx_history) / len(track.calyx_history)
        track.calyx_probability = probability
        track.calyx_prediction = int(probability >= self._calyx_threshold)
        track.calyx_confidence = (
            probability if track.calyx_prediction else 1 - probability
        )
        return track

    def apply_temporal_aggregate(
        self, track_id: int, aggregate: TemporalAggregate, frame_index: int
    ) -> TomatoTrack:
        track = self.get_track(track_id)
        track.health_positive_count = aggregate.positive_count
        track.health_max_positive_run = aggregate.max_positive_run
        track.health_top_k_mean = aggregate.top_k_mean
        if aggregate.finalized:
            track.classified = True
            track.health_decision_locked = True
            track.health_prediction = aggregate.prediction
            track.health_probability = aggregate.probability
            track.health_confidence = aggregate.confidence
            track.health_decision_reason = aggregate.reason
            track.health_decision_frame = frame_index
            track.classification_frame = frame_index
        return track

    def record_quality_failure(
        self,
        track_id: int,
        frame_index: int,
        reasons: list[str],
        metrics: dict[str, float],
    ) -> None:
        self.get_track(track_id).quality_gate_failures.append(
            {"frame": frame_index, "reasons": list(reasons), "metrics": dict(metrics)}
        )

    def update_inactive_tracks(self, observed_track_ids: set[int]) -> None:
        for track in self._tracks.values():
            if track.track_id not in observed_track_ids:
                track.active_status = False
                track.matched_in_current_frame = False

    def mark_passed_tracks(self, frame_index: int, lost_track_buffer: int) -> None:
        for track in self._tracks.values():
            if (
                track.classified
                and not track.active_status
                and not track.passed
                and frame_index - track.last_frame >= lost_track_buffer
            ):
                track.passed = True
                track.passed_frame = frame_index

    def finalize_all(self) -> None:
        for track in self._tracks.values():
            track.active_status = False

    def get_all_tracks(self) -> list[TomatoTrack]:
        return sorted(
            self._tracks.values(), key=lambda item: (item.first_frame, item.track_id)
        )
