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

    def save_classification(
        self,
        track_id: int,
        result: ClassificationResult,
        frame_index: int,
    ) -> TomatoTrack:
        track = self.get_track(track_id)
        if track.classified:
            raise ValueError(f"Track {track_id} has already been classified.")

        track.health_probability = result.health_probability
        track.health_prediction = result.health_prediction
        track.health_confidence = result.health_confidence
        track.calyx_probability = result.calyx_probability
        track.calyx_prediction = result.calyx_prediction
        track.calyx_confidence = result.calyx_confidence
        track.classification_frame = frame_index
        track.health_history.append(result.health_probability)
        track.calyx_history.append(result.calyx_probability)
        track.health_prediction_history.append(result.health_prediction)
        track.calyx_prediction_history.append(result.calyx_prediction)
        track.classification_frames.append(frame_index)
        track.classified = True
        return track

    def save_frame_classification(
        self,
        track_id: int,
        result: ClassificationResult,
        frame_index: int,
    ) -> TomatoTrack:
        """Persist the current result and append an observation for every matched frame."""

        track = self.get_track(track_id)
        track.health_probability = result.health_probability
        track.health_prediction = result.health_prediction
        track.health_confidence = result.health_confidence
        track.calyx_probability = result.calyx_probability
        track.calyx_prediction = result.calyx_prediction
        track.calyx_confidence = result.calyx_confidence
        if track.classification_frame is None:
            track.classification_frame = frame_index
        track.health_history.append(result.health_probability)
        track.calyx_history.append(result.calyx_probability)
        track.health_prediction_history.append(result.health_prediction)
        track.calyx_prediction_history.append(result.calyx_prediction)
        track.classification_frames.append(frame_index)
        track.classified = True
        return track

    def save_temporal_observation(self, track_id, r, f):
        t = self.get_track(track_id)
        t.health_history.append(r.health_probability)
        t.calyx_history.append(r.calyx_probability)
        t.health_prediction_history.append(r.health_prediction)
        t.calyx_prediction_history.append(r.calyx_prediction)
        t.classification_frames.append(f)
        p = sum(t.calyx_history) / len(t.calyx_history)
        t.calyx_probability = p
        t.calyx_prediction = int(p >= self._calyx_threshold)
        t.calyx_confidence = p if t.calyx_prediction else 1 - p
        return t

    def apply_temporal_aggregate(self, track_id, a: TemporalAggregate, f):
        t = self.get_track(track_id)
        t.health_positive_count = a.positive_count
        t.health_max_positive_run = a.max_positive_run
        t.health_top_k_mean = a.top_k_mean
        if a.finalized:
            t.classified = True
            t.health_decision_locked = True
            t.health_prediction = a.prediction
            t.health_probability = a.probability
            t.health_confidence = a.confidence
            t.health_decision_reason = a.reason
            t.health_decision_frame = f
            t.classification_frame = f
        return t

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

    def mark_passed_tracks(self, f, limit):
        for t in self._tracks.values():
            if (
                t.classified
                and not t.active_status
                and not t.passed
                and f - t.last_frame >= limit
            ):
                t.passed = True
                t.passed_frame = f

    def finalize_all(self, f=None) -> None:
        for track in self._tracks.values():
            track.active_status = False
            if track.classified and not track.passed:
                track.passed = True
                track.passed_frame = f

    def get_all_tracks(self) -> list[TomatoTrack]:
        return sorted(
            self._tracks.values(), key=lambda item: (item.first_frame, item.track_id)
        )
