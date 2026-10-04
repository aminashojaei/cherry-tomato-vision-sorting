"""Per-tomato JSON serialization without fabricated classifications."""

from __future__ import annotations

import json
import csv
import math
from pathlib import Path
from typing import Any

from core.schemas import TomatoTrack


class TomatoReportWriter:
    def __init__(self, config: dict[str, Any], size_labels: dict[str, str] | None = None) -> None:
        self.config = config
        self.size_labels = size_labels or {"small": "Small", "medium": "Medium", "large": "Large"}

    def serialize_track(self, track: TomatoTrack) -> dict[str, Any]:
        bbox_by_frame = {item.frame_index: item for item in track.bbox_history}
        classification_history = []
        for (
            frame,
            health_probability,
            health_prediction,
            calyx_probability,
            calyx_prediction,
        ) in zip(
            track.classification_frames,
            track.health_history,
            track.health_prediction_history,
            track.calyx_history,
            track.calyx_prediction_history,
        ):
            observation = bbox_by_frame.get(frame)
            history_item: dict[str, Any] = {
                "frame": frame,
                "health_probability": health_probability,
                "health_prediction": health_prediction,
                "health_class": "unhealthy" if health_prediction == 1 else "healthy",
                "calyx_probability": calyx_probability,
                "calyx_prediction": calyx_prediction,
                "calyx_class": "present" if calyx_prediction == 1 else "absent",
            }
            if observation is not None:
                bbox = observation.bbox
                history_item.update(
                    {
                        "bbox_xyxy": list(bbox.as_xyxy()),
                        "bbox_width_px": bbox.width,
                        "bbox_height_px": bbox.height,
                        "bbox_area_px": bbox.area,
                        "bbox_geometric_mean_px": math.sqrt(max(0.0, bbox.area)),
                        "detection_confidence": observation.detection_confidence,
                    }
                )
            classification_history.append(history_item)
        health_flips = sum(
            previous != current
            for previous, current in zip(
                track.health_prediction_history,
                track.health_prediction_history[1:],
            )
        )
        record: dict[str, Any] = {
            "tomato_id": track.track_id,
            "first_frame": track.first_frame,
            "last_frame": track.last_frame,
            "classified": track.classified,
            "health_class": self._health_label(track.health_prediction),
            "health_probability": track.health_probability,
            "health_confidence": track.health_confidence,
            "calyx_class": self._calyx_label(track.calyx_prediction),
            "calyx_probability": track.calyx_probability,
            "calyx_confidence": track.calyx_confidence,
            "size_class": track.size_class,
            "size_metric_px": track.size_metric_px,
            "final_class": self._final_class(track),
            "classification_frame": track.classification_frame,
            "health_history": list(track.health_history),
            "calyx_history": list(track.calyx_history),
            "classification_frames": list(track.classification_frames),
            "health_prediction_history": list(track.health_prediction_history),
            "calyx_prediction_history": list(track.calyx_prediction_history),
            "classification_history": classification_history,
            "health_prediction_flip_count": health_flips,
            "classification_observation_count": len(classification_history),
            "health_decision_locked": track.health_decision_locked,
            "health_decision_reason": track.health_decision_reason,
            "health_decision_frame": track.health_decision_frame,
            "health_positive_count": track.health_positive_count,
            "health_max_positive_run": track.health_max_positive_run,
            "health_top_k_mean": track.health_top_k_mean,
            "zone_entry_frame": track.zone_entry_frame,
            "last_zone_frame": track.last_zone_frame,
            "size_history_px": list(track.size_history),
            "size_measurement_frames": list(track.size_frames),
            "size_finalized": track.size_finalized,
            "size_finalization_frame": track.size_finalization_frame,
            "passed": track.passed,
            "passed_frame": track.passed_frame,
        }
        if self.config.get("include_bbox_history", True):
            record["bbox_history"] = [
                {
                    "frame": observation.frame_index,
                    "bbox_xyxy": list(observation.bbox.as_xyxy()),
                    "detection_confidence": observation.detection_confidence,
                }
                for observation in track.bbox_history
            ]
        if self.config.get("include_quality_gate_failures", True):
            record["quality_gate_failures"] = list(track.quality_gate_failures)
        return record

    def write(
        self,
        path: Path,
        tracks: list[TomatoTrack],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        records = [self.serialize_track(track) for track in tracks]
        classified = [record for record in records if record["classified"]]
        report = {
            "metadata": metadata,
            "summary": {
                "total_tracks": len(records),
                "classified_tracks": len(classified),
                "unclassified_tracks": len(records) - len(classified),
                "healthy_tracks": sum(
                    record["health_class"] == "healthy" for record in classified
                ),
                "unhealthy_tracks": sum(
                    record["health_class"] == "unhealthy" for record in classified
                ),
                "calyx_present_tracks": sum(
                    record["calyx_class"] == "present" for record in classified
                ),
                "calyx_absent_tracks": sum(
                    record["calyx_class"] == "absent" for record in classified
                ),
                "classification_observations": sum(
                    record["classification_observation_count"] for record in records
                ),
                "health_prediction_flips": sum(
                    record["health_prediction_flip_count"] for record in records
                ),
                "size_measurements_finalized": sum(
                    r["size_finalized"] for r in records
                ),
                "small_tracks": sum(r["size_class"] == self.size_labels["small"] for r in records),
                "medium_tracks": sum(r["size_class"] == self.size_labels["medium"] for r in records),
                "large_tracks": sum(r["size_class"] == self.size_labels["large"] for r in records),
                "passed_tracks": sum(r["passed"] for r in classified),
            },
            "tomatoes": records,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                report,
                indent=int(self.config.get("json_indent", 2)),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return report

    @staticmethod
    def _health_label(prediction: int | None) -> str | None:
        return (
            None
            if prediction is None
            else ("unhealthy" if prediction == 1 else "healthy")
        )

    @staticmethod
    def _calyx_label(prediction: int | None) -> str | None:
        return (
            None if prediction is None else ("present" if prediction == 1 else "absent")
        )

    @classmethod
    def _final_class(cls, track: TomatoTrack) -> str | None:
        health = cls._health_label(track.health_prediction)
        calyx = cls._calyx_label(track.calyx_prediction)
        if track.size_class is None or health is None or calyx is None:
            return None
        return f"{track.size_class.lower()}_{health}_{calyx}"


class SizeMeasurementWriter:
    def write(self, path, tracks):
        cols = [
            "tomato_id",
            "finalized",
            "size_class",
            "short_side_px",
            "sample_count",
            "measurement_frame",
            "health_class",
            "calyx_class",
            "final_class",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for t in tracks:
                w.writerow(
                    {
                        "tomato_id": t.track_id,
                        "finalized": t.size_finalized,
                        "size_class": t.size_class or "",
                        "short_side_px": t.size_metric_px if t.size_finalized else "",
                        "sample_count": len(t.size_history),
                        "measurement_frame": (
                            t.size_finalization_frame
                            if t.size_finalization_frame is not None
                            else ""
                        ),
                        "health_class": TomatoReportWriter._health_label(
                            t.health_prediction
                        )
                        or "",
                        "calyx_class": TomatoReportWriter._calyx_label(
                            t.calyx_prediction
                        )
                        or "",
                        "final_class": TomatoReportWriter._final_class(t) or "",
                    }
                )
