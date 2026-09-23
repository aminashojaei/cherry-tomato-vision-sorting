"""Configuration loading, normalization, and fail-fast validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from core.exceptions import ConfigurationError

LOGGER = logging.getLogger(__name__)


def resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Unable to read configuration {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError("Configuration must contain a YAML mapping.")

    data["_config_path"] = str(path)
    data["_project_root"] = str(path.parent.parent)
    validate_config(data)
    return data


def validate_config(config: dict[str, Any]) -> None:
    required = (
        "runtime",
        "paths",
        "video",
        "detector",
        "tracker",
        "classifier",
        "crop",
        "quality_gate",
        "size_estimation",
        "visualization",
        "logging",
        "benchmark",
    )
    missing = [section for section in required if section not in config]
    if missing:
        raise ConfigurationError(
            f"Missing configuration sections: {', '.join(missing)}"
        )

    detector = config["detector"]
    tracker = config["tracker"]
    classifier = config["classifier"]

    probability_values = {
        "detector.confidence_threshold": detector["confidence_threshold"],
        "detector.iou_threshold": detector["iou_threshold"],
        "tracker.high_threshold": tracker["high_threshold"],
        "tracker.low_threshold": tracker["low_threshold"],
        "tracker.new_track_threshold": tracker["new_track_threshold"],
        "tracker.match_threshold": tracker["match_threshold"],
        "classifier.health.threshold": classifier["health"]["threshold"],
        "classifier.calyx.threshold": classifier["calyx"]["threshold"],
        "quality_gate.min_detection_confidence": config["quality_gate"][
            "min_detection_confidence"
        ],
        "quality_gate.max_bbox_outside_ratio": config["quality_gate"][
            "max_bbox_outside_ratio"
        ],
    }
    for name, value in probability_values.items():
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ConfigurationError(
                f"{name} must be between 0 and 1; received {value!r}."
            )

    if tracker["low_threshold"] > tracker["high_threshold"]:
        raise ConfigurationError(
            "tracker.low_threshold cannot exceed tracker.high_threshold."
        )

    if detector["confidence_threshold"] > tracker["low_threshold"]:
        LOGGER.warning(
            "Detector confidence threshold %.3f exceeds ByteTrack low threshold %.3f; "
            "some low-confidence association candidates will never reach the tracker.",
            detector["confidence_threshold"],
            tracker["low_threshold"],
        )

    if detector["input_size"] <= 0 or classifier["input_size"] <= 0:
        raise ConfigurationError(
            "Detector and classifier input sizes must be positive."
        )

    if not classifier.get("normalize_inside_model", False):
        raise ConfigurationError(
            "The supplied classifier checkpoint requires ImageNet normalization inside the model."
        )

    if config["crop"]["padding_ratio"] < 0:
        raise ConfigurationError("crop.padding_ratio cannot be negative.")

    batch = config.get("batch", {})
    extensions = batch.get("video_extensions", [])
    if not isinstance(extensions, list) or not extensions:
        raise ConfigurationError("batch.video_extensions must be a non-empty list.")
    if not all(isinstance(value, str) and value.strip() for value in extensions):
        raise ConfigurationError(
            "Every batch.video_extensions value must be a non-empty string."
        )

    limits = config["size_estimation"]["thresholds"]
    if float(limits["small_max_px"]) != 150 or float(limits["medium_max_px"]) != 250:
        raise ConfigurationError("Size thresholds must be 150/250.")

    if config["size_estimation"]["metric"] != "min_dimension":
        raise ConfigurationError(
            "The temporal size estimator requires size_estimation.metric=min_dimension."
        )

    if int(config["size_estimation"]["required_samples"]) != 1:
        raise ConfigurationError(
            "The current size policy requires exactly one valid sample."
        )
    if "tracking_zone" not in config["track_management"]:
        raise ConfigurationError("track_management.tracking_zone is required.")
    if (
        config["classification_policy"]["mode"]
        != "lower_zone_asymmetric_temporal_voting"
    ):
        raise ConfigurationError("Unsupported classification policy mode.")
