"""Configuration loading, normalization, and fail-fast validation."""

from __future__ import annotations

import logging
import math
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
        "track_management",
        "classifier",
        "classification_policy",
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
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
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

    size = config["size_estimation"]
    limits = size["thresholds"]
    small, medium = float(limits["small_max_px"]), float(limits["medium_max_px"])
    if not math.isfinite(small) or not math.isfinite(medium) or not 0 < small < medium:
        raise ConfigurationError(
            "Size thresholds must satisfy 0 < small_max_px < medium_max_px."
        )

    if config["size_estimation"]["metric"] != "min_dimension":
        raise ConfigurationError(
            "The temporal size estimator requires size_estimation.metric=min_dimension."
        )

    if type(size["required_samples"]) is not int or size["required_samples"] != 1:
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

    supported_types = (
        ("detector.type", detector["type"], "yolo11n"),
        ("tracker.type", tracker["type"], "bytetrack"),
        ("classifier.type", classifier["type"], "shufflenet_v2_x1_0_multihead"),
        ("size_estimation.type", size["type"], "temporal_bbox_short_side"),
    )
    for name, actual, expected in supported_types:
        if actual != expected:
            raise ConfigurationError(
                f"{name} must be {expected!r}; received {actual!r}."
            )

    policy = config["classification_policy"]
    for name in ("sample_stride_frames", "min_samples_for_healthy"):
        if type(policy[name]) is not int or policy[name] <= 0:
            raise ConfigurationError(
                f"classification_policy.{name} must be a positive integer."
            )
    evidence = policy["unhealthy_evidence"]
    for name in ("consecutive_positive_samples", "total_positive_samples", "top_k"):
        if type(evidence[name]) is not int or evidence[name] <= 0:
            raise ConfigurationError(
                f"classification_policy.unhealthy_evidence.{name} must be a positive integer."
            )

    for name, zone in (
        (
            "track_management.tracking_zone",
            config["track_management"]["tracking_zone"],
        ),
        ("classification_policy.zone", policy["zone"]),
        ("size_estimation.zone", size["zone"]),
    ):
        try:
            x_min, x_max = float(zone["x_min_ratio"]), float(zone["x_max_ratio"])
            y_min, y_max = float(zone["y_min_ratio"]), float(zone["y_max_ratio"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"{name} must contain numeric min/max ratios.") from exc
        if not all(math.isfinite(v) for v in (x_min, x_max, y_min, y_max)) or not (
            0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1
        ):
            raise ConfigurationError(
                f"{name} must satisfy 0 <= min < max <= 1 on both axes."
            )

    video = config["video"]
    if video.get("fallback_fps") is not None and (
        not math.isfinite(float(video["fallback_fps"]))
        or float(video["fallback_fps"]) <= 0
    ):
        raise ConfigurationError("video.fallback_fps must be positive and finite.")
    if video.get("max_frames") is not None and (
        type(video["max_frames"]) is not int or video["max_frames"] <= 0
    ):
        raise ConfigurationError("video.max_frames must be a positive integer.")
    if type(size["frame_edge_margin_px"]) is not int or size["frame_edge_margin_px"] < 0:
        raise ConfigurationError("size_estimation.frame_edge_margin_px must be nonnegative.")
    if type(tracker["lost_track_buffer"]) is not int or tracker["lost_track_buffer"] < 0:
        raise ConfigurationError("tracker.lost_track_buffer must be nonnegative.")
    device = config["runtime"]["device"]
    if device not in ("auto", "cpu", "mps", "cuda") and not (
        isinstance(device, str) and device.startswith("cuda:") and device[5:].isdigit()
    ):
        raise ConfigurationError(f"Unsupported runtime.device: {device!r}.")
