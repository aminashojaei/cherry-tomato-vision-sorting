"""Configurable, nonphysical pixel-space size categories."""

from __future__ import annotations

from math import sqrt
from typing import Any

from core.schemas import TomatoTrack


class PixelSizeEstimator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def estimate(self, track: TomatoTrack) -> str | None:
        if not self.config.get("enabled", True) or not track.bbox.is_valid:
            return None

        width, height = track.bbox.width, track.bbox.height
        name = self.config["metric"]
        if name == "width":
            metric = width
        elif name == "height":
            metric = height
        elif name == "min_dimension":
            metric = min(width, height)
        elif name == "max_dimension":
            metric = max(width, height)
        elif name == "geometric_mean":
            metric = sqrt(width * height)
        elif name == "area":
            metric = width * height
        else:
            raise ValueError(f"Unsupported size metric: {name}")

        track.size_metric_px = metric
        limits = self.config["thresholds"]
        labels = self.config["labels"]
        if metric <= float(limits["small_max_px"]):
            return labels["small"]
        if metric <= float(limits["medium_max_px"]):
            return labels["medium"]
        return labels["large"]
