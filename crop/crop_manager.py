"""Extract padded crops exclusively from untouched original frames."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.exceptions import CropError
from core.schemas import BoundingBox


class CropManager:
    def __init__(self, config: dict[str, Any]) -> None:
        self.padding_ratio = float(config["padding_ratio"])
        self.allow_padding_clipping = bool(config["allow_padding_clipping"])

    def extract(self, original_frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        if original_frame.ndim < 2 or not bbox.is_valid:
            raise CropError("Cannot crop an invalid frame or bounding box.")

        height, width = original_frame.shape[:2]
        padded = bbox.expand_by_ratio(self.padding_ratio)
        if (
            not self.allow_padding_clipping
            and padded.outside_ratio(width, height) > 0.0
        ):
            raise CropError("Padded crop exceeds the frame and clipping is disabled.")

        clipped = padded.clip_to_frame(width, height)
        x1, y1 = math.floor(clipped.x1), math.floor(clipped.y1)
        x2, y2 = math.ceil(clipped.x2), math.ceil(clipped.y2)
        crop = original_frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise CropError(f"Bounding box {bbox.as_xyxy()} produced an empty crop.")
        return crop.copy()
