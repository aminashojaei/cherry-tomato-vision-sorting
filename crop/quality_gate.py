"""Explainable, configurable pre-classification quality checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.schemas import QualityGateResult, TomatoTrack
from crop.crop_manager import CropManager


class CropQualityGate:
    def __init__(self, config: dict[str, Any], crop_manager: CropManager) -> None:
        self.config = config
        self.crop_manager = crop_manager

    def evaluate(
        self, original_frame: np.ndarray, track: TomatoTrack
    ) -> QualityGateResult:
        if not self.config.get("enabled", True):
            return QualityGateResult(True, [], {})

        bbox = track.bbox
        height, width = original_frame.shape[:2]
        reasons: list[str] = []
        metrics: dict[str, float] = {
            "detection_confidence": float(track.detection_confidence),
            "bbox_width_px": float(bbox.width),
            "bbox_height_px": float(bbox.height),
            "bbox_area_px": float(bbox.area),
        }

        if not bbox.is_valid:
            reasons.append("invalid_bbox")
            return QualityGateResult(False, reasons, metrics)

        outside_ratio = bbox.outside_ratio(width, height)
        metrics["bbox_outside_ratio"] = outside_ratio

        if track.detection_confidence < float(self.config["min_detection_confidence"]):
            reasons.append("detection_confidence_too_low")
        if bbox.width < float(self.config["min_bbox_width_px"]):
            reasons.append("bbox_width_too_small")
        if bbox.height < float(self.config["min_bbox_height_px"]):
            reasons.append("bbox_height_too_small")
        if bbox.area < float(self.config["min_bbox_area_px"]):
            reasons.append("bbox_area_too_small")
        if self.config["reject_if_bbox_outside_frame"] and outside_ratio > float(
            self.config["max_bbox_outside_ratio"]
        ):
            reasons.append("bbox_exceeds_frame")

        if reasons:
            return QualityGateResult(False, reasons, metrics)

        blur_config = self.config.get("blur", {})
        if blur_config.get("enabled", False):
            import cv2

            try:
                crop = self.crop_manager.extract(original_frame, bbox)
            except Exception:
                reasons.append("crop_extraction_failed")
                return QualityGateResult(False, reasons, metrics)

            grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
            metrics["blur_score"] = blur_score
            if blur_score < float(blur_config["min_score"]):
                reasons.append("crop_too_blurry")

        return QualityGateResult(not reasons, reasons, metrics)
