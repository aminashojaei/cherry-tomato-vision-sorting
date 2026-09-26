"""YOLO11n adapter returning original-frame tomato detections only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.exceptions import ModelLoadError
from core.schemas import BoundingBox, Detection


class YOLODetector:
    def __init__(self, config: dict[str, Any], weights_path: Path, device: str) -> None:
        if not weights_path.is_file():
            raise ModelLoadError(
                f"Tomato detector checkpoint not found: {weights_path}. "
                "Confirm that the detector weights path in the configuration points to an existing checkpoint."
            )

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(weights_path))
        except Exception as exc:
            raise ModelLoadError(
                f"Unable to load YOLO11n detector {weights_path}: {exc}"
            ) from exc

        self.config = config
        self.device = device
        self.class_id = int(config["tomato_class_id"])

    def detect(self, frame: np.ndarray) -> list[Detection]:
        predictions = self.model.predict(
            source=frame,
            conf=float(self.config["confidence_threshold"]),
            iou=float(self.config["iou_threshold"]),
            imgsz=int(self.config["input_size"]),
            classes=[self.class_id],
            max_det=int(self.config["max_detections"]),
            device=self.device,
            verbose=False,
        )
        if not predictions or predictions[0].boxes is None:
            return []

        boxes = predictions[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        detections: list[Detection] = []
        for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids):
            if int(class_id) != self.class_id:
                continue
            bbox = BoundingBox(*(float(value) for value in coordinates))
            if bbox.is_valid:
                detections.append(Detection(bbox, float(confidence), int(class_id)))
        return detections
