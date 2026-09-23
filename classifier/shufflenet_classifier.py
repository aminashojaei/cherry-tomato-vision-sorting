"""Checkpoint-compatible ShuffleNetV2 multi-head inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.exceptions import ModelLoadError
from core.schemas import ClassificationResult


class ShuffleNetMultiHeadClassifier:
    def __init__(
        self,
        config: dict[str, Any],
        weights_path: Path,
        thresholds_path: Path,
        device: str,
    ) -> None:
        import torch
        from classifier.model import ShuffleNetMultiHeadModel

        if not weights_path.is_file():
            raise ModelLoadError(
                f"Classifier checkpoint does not exist: {weights_path}"
            )
        if not thresholds_path.is_file():
            raise ModelLoadError(
                f"Classifier threshold metadata does not exist: {thresholds_path}"
            )

        self._torch = torch
        self.device = device
        self.input_size = int(config["input_size"])
        self.health_threshold = float(config["health"]["threshold"])
        self.calyx_threshold = float(config["calyx"]["threshold"])
        self._validate_thresholds(thresholds_path)

        self.model = ShuffleNetMultiHeadModel(
            feature_dim=int(config["feature_dim"]),
            hidden_dim=int(config["head_hidden_dim"]),
            dropout=float(config["dropout"]),
            mean=config["normalization"]["mean"],
            std=config["normalization"]["std"],
        )

        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                raise ValueError(
                    "Expected a dictionary containing the trained state_dict."
                )
            model_key = checkpoint.get("model_key")
            if model_key not in (None, "shufflenet_v2_x1_0"):
                raise ValueError(f"Unexpected classifier model_key: {model_key!r}")
            checkpoint_health = checkpoint.get("health_threshold")
            checkpoint_calyx = checkpoint.get("calyx_threshold")
            if (
                checkpoint_health is not None
                and abs(float(checkpoint_health) - self.health_threshold) > 1e-9
            ):
                raise ValueError(
                    f"Checkpoint health threshold {checkpoint_health} does not match "
                    f"configured threshold {self.health_threshold}."
                )
            if (
                checkpoint_calyx is not None
                and abs(float(checkpoint_calyx) - self.calyx_threshold) > 1e-9
            ):
                raise ValueError(
                    f"Checkpoint calyx threshold {checkpoint_calyx} does not match "
                    f"configured threshold {self.calyx_threshold}."
                )
            self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        except Exception as exc:
            raise ModelLoadError(
                f"Classifier checkpoint does not match the required ShuffleNetV2 two-head "
                f"architecture: {exc}"
            ) from exc

        self.model.to(device)
        self.model.eval()

    def _validate_thresholds(self, thresholds_path: Path) -> None:
        try:
            metadata = json.loads(thresholds_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadError(
                f"Cannot read classifier thresholds {thresholds_path}: {exc}"
            ) from exc

        expected_health = float(metadata["health_threshold"])
        expected_calyx = float(metadata["calyx_threshold"])
        if abs(self.health_threshold - expected_health) > 1e-9:
            raise ModelLoadError(
                f"Configured health threshold {self.health_threshold} does not match the trained "
                f"validation-selected threshold {expected_health}."
            )
        if abs(self.calyx_threshold - expected_calyx) > 1e-9:
            raise ModelLoadError(
                f"Configured calyx threshold {self.calyx_threshold} does not match the trained "
                f"validation-selected threshold {expected_calyx}."
            )

    def classify(self, crop: np.ndarray) -> ClassificationResult:
        from classifier.preprocessing import preprocess_crop

        image = preprocess_crop(crop, self.input_size, self.device)
        with self._torch.inference_mode():
            outputs = self.model(image)
            health_probability = float(
                self._torch.sigmoid(outputs["health_logits"])[0].item()
            )
            calyx_probability = float(
                self._torch.sigmoid(outputs["calyx_logits"])[0].item()
            )

        health_prediction = int(health_probability >= self.health_threshold)
        calyx_prediction = int(calyx_probability >= self.calyx_threshold)

        return ClassificationResult(
            health_probability=health_probability,
            health_prediction=health_prediction,
            health_confidence=(
                health_probability if health_prediction else 1.0 - health_probability
            ),
            calyx_probability=calyx_probability,
            calyx_prediction=calyx_prediction,
            calyx_confidence=(
                calyx_probability if calyx_prediction else 1.0 - calyx_probability
            ),
        )
