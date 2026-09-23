"""CPU/GPU-aware stage timing and percentile benchmark summaries."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class PerformanceMonitor:
    def __init__(
        self,
        config: dict[str, Any],
        device: str,
        synchronize_cuda: bool,
        source_fps: float | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.synchronize_cuda = synchronize_cuda and device.startswith("cuda")
        self.records: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.classifier_invocations = 0
        self.frame_count = 0
        self.started_at = time.perf_counter()
        self.source_fps = float(source_fps) if source_fps else None

    def _synchronize(self) -> None:
        if self.synchronize_cuda:
            import torch

            torch.cuda.synchronize()

    @contextmanager
    def measure(self, stage: str, frame_index: int) -> Iterator[dict[str, float]]:
        self._synchronize()
        started = time.perf_counter()
        measurement: dict[str, float] = {}
        try:
            yield measurement
        finally:
            self._synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            measurement["elapsed_ms"] = elapsed_ms
            self.records[stage].append((frame_index, elapsed_ms))

    def record_frame(self) -> None:
        self.frame_count += 1

    def summary(self) -> dict[str, Any]:
        warmup = int(self.config.get("warmup_frames", 0))
        percentiles = [
            int(value) for value in self.config.get("report_percentiles", [])
        ]
        latency: dict[str, dict[str, float | int]] = {}
        for stage, observations in self.records.items():
            values = np.asarray(
                [
                    value
                    for index, value in observations
                    if warmup <= index < self.frame_count
                ],
                dtype=float,
            )
            if values.size == 0:
                continue
            details: dict[str, float | int] = {
                "count": int(values.size),
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for percentile in percentiles:
                details[f"p{percentile}"] = float(np.percentile(values, percentile))
            latency[stage] = details

        wall_seconds = max(time.perf_counter() - self.started_at, 1e-12)
        total_measurements = [
            item
            for item in self.records.get("total_frame_ms", [])
            if item[0] < self.frame_count
        ]
        total_ms = sum(value for _, value in total_measurements)
        fps = self.frame_count / wall_seconds
        ratio = fps / self.source_fps if self.source_fps else None
        return {
            "processed_frames": self.frame_count,
            "warmup_frames_excluded": warmup,
            "classifier_invocations": self.classifier_invocations,
            "device": self.device,
            "cuda_timing_synchronized": self.synchronize_cuda,
            "wall_time_seconds": wall_seconds,
            "processing_fps": fps,
            "input_video_fps": self.source_fps,
            "frame_budget_ms": 1000 / self.source_fps if self.source_fps else None,
            "processing_ms_per_frame": 1000 / max(fps, 1e-12),
            "realtime_ratio": ratio,
            "slowdown_factor": 1 / ratio if ratio else None,
            "realtime_by_average": ratio >= 1 if ratio is not None else None,
            "measured_frame_fps": self.frame_count / max(total_ms / 1000.0, 1e-12),
            "latency_ms": latency,
        }

    def write_json(self, path: Path, indent: int = 2) -> dict[str, Any]:
        summary = self.summary()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, indent=indent, ensure_ascii=False), encoding="utf-8"
        )
        return summary
