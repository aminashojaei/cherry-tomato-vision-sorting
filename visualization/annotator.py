"""Video overlays for tracks, decision zones, and aggregate counts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class FrameAnnotator:
    def __init__(self, config: dict[str, Any]) -> None:
        import cv2

        self.cv2 = cv2
        self.config = config
        self.font_scale = float(config["font_scale"])
        self.font_thickness = int(config["font_thickness"])

    def draw(self, frame, active_tracks, tracks=None, frame_index=0):
        if not self.config.get("enabled", True):
            return frame.copy()
        canvas = frame.copy()
        height, width = canvas.shape[:2]

        self._draw_zone(
            canvas,
            width,
            height,
            enabled=bool(self.config.get("draw_classification_zone", False)),
            zone_key="classification_zone",
            color_key="classification_zone_color",
        )
        self._draw_zone(
            canvas,
            width,
            height,
            enabled=bool(self.config.get("draw_size_measurement_zone", False)),
            zone_key="size_measurement_zone",
            color_key="size_measurement_zone_color",
        )

        for track in active_tracks:
            self._draw_track(canvas, track)

        return self._draw_sidebar(canvas, tracks or active_tracks)

    def _draw_zone(self, canvas, width, height, *, enabled, zone_key, color_key):
        if not enabled:
            return
        zone = self.config.get(zone_key)
        if not zone:
            return
        self.cv2.rectangle(
            canvas,
            (int(zone["x_min_ratio"] * width), int(zone["y_min_ratio"] * height)),
            (
                int(zone["x_max_ratio"] * width) - 1,
                int(zone["y_max_ratio"] * height) - 1,
            ),
            tuple(self.config[color_key]),
            2,
        )

    def _draw_track(self, canvas, track) -> None:
        x1, y1, x2, y2 = (int(round(value)) for value in track.bbox.as_xyxy())
        color = self._color(track)
        self.cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            color,
            int(self.config["line_thickness"]),
        )

        label = self._track_label(track)
        if label:
            (text_width, text_height), baseline = self.cv2.getTextSize(
                label,
                self.cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.font_thickness,
            )
            label_top = max(0, y1 - text_height - baseline - 8)
            self.cv2.rectangle(
                canvas,
                (x1, label_top),
                (x1 + text_width + 10, y1),
                color,
                -1,
            )
            self.cv2.putText(
                canvas,
                label,
                (x1 + 5, y1 - baseline - 4),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                self.font_thickness,
                self.cv2.LINE_AA,
            )

        if (
            self.config.get("draw_calyx", True)
            and track.classified
            and track.calyx_prediction == 1
        ):
            radius = int(self.config.get("calyx_star_radius_px", 40))
            self._draw_star(
                canvas,
                ((x1 + x2) // 2, max(radius + 2, y1 - radius - 6)),
                radius,
                color,
            )

    def _track_label(self, track) -> str:
        parts = []
        if self.config.get("draw_track_id", True):
            parts.append(f"ID {track.track_id}")
        if self.config.get("draw_detection_confidence", False):
            parts.append(f"det {track.detection_confidence:.2f}")
        if track.classified:
            if self.config.get("draw_health", True):
                parts.append("unhealthy" if track.health_prediction == 1 else "healthy")
            if self.config.get("draw_calyx", True):
                parts.append("calyx" if track.calyx_prediction == 1 else "no-calyx")
        elif self.config.get("show_pending_status", True):
            parts.append(str(self.config.get("pending_label", "PENDING")))
        if self.config.get("draw_size", True) and track.size_finalized:
            parts.append(f"{track.size_class} {track.size_metric_px:.0f}px")
        return " | ".join(parts)

    def _draw_star(self, canvas, center, radius, color) -> None:
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            point_radius = radius if index % 2 == 0 else radius * 0.42
            points.append(
                (
                    int(center[0] + math.cos(angle) * point_radius),
                    int(center[1] + math.sin(angle) * point_radius),
                )
            )
        polygon = np.asarray(points, np.int32)
        self.cv2.fillPoly(canvas, [polygon], color)
        self.cv2.polylines(canvas, [polygon], True, (255, 255, 255), 2)

    def _draw_sidebar(self, canvas, tracks):
        sidebar_config = self.config["sidebar"]
        if not sidebar_config.get("enabled", True):
            return canvas

        width = int(sidebar_config["width_px"])
        panel = np.full((canvas.shape[0], width, 3), (28, 31, 38), np.uint8)
        classified = [track for track in tracks if track.classified]
        passed = [track for track in classified if track.passed]
        self.cv2.putText(
            panel,
            str(sidebar_config.get("title", "TOMATO SORTING")),
            (20, 38),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (245, 245, 245),
            2,
            self.cv2.LINE_AA,
        )

        counters = [
            ("CLASSIFIED", len(classified)),
            ("PASSED", len(passed)),
            ("HEALTHY", sum(track.health_prediction == 0 for track in passed)),
            ("DEFECT", sum(track.health_prediction == 1 for track in passed)),
        ]
        column_width = max(1, width // 2)
        for index, (name, value) in enumerate(counters):
            x = 15 + (index % 2) * column_width
            y = 58 + (index // 2) * 58
            self.cv2.putText(
                panel,
                f"{name}: {value}",
                (x, y + 28),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
                self.cv2.LINE_AA,
            )

        y = 190
        max_rows = int(sidebar_config.get("max_rows", 14))
        for track in sorted(classified, key=lambda item: item.track_id, reverse=True)[
            :max_rows
        ]:
            status = "DEFECT" if track.health_prediction == 1 else "HEALTHY"
            size = (
                f"{track.size_class[0]} {track.size_metric_px:.0f}px"
                if track.size_finalized
                else "--"
            )
            self.cv2.putText(
                panel,
                f"{track.track_id:<4} {status:<12} {size}",
                (15, y),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 238, 244),
                1,
                self.cv2.LINE_AA,
            )
            y += 28
        return np.concatenate([canvas, panel], axis=1)

    def _color(self, track):
        key = (
            "pending"
            if not track.classified
            else "unhealthy" if track.health_prediction == 1 else "healthy"
        )
        return tuple(self.config["colors"][key])
