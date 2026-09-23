"""Fail-fast OpenCV video input and output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.exceptions import VideoIOError


class VideoReader:
    def __init__(self, path: Path, fallback_fps: float | None) -> None:
        import cv2

        if not path.is_file():
            raise VideoIOError(f"Input video does not exist: {path}")
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise VideoIOError(f"Unable to open input video: {path}")

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if self.width <= 0 or self.height <= 0:
            self.capture.release()
            raise VideoIOError(
                f"Input video has invalid frame dimensions: {self.width}x{self.height}"
            )
        if fps <= 0 and fallback_fps is None:
            self.capture.release()
            raise VideoIOError(
                "Input video FPS is unavailable; set video.fallback_fps explicitly."
            )
        self.fps = fps if fps > 0 else float(fallback_fps)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        success, frame = self.capture.read()
        return frame if success else None

    def release(self) -> None:
        self.capture.release()


class VideoWriter:
    def __init__(
        self, path: Path, width: int, height: int, fps: float, codec: str
    ) -> None:
        import cv2

        if len(codec) != 4:
            raise VideoIOError(
                f"Video codec must contain exactly four characters: {codec!r}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise VideoIOError(
                f"Unable to initialize output video writer with codec {codec}: {path}"
            )

    def write(self, frame) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
