"""Discover and process any number of videos with isolated per-video state."""

from __future__ import annotations

import copy
import glob
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from core.config_loader import resolve_project_path

LOGGER = logging.getLogger(__name__)
DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
_GLOB_MARKERS = frozenset("*?[")
_UNSAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_video_extensions(values: list[str] | tuple[str, ...] | None) -> set[str]:
    extensions = values or DEFAULT_VIDEO_EXTENSIONS
    normalized = {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in extensions
        if value
    }
    if not normalized:
        raise ValueError("At least one supported video extension is required.")
    return normalized


def discover_video_files(
    sources: list[str],
    project_root: Path,
    extensions: list[str] | tuple[str, ...] | None = None,
    recursive: bool = False,
) -> list[Path]:
    """Resolve files, directories, and glob patterns into a stable de-duplicated list."""

    supported = normalize_video_extensions(extensions)
    discovered: list[Path] = []

    for source in sources:
        candidate = Path(source).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        candidate_text = str(candidate)

        if any(marker in candidate_text for marker in _GLOB_MARKERS):
            matches = [
                Path(value) for value in glob.glob(candidate_text, recursive=recursive)
            ]
            if not matches:
                raise FileNotFoundError(f"Input pattern matched no files: {source}")
        elif candidate.is_dir():
            iterator = candidate.rglob("*") if recursive else candidate.iterdir()
            matches = sorted(path for path in iterator if path.is_file())
        elif candidate.is_file():
            matches = [candidate]
        else:
            raise FileNotFoundError(f"Input video source does not exist: {source}")

        explicit_file = (
            len(matches) == 1 and matches[0] == candidate and candidate.is_file()
        )
        for path in matches:
            if path.suffix.lower() in supported:
                discovered.append(path.resolve())
            elif explicit_file:
                raise ValueError(f"Unsupported video extension for input: {path}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if not unique:
        raise FileNotFoundError(
            f"No supported videos found. Supported extensions: {sorted(supported)}"
        )
    return unique


def _safe_output_stem(path: Path) -> str:
    stem = _UNSAFE_STEM.sub("_", path.stem).strip("._-")
    return stem or "video"


class BatchVideoRunner:
    """Run a fresh VideoPipeline for every input and keep all outputs separate."""

    def __init__(
        self,
        config: dict[str, Any],
        videos: list[Path],
        *,
        output_root: str | Path | None = None,
        explicit_output: str | Path | None = None,
        continue_on_error: bool = True,
        pipeline_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        if not videos:
            raise ValueError("BatchVideoRunner requires at least one input video.")
        if explicit_output is not None and len(videos) != 1:
            raise ValueError("--output can only be used with exactly one input video.")

        self.config = config
        self.videos = videos
        self.root = Path(config["_project_root"])
        configured_root = config["paths"].get("output_dir", "outputs")
        self.output_root = resolve_project_path(
            str(output_root if output_root is not None else configured_root), self.root
        )
        self.explicit_output = (
            resolve_project_path(str(explicit_output), self.root)
            if explicit_output is not None
            else None
        )
        self.continue_on_error = continue_on_error
        if pipeline_factory is None:
            from pipeline.video_pipeline import VideoPipeline

            pipeline_factory = VideoPipeline
        self.pipeline_factory = pipeline_factory

    def _job_configs(self) -> list[tuple[Path, dict[str, Any]]]:
        jobs: list[tuple[Path, dict[str, Any]]] = []
        used_stems: dict[str, int] = {}
        batch_layout = len(self.videos) > 1 or self.explicit_output is None

        for video in self.videos:
            job_config = copy.deepcopy(self.config)
            job_config["paths"]["input_video"] = str(video)

            if self.explicit_output is not None:
                job_config["paths"]["output_video"] = str(self.explicit_output)
            elif batch_layout:
                base_stem = _safe_output_stem(video)
                occurrence = used_stems.get(base_stem, 0) + 1
                used_stems[base_stem] = occurrence
                stem = base_stem if occurrence == 1 else f"{base_stem}_{occurrence}"
                video_output_dir = self.output_root / stem
                job_config["paths"]["output_video"] = str(
                    video_output_dir / "annotated.mp4"
                )
                job_config["paths"]["output_json"] = str(
                    video_output_dir / "tomatoes.json"
                )
                job_config["paths"]["benchmark_json"] = str(
                    video_output_dir / "benchmark.json"
                )
                job_config["paths"]["size_measurements_csv"] = str(
                    video_output_dir / "size_measurements.csv"
                )

            jobs.append((video, job_config))
        return jobs

    def run(self) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for video, job_config in self._job_configs():
            LOGGER.info("Starting batch item: %s", video)
            try:
                pipeline_result = self.pipeline_factory(job_config).run()
                results.append(
                    {
                        "input_video": str(video),
                        "status": "succeeded",
                        **pipeline_result,
                    }
                )
            except Exception as exc:
                LOGGER.exception("Video failed: %s", video)
                results.append(
                    {
                        "input_video": str(video),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if not self.continue_on_error:
                    self._write_summary(results)
                    raise

        return self._write_summary(results)

    def _write_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        succeeded = sum(item["status"] == "succeeded" for item in results)
        summary = {
            "input_count": len(self.videos),
            "successful_videos": succeeded,
            "failed_videos": len(results) - succeeded,
            "results": results,
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        summary_path = self.output_root / "batch_summary.json"
        summary_path.write_text(
            json.dumps(
                summary, indent=int(self.config["logging"].get("json_indent", 2))
            ),
            encoding="utf-8",
        )
        summary["batch_summary_json"] = str(summary_path)
        return summary
