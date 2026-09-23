"""Command-line entry point for the Colab-compatible video pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from core.config_loader import load_config, validate_config
from core.exceptions import PipelineError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cherry tomato sorting video pipeline."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to project YAML configuration.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        action="append",
        help="One or more video files, directories, or glob patterns. May be repeated.",
    )
    parser.add_argument(
        "--output", help="Override the configured annotated output video path."
    )
    parser.add_argument(
        "--output-dir",
        help="Root directory for separate per-video output folders.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover videos inside input directories and ** glob patterns.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch at the first failed video instead of continuing.",
    )
    parser.add_argument(
        "--detector-weights",
        help="Override the trained YOLO11n detector checkpoint path.",
    )
    parser.add_argument(
        "--classifier-weights",
        help="Override the trained multi-head classifier checkpoint path.",
    )
    parser.add_argument(
        "--device", help="Override runtime device: auto, cpu, cuda, or cuda:0."
    )
    parser.add_argument(
        "--max-frames", type=int, help="Process only the first N frames."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        if args.detector_weights:
            config["detector"]["weights_path"] = args.detector_weights
        if args.classifier_weights:
            config["classifier"]["weights_path"] = args.classifier_weights
        if args.device:
            config["runtime"]["device"] = args.device
        if args.max_frames is not None:
            config["video"]["max_frames"] = args.max_frames
        validate_config(config)

        logging.basicConfig(
            level=getattr(
                logging,
                str(config["logging"].get("level", "INFO")).upper(),
                logging.INFO,
            ),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        from pipeline.batch_pipeline import BatchVideoRunner, discover_video_files

        if args.input:
            sources = [value for group in args.input for value in group]
        else:
            sources = [
                str(config["paths"].get("input_videos", config["paths"]["input_video"]))
            ]
        batch_config = config.get("batch", {})
        videos = discover_video_files(
            sources,
            Path(config["_project_root"]),
            extensions=batch_config.get("video_extensions"),
            recursive=bool(args.recursive or batch_config.get("recursive", False)),
        )
        result = BatchVideoRunner(
            config,
            videos,
            output_root=args.output_dir,
            explicit_output=args.output,
            continue_on_error=not bool(args.stop_on_error),
        ).run()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if result["failed_videos"] else 0
    except (PipelineError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
