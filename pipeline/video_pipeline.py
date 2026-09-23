"""End-to-end processing for a single video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from classifier.shufflenet_classifier import ShuffleNetMultiHeadClassifier
from classifier.temporal_policy import AsymmetricTemporalPolicy
from core.config_loader import resolve_project_path
from core.exceptions import CropError
from crop.crop_manager import CropManager
from crop.quality_gate import CropQualityGate
from detector.yolo_detector import YOLODetector
from geometry.temporal_size_estimator import TemporalShortSideMeasurer
from tracker.bytetrack_tracker import ByteTrackTracker
from tracker.track_manager import TrackManager
from utils.device import select_device
from utils.logger import SizeMeasurementWriter, TomatoReportWriter
from utils.timing import PerformanceMonitor
from utils.video_io import VideoReader, VideoWriter
from visualization.annotator import FrameAnnotator


def filter_detections_to_zone(
    detections: list[Any],
    frame_shape: tuple[int, ...],
    zone: dict[str, float],
) -> list[Any]:
    height, width = frame_shape[:2]
    return [
        detection
        for detection in detections
        if float(zone["x_min_ratio"])
        <= (detection.bbox.x1 + detection.bbox.x2) / (2 * width)
        <= float(zone["x_max_ratio"])
        and float(zone["y_min_ratio"])
        <= (detection.bbox.y1 + detection.bbox.y2) / (2 * height)
        <= float(zone["y_max_ratio"])
    ]


class VideoPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.root = Path(config["_project_root"])
        self.device = select_device(str(config["runtime"]["device"]))
        paths = config["paths"]

        self.input_path = resolve_project_path(paths["input_video"], self.root)
        self.output_path = resolve_project_path(paths["output_video"], self.root)
        self.json_path = resolve_project_path(paths["output_json"], self.root)
        self.benchmark_path = resolve_project_path(paths["benchmark_json"], self.root)
        self.size_csv_path = resolve_project_path(
            paths["size_measurements_csv"], self.root
        )
        self.reader = VideoReader(self.input_path, config["video"].get("fallback_fps"))
        self.writer = None

        try:
            sidebar = config["visualization"]["sidebar"]
            sidebar_width = (
                int(sidebar["width_px"])
                if config["visualization"].get("enabled", True)
                and sidebar.get("enabled", True)
                else 0
            )
            self.writer = VideoWriter(
                self.output_path,
                self.reader.width + sidebar_width,
                self.reader.height,
                self.reader.fps,
                str(config["video"]["codec"]),
            )
            detector_weights = resolve_project_path(
                config["detector"]["weights_path"], self.root
            )
            classifier_weights = resolve_project_path(
                config["classifier"]["weights_path"], self.root
            )
            thresholds_path = resolve_project_path(
                config["classifier"]["thresholds_path"], self.root
            )

            self.detector = YOLODetector(
                config["detector"], detector_weights, self.device
            )
            self.tracker = ByteTrackTracker(config["tracker"], self.reader.fps)
            self.track_manager = TrackManager(
                bool(config["track_management"].get("save_bbox_history", True)),
                float(config["classifier"]["calyx"]["threshold"]),
            )
            self.crop_manager = CropManager(config["crop"])
            self.quality_gate = CropQualityGate(
                config["quality_gate"], self.crop_manager
            )
            self.classifier = ShuffleNetMultiHeadClassifier(
                config["classifier"], classifier_weights, thresholds_path, self.device
            )
            self.temporal_policy = AsymmetricTemporalPolicy(
                config["classification_policy"]
            )
            self.size_estimator = TemporalShortSideMeasurer(config["size_estimation"])
            self.annotator = FrameAnnotator(config["visualization"])
            self.report_writer = TomatoReportWriter(config["logging"])
            self.size_writer = SizeMeasurementWriter()
            self.monitor = PerformanceMonitor(
                config["benchmark"],
                self.device,
                bool(config["runtime"].get("synchronize_cuda_for_timing", False)),
                self.reader.fps,
            )
        except Exception:
            self.reader.release()
            if self.writer:
                self.writer.release()
            raise

    def run(self) -> dict[str, Any]:
        frame_index = 0
        tracking_zone = self.config["track_management"]["tracking_zone"]
        max_frames = self.config["video"].get("max_frames")
        lost_track_buffer = int(self.config["tracker"]["lost_track_buffer"])

        try:
            while max_frames is None or frame_index < int(max_frames):
                with self.monitor.measure("total_frame_ms", frame_index):
                    with self.monitor.measure("frame_reading_ms", frame_index):
                        frame = self.reader.read()
                    if frame is None:
                        break

                    with self.monitor.measure("detector_inference_ms", frame_index):
                        detections = self.detector.detect(frame)
                    tracking_detections = filter_detections_to_zone(
                        detections, frame.shape, tracking_zone
                    )
                    with self.monitor.measure("tracking_ms", frame_index):
                        observations = self.tracker.update(
                            tracking_detections, frame_index, frame.shape
                        )

                    active_tracks = []
                    for observation in observations:
                        track = self.track_manager.upsert(observation, frame_index)
                        decision = self.temporal_policy.sampling_decision(
                            track, frame_index, frame.shape
                        )
                        if track.matched_in_current_frame and decision.should_sample:
                            quality = self.quality_gate.evaluate(frame, track)
                            if not quality.accepted:
                                self.track_manager.record_quality_failure(
                                    track.track_id,
                                    frame_index,
                                    quality.reasons,
                                    quality.metrics,
                                )
                            else:
                                try:
                                    with self.monitor.measure(
                                        "crop_extraction_ms", frame_index
                                    ):
                                        crop = self.crop_manager.extract(
                                            frame, track.bbox
                                        )
                                except CropError as exc:
                                    self.track_manager.record_quality_failure(
                                        track.track_id, frame_index, [str(exc)], {}
                                    )
                                else:
                                    with self.monitor.measure(
                                        "classifier_inference_ms", frame_index
                                    ):
                                        result = self.classifier.classify(crop)
                                    self.monitor.classifier_invocations += 1
                                    self.track_manager.save_temporal_observation(
                                        track.track_id, result, frame_index
                                    )
                                    aggregate = self.temporal_policy.aggregate(track)
                                    self.track_manager.apply_temporal_aggregate(
                                        track.track_id, aggregate, frame_index
                                    )

                        if track.matched_in_current_frame:
                            with self.monitor.measure(
                                "size_estimation_ms", frame_index
                            ):
                                self.size_estimator.observe(
                                    track, frame_index, frame.shape
                                )
                        active_tracks.append(track)

                    observed_ids = {track.track_id for track in active_tracks}
                    self.track_manager.update_inactive_tracks(observed_ids)
                    self.track_manager.mark_passed_tracks(
                        frame_index, lost_track_buffer
                    )
                    all_tracks = self.track_manager.get_all_tracks()

                    with self.monitor.measure("visualization_ms", frame_index):
                        annotated = self.annotator.draw(
                            frame, active_tracks, all_tracks, frame_index
                        )
                    with self.monitor.measure("video_writing_ms", frame_index):
                        self.writer.write(annotated)

                self.monitor.record_frame()
                frame_index += 1
        finally:
            self.reader.release()
            if self.writer:
                self.writer.release()

        self.track_manager.finalize_all(frame_index)
        benchmark = self.monitor.write_json(
            self.benchmark_path,
            int(self.config["logging"].get("json_indent", 2)),
        )
        tracks = self.track_manager.get_all_tracks()
        metadata = {
            "input_video": str(self.input_path),
            "output_video": str(self.output_path),
            "frame_count_processed": frame_index,
            "video_fps": self.reader.fps,
            "processing_fps": benchmark["processing_fps"],
            "tracking_zone": tracking_zone,
            "classification_policy": self.config["classification_policy"],
            "crop_padding_ratio": self.config["crop"]["padding_ratio"],
            "size_measurement": {
                "metric": "minimum_bbox_dimension_px",
                "physical_measurement": False,
                "required_samples": 1,
                "thresholds_px": self.config["size_estimation"]["thresholds"],
                "zone": self.config["size_estimation"]["zone"],
            },
        }
        report = self.report_writer.write(self.json_path, tracks, metadata)
        self.size_writer.write(self.size_csv_path, tracks)
        return {
            "output_video": str(self.output_path),
            "output_json": str(self.json_path),
            "benchmark_json": str(self.benchmark_path),
            "size_measurements_csv": str(self.size_csv_path),
            "summary": report["summary"],
            "benchmark": benchmark,
        }
