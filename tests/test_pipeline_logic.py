"""Unit tests runnable before heavyweight Colab dependencies are installed."""

from __future__ import annotations

import json
import pickletools
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np

from classifier.temporal_policy import AsymmetricTemporalPolicy
from core.config_loader import load_config
from core.exceptions import CropError
from core.schemas import (
    BoundingBox,
    ClassificationResult,
    Detection,
    TomatoTrack,
    TrackerObservation,
)
from crop.crop_manager import CropManager
from crop.quality_gate import CropQualityGate
from geometry.size_estimator import PixelSizeEstimator
from geometry.temporal_size_estimator import TemporalShortSideMeasurer
from tracker.bytetrack_tracker import instantiate_bytetracker
from tracker.track_manager import TrackManager
from pipeline.batch_pipeline import BatchVideoRunner, discover_video_files
from utils.logger import SizeMeasurementWriter, TomatoReportWriter
from utils.timing import PerformanceMonitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_real_model_configuration(self) -> None:
        config = load_config(PROJECT_ROOT / "config/config.yaml")
        self.assertEqual(config["detector"]["input_size"], 416)
        self.assertEqual(config["detector"]["tomato_class_id"], 0)
        self.assertEqual(config["classifier"]["type"], "shufflenet_v2_x1_0_multihead")
        self.assertEqual(config["classifier"]["input_size"], 256)
        self.assertEqual(config["classifier"]["feature_dim"], 1024)
        self.assertAlmostEqual(config["classifier"]["health"]["threshold"], 0.375)
        self.assertAlmostEqual(config["classifier"]["calyx"]["threshold"], 0.77)
        self.assertAlmostEqual(config["crop"]["padding_ratio"], 0.10)
        self.assertEqual(
            config["classification_policy"]["mode"],
            "lower_zone_asymmetric_temporal_voting",
        )
        self.assertTrue(config["visualization"]["draw_classification_zone"])
        self.assertEqual(
            config["detector"]["weights_path"],
            "models/detector/yolo_tomato_detector.pt",
        )
        self.assertTrue((PROJECT_ROOT / config["detector"]["weights_path"]).is_file())

    def test_bundled_detector_is_a_real_pytorch_checkpoint(self) -> None:
        checkpoint = PROJECT_ROOT / "models/detector/yolo_tomato_detector.pt"
        self.assertEqual(checkpoint.stat().st_size, 5447834)
        self.assertTrue(zipfile.is_zipfile(checkpoint))

    def test_bundled_checkpoint_contains_both_heads_and_internal_normalizer(
        self,
    ) -> None:
        config = load_config(PROJECT_ROOT / "config/config.yaml")
        path = PROJECT_ROOT / config["classifier"]["weights_path"]
        self.assertEqual(path.stat().st_size, 6257995)
        with zipfile.ZipFile(path) as archive:
            pickle_path = next(
                name for name in archive.namelist() if name.endswith("data.pkl")
            )
            strings = {
                argument
                for operation, argument, _ in pickletools.genops(
                    archive.read(pickle_path)
                )
                if operation.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}
            }
        required = {
            "normalizer.mean",
            "normalizer.std",
            "health_head.0.weight",
            "health_head.3.weight",
            "calyx_head.0.weight",
            "calyx_head.3.weight",
            "backbone.conv1.0.weight",
            "backbone.stage2.0.branch1.0.weight",
            "backbone.conv5.0.weight",
            "stage",
            "finetune",
            "health_threshold",
            "calyx_threshold",
        }
        self.assertTrue(required.issubset(strings))

    def test_bundled_threshold_metadata_matches_configuration(self) -> None:
        config = load_config(PROJECT_ROOT / "config/config.yaml")
        metadata_path = PROJECT_ROOT / config["classifier"]["thresholds_path"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(
            metadata["health_threshold"], config["classifier"]["health"]["threshold"]
        )
        self.assertAlmostEqual(
            metadata["calyx_threshold"], config["classifier"]["calyx"]["threshold"]
        )


class BoundingBoxTests(unittest.TestCase):
    def test_clip_and_outside_ratio(self) -> None:
        bbox = BoundingBox(-10, 10, 30, 50)
        clipped = bbox.clip_to_frame(100, 100)
        self.assertEqual(clipped.as_xyxy(), (0.0, 10, 30, 50))
        self.assertAlmostEqual(bbox.outside_ratio(100, 100), 0.25)

    def test_padding(self) -> None:
        bbox = BoundingBox(20, 30, 60, 70).expand_by_ratio(0.10)
        self.assertEqual(bbox.as_xyxy(), (16.0, 26.0, 64.0, 74.0))


class CropTests(unittest.TestCase):
    def test_extracts_original_resolution_pixels(self) -> None:
        config = load_config(PROJECT_ROOT / "config/config.yaml")
        frame = np.arange(100 * 100 * 3, dtype=np.int32).reshape(100, 100, 3)
        manager = CropManager(config["crop"])
        crop = manager.extract(frame, BoundingBox(20, 30, 60, 70))
        self.assertEqual(crop.shape, (48, 48, 3))
        np.testing.assert_array_equal(crop[0, 0], frame[26, 16])

    def test_disallows_padding_clipping_when_configured(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        manager = CropManager({"padding_ratio": 0.20, "allow_padding_clipping": False})
        with self.assertRaises(CropError):
            manager.extract(frame, BoundingBox(0, 0, 20, 20))


class TrackManagerTests(unittest.TestCase):
    def test_classifies_once_after_delayed_quality(self) -> None:
        manager = TrackManager()
        first = TrackerObservation(7, BoundingBox(0, 0, 30, 30), 0.40, 1, True, True)
        second = TrackerObservation(7, BoundingBox(2, 2, 34, 34), 0.90, 2, True, True)
        manager.upsert(first, 10)
        track = manager.upsert(second, 11)
        result = ClassificationResult(0.91, 1, 0.91, 0.87, 1, 0.87)
        manager.save_classification(7, result, 11)
        self.assertTrue(track.classified)
        self.assertEqual(track.first_frame, 10)
        self.assertEqual(track.classification_frame, 11)
        self.assertEqual(track.health_history, [0.91])
        self.assertEqual(len(track.bbox_history), 2)
        with self.assertRaises(ValueError):
            manager.save_classification(7, result, 12)

    def test_finalize_marks_unpassed_classified_track(self) -> None:
        manager = TrackManager()
        observation = TrackerObservation(
            9, BoundingBox(0, 0, 30, 30), 0.9, 1, True, True
        )
        manager.upsert(observation, 4)
        result = ClassificationResult(0.91, 1, 0.91, 0.20, 0, 0.80)
        track = manager.save_classification(9, result, 4)
        manager.finalize_all(12)
        self.assertTrue(track.passed)
        self.assertEqual(track.passed_frame, 12)


class ByteTrackCompatibilityTests(unittest.TestCase):
    def test_current_ultralytics_signature_accepts_args_only(self) -> None:
        class CurrentByteTracker:
            def __init__(self, args):
                self.args = args

        args = Namespace(track_buffer=30)
        tracker = instantiate_bytetracker(CurrentByteTracker, args, 29.97)
        self.assertIs(tracker.args, args)

    def test_legacy_frame_rate_signature_is_supported(self) -> None:
        class LegacyByteTracker:
            def __init__(self, args, frame_rate=30):
                self.args = args
                self.frame_rate = frame_rate

        tracker = instantiate_bytetracker(LegacyByteTracker, Namespace(), 24.6)
        self.assertEqual(tracker.frame_rate, 25)

    def test_alternate_fps_signature_is_supported(self) -> None:
        class AlternateByteTracker:
            def __init__(self, args, fps=30):
                self.args = args
                self.fps = fps

        tracker = instantiate_bytetracker(AlternateByteTracker, Namespace(), 0.2)
        self.assertEqual(tracker.fps, 1)


class QualityGateTests(unittest.TestCase):
    def test_low_confidence_is_explained(self) -> None:
        config = load_config(PROJECT_ROOT / "config/config.yaml")
        config["quality_gate"]["enabled"] = True
        config["quality_gate"]["min_detection_confidence"] = 0.65
        config["quality_gate"]["blur"]["enabled"] = False
        gate = CropQualityGate(config["quality_gate"], CropManager(config["crop"]))
        track = TomatoTrack(4, 0, 0, BoundingBox(10, 10, 50, 50), 0.20)
        result = gate.evaluate(np.zeros((100, 100, 3), dtype=np.uint8), track)
        self.assertFalse(result.accepted)
        self.assertIn("detection_confidence_too_low", result.reasons)

    def test_good_geometry_passes_without_blur_dependency(self) -> None:
        config = load_config(PROJECT_ROOT / "config/config.yaml")
        config["quality_gate"]["enabled"] = True
        config["quality_gate"]["blur"]["enabled"] = False
        gate = CropQualityGate(config["quality_gate"], CropManager(config["crop"]))
        track = TomatoTrack(4, 0, 0, BoundingBox(10, 10, 50, 50), 0.90)
        result = gate.evaluate(np.zeros((100, 100, 3), dtype=np.uint8), track)
        self.assertTrue(result.accepted)


class SizeEstimatorTests(unittest.TestCase):
    def test_size_thresholds(self) -> None:
        config = load_config(PROJECT_ROOT / "config/config.yaml")
        estimator = PixelSizeEstimator(config["size_estimation"])
        cases = [(150, "Small"), (151, "Medium"), (250, "Medium"), (251, "Large")]
        for dimension, expected in cases:
            track = TomatoTrack(1, 0, 0, BoundingBox(0, 0, dimension, dimension), 0.9)
            self.assertEqual(estimator.estimate(track), expected)


class ReportTests(unittest.TestCase):
    def test_unclassified_track_has_null_predictions(self) -> None:
        config = load_config(PROJECT_ROOT / "config/config.yaml")
        writer = TomatoReportWriter(config["logging"])
        track = TomatoTrack(5, 2, 7, BoundingBox(1, 2, 10, 20), 0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            result = writer.write(path, [track], {"frame_count_processed": 8})
            stored = json.loads(path.read_text())
        self.assertEqual(result["summary"]["unclassified_tracks"], 1)
        self.assertIsNone(stored["tomatoes"][0]["health_class"])
        self.assertIsNone(stored["tomatoes"][0]["calyx_class"])

    def test_final_class_combines_size_health_and_calyx(self) -> None:
        config = load_config(PROJECT_ROOT / "config/config.yaml")
        writer = TomatoReportWriter(config["logging"])
        track = TomatoTrack(8, 1, 3, BoundingBox(10, 10, 180, 180), 0.95)
        track.classified = True
        track.size_class = "Medium"
        track.health_prediction = 0
        track.calyx_prediction = 1

        record = writer.serialize_track(track)

        self.assertEqual(record["final_class"], "medium_healthy_present")


class BatchInputTests(unittest.TestCase):
    def test_discovers_multiple_videos_and_ignores_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.MOV").write_bytes(b"video")
            (root / "a.mp4").write_bytes(b"video")
            (root / "notes.txt").write_text("ignore")
            videos = discover_video_files([str(root)], PROJECT_ROOT)
        self.assertEqual([path.name for path in videos], ["a.mp4", "b.MOV"])

    def test_deduplicates_explicit_and_directory_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.mp4"
            video.write_bytes(b"video")
            videos = discover_video_files([str(root), str(video)], PROJECT_ROOT)
        self.assertEqual(videos, [video.resolve()])

    def test_batch_outputs_are_isolated_per_video(self) -> None:
        config = load_config(PROJECT_ROOT / "config/config.yaml")
        captured: list[dict] = []

        class FakePipeline:
            def __init__(self, job_config):
                captured.append(job_config)
                self.job_config = job_config

            def run(self):
                paths = self.job_config["paths"]
                return {
                    "output_video": paths["output_video"],
                    "output_json": paths["output_json"],
                    "benchmark_json": paths["benchmark_json"],
                    "summary": {},
                    "benchmark": {},
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = [root / "one.mp4", root / "two.mp4"]
            for video in videos:
                video.write_bytes(b"video")
            runner = BatchVideoRunner(
                config,
                videos,
                output_root=root / "outputs",
                pipeline_factory=FakePipeline,
            )
            result = runner.run()

        self.assertEqual(result["successful_videos"], 2)
        self.assertEqual(result["failed_videos"], 0)
        self.assertNotEqual(
            captured[0]["paths"]["output_video"],
            captured[1]["paths"]["output_video"],
        )
        self.assertEqual(
            Path(captured[0]["paths"]["output_video"]).name, "annotated.mp4"
        )


class OrchestrationTests(unittest.TestCase):
    def test_pipeline_aggregates_temporal_classifications(self) -> None:
        from pipeline.video_pipeline import VideoPipeline

        config = load_config(PROJECT_ROOT / "config/config.yaml")
        config["quality_gate"]["blur"]["enabled"] = False
        config["benchmark"]["warmup_frames"] = 0
        config["classification_policy"]["sample_stride_frames"] = 1
        config["classification_policy"]["min_samples_for_healthy"] = 3
        config["classification_policy"]["zone"] = {
            "x_min_ratio": 0.0,
            "x_max_ratio": 1.0,
            "y_min_ratio": 0.0,
            "y_max_ratio": 1.0,
        }

        class FakeReader:
            fps = 30.0
            width = 200
            height = 100

            def __init__(self) -> None:
                self.remaining = 3
                self.released = False

            def read(self):
                if self.remaining <= 0:
                    return None
                self.remaining -= 1
                return np.full((100, 200, 3), 180, dtype=np.uint8)

            def release(self) -> None:
                self.released = True

        class FakeWriter:
            def __init__(self) -> None:
                self.frames = []
                self.released = False

            def write(self, frame) -> None:
                self.frames.append(frame.copy())

            def release(self) -> None:
                self.released = True

        class FakeDetector:
            def detect(self, frame):
                return [Detection(BoundingBox(10, 10, 50, 50), 0.9, 0)]

        class FakeTracker:
            def update(self, detections, frame_index, frame_shape):
                boxes = [
                    BoundingBox(10, 5, 20, 15),
                    BoundingBox(180, 45, 190, 55),
                    BoundingBox(60, 45, 70, 55),
                ]
                return [
                    TrackerObservation(
                        7,
                        boxes[frame_index],
                        0.90,
                        frame_index + 1,
                        True,
                        True,
                    )
                ]

        class FakeClassifier:
            def __init__(self) -> None:
                self.invocations = 0

            def classify(self, crop):
                self.invocations += 1
                probability = [0.10, 0.91, 0.20][self.invocations - 1]
                prediction = int(probability >= 0.375)
                confidence = probability if prediction else 1.0 - probability
                return ClassificationResult(
                    probability, prediction, confidence, 0.87, 1, 0.87
                )

        class FakeAnnotator:
            def draw(self, frame, active_tracks, tracks, frame_index):
                return frame.copy()

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            pipeline = VideoPipeline.__new__(VideoPipeline)
            pipeline.config = config
            pipeline.root = PROJECT_ROOT
            pipeline.device = "cpu"
            pipeline.input_path = temporary / "input.mp4"
            pipeline.output_path = temporary / "output.mp4"
            pipeline.json_path = temporary / "tomatoes.json"
            pipeline.benchmark_path = temporary / "benchmark.json"
            pipeline.size_csv_path = temporary / "size_measurements.csv"
            pipeline.reader = FakeReader()
            pipeline.writer = FakeWriter()
            pipeline.detector = FakeDetector()
            pipeline.tracker = FakeTracker()
            pipeline.track_manager = TrackManager()
            pipeline.crop_manager = CropManager(config["crop"])
            pipeline.quality_gate = CropQualityGate(
                config["quality_gate"], pipeline.crop_manager
            )
            pipeline.classifier = FakeClassifier()
            pipeline.temporal_policy = AsymmetricTemporalPolicy(
                config["classification_policy"]
            )
            pipeline.size_estimator = TemporalShortSideMeasurer(
                config["size_estimation"]
            )
            pipeline.annotator = FakeAnnotator()
            pipeline.report_writer = TomatoReportWriter(config["logging"])
            pipeline.size_writer = SizeMeasurementWriter()
            pipeline.monitor = PerformanceMonitor(config["benchmark"], "cpu", False)

            result = pipeline.run()
            report = json.loads(pipeline.json_path.read_text())
            benchmark = json.loads(pipeline.benchmark_path.read_text())

        self.assertEqual(result["summary"]["total_tracks"], 1)
        self.assertEqual(result["summary"]["classified_tracks"], 1)
        self.assertEqual(pipeline.classifier.invocations, 3)
        self.assertEqual(len(pipeline.writer.frames), 3)
        self.assertTrue(pipeline.reader.released)
        self.assertTrue(pipeline.writer.released)
        tomato = report["tomatoes"][0]
        self.assertEqual(tomato["classification_frame"], 2)
        self.assertEqual(tomato["classification_frames"], [0, 1, 2])
        self.assertEqual(tomato["health_prediction_history"], [0, 1, 0])
        self.assertEqual(tomato["health_prediction_flip_count"], 2)
        self.assertEqual(tomato["classification_observation_count"], 3)
        self.assertEqual(tomato["health_class"], "healthy")
        self.assertEqual(
            [item["health_class"] for item in tomato["classification_history"]],
            ["healthy", "unhealthy", "healthy"],
        )
        self.assertEqual(len(report["tomatoes"][0]["quality_gate_failures"]), 0)
        self.assertEqual(benchmark["processed_frames"], 3)
        self.assertEqual(benchmark["classifier_invocations"], 3)
        self.assertEqual(report["summary"]["classification_observations"], 3)
        self.assertEqual(report["summary"]["health_prediction_flips"], 2)
        self.assertEqual(benchmark["latency_ms"]["total_frame_ms"]["count"], 3)


if __name__ == "__main__":
    unittest.main()
