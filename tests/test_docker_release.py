"""Static release checks for the CPU Docker workflow."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DockerReleaseTests(unittest.TestCase):
    def test_runtime_depends_on_successful_test_stage(self) -> None:
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("python:3.11.11-slim-bookworm@sha256:", dockerfile)
        self.assertIn("FROM base AS test", dockerfile)
        self.assertIn("python -m unittest discover -s tests -v", dockerfile)
        self.assertIn("COPY --from=test /tmp/tests-passed", dockerfile)

    def test_cpu_requirements_are_pinned(self) -> None:
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        requirements = (
            PROJECT_ROOT / "requirements-docker-cpu.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("torch==2.5.1", dockerfile)
        self.assertIn("torchvision==0.20.1", dockerfile)
        self.assertIn("https://download.pytorch.org/whl/cpu", dockerfile)
        expected = (
            "ultralytics==8.4.125",
            "opencv-python==4.10.0.84",
            "numpy==1.26.4",
            "lap==0.5.12",
            "PyYAML==6.0.2",
        )
        for dependency in expected:
            self.assertIn(dependency, requirements)

    def test_compose_mounts_input_read_only_and_output_writable(self) -> None:
        compose = yaml.safe_load(
            (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
        )
        service = compose["services"]["tomato-sorting"]
        self.assertEqual(service["platform"], "linux/amd64")
        self.assertIn("./inputs:/data/input:ro", service["volumes"])
        self.assertIn("./outputs:/data/output", service["volumes"])

    def test_checkpoint_manifest_matches_release_files(self) -> None:
        manifest = PROJECT_ROOT / "models/checksums.sha256"
        lines = [line.split() for line in manifest.read_text().splitlines() if line]
        self.assertEqual(len(lines), 2)
        for digest, relative_path in lines:
            self.assertEqual(len(digest), 64)
            checkpoint = PROJECT_ROOT / relative_path
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
