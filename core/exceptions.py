"""Readable errors raised by the tomato sorting pipeline."""


class PipelineError(Exception):
    """Base class for expected, actionable pipeline failures."""


class ConfigurationError(PipelineError):
    """Raised for invalid or contradictory configuration."""


class ModelLoadError(PipelineError):
    """Raised when trained model assets cannot be loaded safely."""


class VideoIOError(PipelineError):
    """Raised when an input or output video cannot be opened."""


class CropError(PipelineError):
    """Raised when a bounding box cannot produce a valid crop."""
