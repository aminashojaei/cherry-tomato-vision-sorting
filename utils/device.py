"""Runtime device selection with explicit CUDA validation."""

from __future__ import annotations

from core.exceptions import ConfigurationError


def select_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigurationError(
            "CUDA was requested but is unavailable. Select a GPU runtime in Colab or set device=cpu."
        )
    return requested
