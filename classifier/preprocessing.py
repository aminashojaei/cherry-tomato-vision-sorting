"""Convert an OpenCV BGR crop into an unnormalized RGB tensor."""

from __future__ import annotations

import numpy as np


def preprocess_crop(crop: np.ndarray, input_size: int, device: str):
    import cv2
    import torch

    if crop.size == 0:
        raise ValueError("Classifier received an empty crop.")

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    contiguous = np.ascontiguousarray(resized.transpose(2, 0, 1))
    tensor = torch.from_numpy(contiguous).float().div(255.0).unsqueeze(0)
    # The trained ShuffleNet checkpoint has normalizer.mean/std buffers. Never normalize here.
    return tensor.to(device)
