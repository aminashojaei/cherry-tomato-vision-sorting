"""Exact ShuffleNetV2 architecture required by the bundled state_dict."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import shufflenet_v2_x1_0


class ImageNormalizer(nn.Module):
    def __init__(self, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


class ShuffleNetMultiHeadModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        dropout: float,
        mean: list[float],
        std: list[float],
    ) -> None:
        super().__init__()
        base = shufflenet_v2_x1_0(weights=None)
        # The training model replaced torchvision's ImageNet classifier with an
        # identity layer and attached two independent binary heads below.
        base.fc = nn.Identity()
        self.normalizer = ImageNormalizer(mean, std)
        self.backbone = base
        self.health_head = self._make_head(feature_dim, hidden_dim, dropout)
        self.calyx_head = self._make_head(feature_dim, hidden_dim, dropout)

    @staticmethod
    def _make_head(feature_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(self.normalizer(image))
        return {
            "health_logits": self.health_head(features).squeeze(1),
            "calyx_logits": self.calyx_head(features).squeeze(1),
        }
