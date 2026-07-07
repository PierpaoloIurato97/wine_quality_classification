import torch

import config


class WineQualityClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(len(config.FEATURES), 14),
            torch.nn.BatchNorm1d(14),
            torch.nn.ReLU(),
            torch.nn.Linear(14, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 1 or x.shape[1] != len(config.FEATURES):
            raise ValueError(
                f"Input tensor must have shape (N, {len(config.FEATURES)}) where N >= 1."
            )

        return self.layers(x)
