import torch


class WineQualityClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 1 or x.shape[1] != 11:
            raise ValueError(
                "Input tensor must have shape (N, 11) where N >= 1."
            )

        return self.layers(x)
