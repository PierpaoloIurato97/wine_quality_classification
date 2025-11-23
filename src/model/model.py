import torch


class WineQualityClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 33),
            torch.nn.BatchNorm1d(33),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(33, 11),
            torch.nn.BatchNorm1d(11),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(11, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 1 or x.shape[1] != 11:
            raise ValueError(
                "Input tensor must have shape (N, 11) where N >= 1."
            )

        return self.layers(x)
