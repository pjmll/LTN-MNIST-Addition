import torch
from torch import Tensor, nn


class DigitCNN(nn.Module):
    """Map each 28×28 grayscale image to ten digit logits."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.Tanh(),
            nn.Linear(100, 84),
            nn.BatchNorm1d(84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.normal_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.normal_(layer.bias)

    def forward(self, images: Tensor) -> Tensor:
        return self.layers(images)

    def forward_pair(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        """Apply shared weights and BatchNorm statistics to both positions."""
        logits = self(torch.cat((left, right), dim=0))
        return logits.split((len(left), len(right)), dim=0)
