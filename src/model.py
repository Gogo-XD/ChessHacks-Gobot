import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = out + x
        out = F.relu(out)
        return out


class ValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 18,
        channels: int = 16,
        num_blocks: int = 2,
        fc_hidden: int = 32,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, bias=False
        )

        self.blocks = nn.ModuleList(
            ResidualBlock(channels) for _ in range(num_blocks)
        )

        self.fc1 = nn.Linear(channels, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        out = F.relu(out)

        for block in self.blocks:
            out = block(out)

        out = out.mean(dim=(2, 3))

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        out = torch.tanh(out)
        return out
