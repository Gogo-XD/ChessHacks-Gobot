import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
  def __init__(self, channels: int):
    super().__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += residual
    out = F.relu(out)
    return out


class ValueNet(nn.Module):
  def __init__(self, in_channels=32, channels=32, num_blocks=4):
    super().__init__()
    # Input conv
    self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
    self.bn_in = nn.BatchNorm2d(channels)

    # Residual blocks
    self.blocks = nn.ModuleList(
        [ResidualBlock(channels) for _ in range(num_blocks)]
    )

    # Value head
    self.fc1 = nn.Linear(channels, 64)
    self.fc2 = nn.Linear(64, 1)

  def forward(self, x):
    # x: [B, 18, 8, 8]
    out = self.conv_in(x)
    out = self.bn_in(out)
    out = F.relu(out)

    for block in self.blocks:
        out = block(out)

    # Global average pooling over 8x8 → [B, C]
    out = out.mean(dim=[2, 3])

    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    # Optional tanh to keep in [-1, 1]
    out = torch.tanh(out)
    return out
