"""Architectures share the same 10-class label order and native 64x64 patch size."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from landcover.data import RGB_INDICES


class CompactCNN(nn.Module):
    def __init__(self, channels=3, dropout=0.3):
        super().__init__()
        layers = []
        for out_channels in (32, 64, 128, 256):
            layers += [
                nn.Conv2d(channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            channels = out_channels
        self.network = nn.Sequential(
            *layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.network(x)


def expand_rgb_weights(weight):
    """RGB kernels at B4/B3/B2; RGB mean elsewhere; scale all channels by 3/13."""
    expanded = weight.mean(dim=1, keepdim=True).repeat(1, 13, 1, 1)
    for source, target in enumerate(RGB_INDICES):
        expanded[:, target] = weight[:, source]
    return expanded * (3 / 13)


def build_model(config, load_pretrained=True):
    channels = 13 if config.input_type == "ms" else 3
    if config.model == "compact":
        return CompactCNN(channels, config.dropout)
    pretrained = config.initialization == "pretrained" and load_pretrained
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if channels == 13:
        original = model.conv1.weight.detach().clone()
        model.conv1 = nn.Conv2d(13, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                model.conv1.weight.copy_(expand_rgb_weights(original))
        else:
            nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model
