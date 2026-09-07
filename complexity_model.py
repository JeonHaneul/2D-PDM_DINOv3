"""RGB-D scene complexity pilot; no target, category, or segmentation input.

The 64-channel feature reserves nine channels for directly computed depth cues.
The other 55 channels are learned from frozen DINOv3 features and depth cues.
Auxiliary maps describe visible label counts and occupancy, not target probability.
"""
from pathlib import Path

import torch
from torch import nn


class ComplexityModel(nn.Module):
    def __init__(self, rgb_channels=768, geometry_channels=9, output_channels=4):
        super().__init__()
        self.rgb = nn.Sequential(nn.Conv2d(rgb_channels, 64, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.depth = nn.Sequential(nn.Conv2d(geometry_channels, 64, 3, padding=1),
                                   nn.GroupNorm(8, 64), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                  nn.GroupNorm(8, 64), nn.GELU(),
                                  nn.Conv2d(64, 64 - geometry_channels, 1))
        self.head = nn.Conv2d(64, output_channels, 1)

    def forward(self, rgb_features, geometry, mode="rgbd"):
        if mode not in ("rgbd", "depth"):
            raise ValueError(f"unknown modality: {mode}")
        if mode == "depth":
            rgb_features = torch.zeros_like(rgb_features)
        learned = self.fuse(torch.cat((self.rgb(rgb_features), self.depth(geometry)), dim=1))
        features = torch.cat((learned, geometry), dim=1)
        return self.head(features).sigmoid(), features


def load_frozen_dino(repo, weights, device):
    """Use explicit local architecture and weights; never silently download code."""
    repo, weights = Path(repo).expanduser(), Path(weights).expanduser()
    if not (repo / "hubconf.py").is_file() or not weights.is_file():
        raise FileNotFoundError("local DINO repo/hubconf.py and ViT-B/16 weights are required")
    model = torch.hub.load(str(repo), "dinov3_vitb16", source="local", weights=str(weights))
    model.eval().requires_grad_(False)
    return model.to(device)


def dino_layer11(model, images):
    return model.get_intermediate_layers(images, n=[11], reshape=True, norm=True)[0]
