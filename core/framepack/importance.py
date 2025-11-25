"""
Importance scoring heuristics for FramePack selection.
Lightweight, dependency-free functions to estimate which latent regions are important.

These are pluggable — later you can replace them with a face detector / keypoint detector
or a small CNN that outputs attention maps.
"""

from typing import Optional
import torch
import torch.nn.functional as F


def edge_salience_score(latent: torch.Tensor) -> float:
    """
    Compute a simple salience score by estimating gradient magnitude in the latent.
    latent: (B, C, H, W) -> returns mean across batch
    """
    if latent is None:
        return 0.0
    # convert to single-channel pseudo-image by channel-mean
    img = latent.mean(dim=1, keepdim=True)  # (B,1,H,W)
    # Sobel-like filtering via conv kernels
    kernel_x = torch.tensor([[-1, 0, 1],[-2,0,2],[-1,0,1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
    kernel_y = kernel_x.transpose(-1, -2)
    grad_x = F.conv2d(img, kernel_x, padding=1)
    grad_y = F.conv2d(img, kernel_y, padding=1)
    mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-8)
    return float(mag.mean().item())


def variance_score(latent: torch.Tensor) -> float:
    """
    Use per-pixel variance as a proxy for 'detail' in the latent.
    """
    if latent is None:
        return 0.0
    return float(latent.var().item())


def importance_combined(latent: torch.Tensor, weights: Optional[dict] = None) -> float:
    """
    Combined heuristic. weights: {"edge":0.6, "var":0.4}
    """
    if weights is None:
        weights = {"edge": 0.7, "var": 0.3}
    e = edge_salience_score(latent)
    v = variance_score(latent)
    return weights.get("edge", 0.7) * e + weights.get("var", 0.3) * v


def batch_importance_scores(batch_latents: torch.Tensor):
    """
    Return importance scores per frame in batch.
    batch_latents: (B, C, H, W)
    returns: list[float] length B
    """
    scores = []
    for i in range(batch_latents.shape[0]):
        scores.append(importance_combined(batch_latents[i:i+1]))
    return scores
