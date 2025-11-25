"""
Spatial pooling based compressor for FramePack summary memory.
Provides functions to pool latents, merge tokens, and utilities to control
precision/placement (CPU/GPU).

Design choices:
- Uses avg pooling to reduce HxW resolution while preserving UNet-native spatial shape.
- Keeps tensors as torch.float32 by default; caller may cast to fp16 before passing to model.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def spatial_pool(latent: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Downsample a latent tensor from (B, C, H, W) to (B, C, h, w) via average pooling.

    Args:
        latent: (B, C, H, W)
        target_hw: (h, w) target spatial dims
    Returns:
        pooled: (B, C, h, w)
    """
    if latent is None:
        return None
    assert latent.dim() == 4, "latent must be (B, C, H, W)"
    B, C, H, W = latent.shape
    th, tw = target_hw
    if (H, W) == (th, tw):
        return latent

    # Compute pooling kernel sizes that evenly divide H and W (fallback to adaptive interpolate)
    stride_h = H // th if H % th == 0 else None
    stride_w = W // tw if W % tw == 0 else None

    if stride_h and stride_w:
        kernel = (stride_h, stride_w)
        pooled = F.avg_pool2d(latent, kernel, stride=kernel)
    else:
        # fallback to interpolation for non-divisible sizes
        pooled = F.interpolate(latent, size=(th, tw), mode='area')

    return pooled


def multi_scale_pool(latent: torch.Tensor, scales: Tuple[Tuple[int, int], ...]) -> Tuple[torch.Tensor, ...]:
    """
    Return multiple pooled versions for multi-scale summary tokens.
    Example scales: ((16,16),(8,8))
    """
    return tuple(spatial_pool(latent, s) for s in scales)


def merge_summary_token(existing: torch.Tensor, new_token: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Weighted merge of two summary tensors.

    Both tensors expected shapes: (C, h, w) or (1, C, h, w). We keep shape consistent.
    alpha ~ 0..1 is weight for new_token.
    """
    if existing is None:
        return new_token.clone()
    # align shapes (expand dims if necessary)
    if existing.dim() == 3:
        existing = existing.unsqueeze(0)
    if new_token.dim() == 3:
        new_token = new_token.unsqueeze(0)
    # ensure same device / dtype
    new_token = new_token.to(existing.device).type_as(existing)
    return (1 - alpha) * existing + alpha * new_token


def detach_to_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.detach().cpu()


def to_device(t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.to(device)
