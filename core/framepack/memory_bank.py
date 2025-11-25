"""
FramePackMemory: Anchor + Recent + Summary memory implementation.

Hybrid strategy:
- Recent window stores full-resolution latents (B, C, H, W) for K frames
- Summary memory stores pooled latents (C, h, w) where h,w << H,W
- Anchor stores 1-3 full-resolution anchor latents

Public API:
- update(new_latents) -> ingests a batch, updates recent and summary
- get_context_for_sampler() -> returns (anchor_tokens, recent_tensor, summary_tensor)
- save(path_prefix) / load(path_prefix)
"""

import os
import json
import logging
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .compressor import spatial_pool, merge_summary_token, detach_to_cpu
from .importance import batch_importance_scores

logger = logging.getLogger(__name__)


_DEFAULT_RECENT_K = 6
_DEFAULT_SUMMARY_HW = (16, 16)  # pooled spatial size
_DEFAULT_MAX_SUMMARY = 8  # number of pooled tensors kept
_DEFAULT_SCENE_CUT_THRESHOLD = 0.3  # MSE threshold for scene cut detection


@dataclass
class FramePackMemory:
    # Config
    recent_k: int = _DEFAULT_RECENT_K
    summary_hw: Tuple[int, int] = _DEFAULT_SUMMARY_HW
    max_summary: int = _DEFAULT_MAX_SUMMARY
    scene_cut_threshold: float = _DEFAULT_SCENE_CUT_THRESHOLD

    # Core tensors
    anchor_latents: Optional[torch.Tensor] = None  # (C,H,W) or (1,C,H,W)
    recent_latents: List[torch.Tensor] = field(default_factory=list)  # list of (1,C,H,W)
    summary_tokens: List[torch.Tensor] = field(default_factory=list)  # list of (C,h,w) or (1,C,h,w)
    summary_scores: List[float] = field(default_factory=list)  # importance scores for summary tokens

    # counters
    total_frames_processed: int = 0
    update_step: int = 0

    # scene tracking
    last_frame_for_scene_detection: Optional[torch.Tensor] = None

    # misc metadata
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.recent_k > 64 or self.max_summary > 64:
            logger.warning(
                "FramePackMemory configured with large recent_k or max_summary (recent_k=%d, max_summary=%d)",
                self.recent_k,
                self.max_summary,
            )

    def sanity_check(self):
        if self.anchor_latents is not None:
            assert isinstance(self.anchor_latents, torch.Tensor)
        assert len(self.summary_tokens) == len(self.summary_scores), "Summary tokens and scores must be in sync"

    def detect_scene_cut(self, current_frame: torch.Tensor) -> bool:
        """
        Detect if there's a scene cut between the last frame and current frame.
        Uses MSE between frames as a simple heuristic.
        
        Args:
            current_frame: (1, C, H, W) or (C, H, W)
        Returns:
            True if scene cut detected
        """
        if self.last_frame_for_scene_detection is None:
            return False
        
        # Ensure both frames have same shape
        prev = self.last_frame_for_scene_detection
        curr = current_frame
        
        if prev.dim() == 4 and prev.shape[0] == 1:
            prev = prev.squeeze(0)
        if curr.dim() == 4 and curr.shape[0] == 1:
            curr = curr.squeeze(0)
        
        # Calculate MSE
        mse = torch.nn.functional.mse_loss(prev.float(), curr.float()).item()
        
        logger.debug(f"Scene cut detection MSE: {mse:.4f} (threshold: {self.scene_cut_threshold})")
        return mse > self.scene_cut_threshold

    def reset_anchor(self, new_anchor: torch.Tensor):
        """
        Reset the anchor frame (e.g., on scene cut).
        
        Args:
            new_anchor: (1, C, H, W) new anchor frame
        """
        logger.info("Resetting anchor frame due to scene cut")
        self.anchor_latents = new_anchor.detach().clone().cpu()
        # Optionally clear summary memory on scene change
        # self.summary_tokens.clear()
        # self.summary_scores.clear()

    # Persistence 
    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix) or '.', exist_ok=True)
        meta_path = f"{path_prefix}.meta.json"
        pt_path = f"{path_prefix}.pt"

        meta = {
            'recent_count': len(self.recent_latents),
            'summary_count': len(self.summary_tokens),
            'summary_scores': self.summary_scores,
            'total_frames_processed': self.total_frames_processed,
            'update_step': self.update_step,
            'meta': self.meta,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # Save tensors as CPU to avoid device mismatch
        tensors = {}
        if self.anchor_latents is not None:
            tensors['anchor'] = detach_to_cpu(self.anchor_latents)
        for i, t in enumerate(self.recent_latents):
            tensors[f'recent__{i}'] = detach_to_cpu(t)
        for i, t in enumerate(self.summary_tokens):
            tensors[f'summary__{i}'] = detach_to_cpu(t)

        torch.save(tensors, pt_path)
        return {'meta': meta_path, 'pt': pt_path}

    @staticmethod
    def load(path_prefix: str, map_location: Optional[torch.device] = None):
        meta_path = f"{path_prefix}.meta.json"
        pt_path = f"{path_prefix}.pt"
        if not os.path.exists(meta_path) or not os.path.exists(pt_path):
            raise FileNotFoundError('Missing memory files')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        tensors = torch.load(pt_path, map_location=map_location)
        mem = FramePackMemory()
        mem.total_frames_processed = meta.get('total_frames_processed', 0)
        mem.update_step = meta.get('update_step', 0)
        mem.meta = meta.get('meta', {})
        mem.summary_scores = meta.get('summary_scores', [])

        if 'anchor' in tensors:
            mem.anchor_latents = tensors['anchor']
        mem.recent_latents = [tensors[k] for k in sorted(tensors.keys()) if k.startswith('recent__')]
        mem.summary_tokens = [tensors[k] for k in sorted(tensors.keys()) if k.startswith('summary__')]
        
        # Ensure summary_scores length matches summary_tokens
        while len(mem.summary_scores) < len(mem.summary_tokens):
            mem.summary_scores.append(0.0)
        
        return mem

    #  update / compress 
    def ingest_batch(self, new_latents: torch.Tensor, enable_scene_detection: bool = False):
        """
        Add a batch of new latents: new_latents shape (B, C, H, W)
        
        Args:
            new_latents: Batch of latent frames (B, C, H, W)
            enable_scene_detection: If True, detect scene cuts and reset anchor
        """
        B = new_latents.shape[0]
        
        # Scene cut detection on first frame of batch
        if enable_scene_detection and B > 0:
            if self.detect_scene_cut(new_latents[0:1]):
                self.reset_anchor(new_latents[0:1])
        
        # set anchor if missing
        if self.anchor_latents is None:
            self.anchor_latents = new_latents[0:1].detach().clone().cpu()
        
        # Update last frame for scene detection
        if B > 0:
            self.last_frame_for_scene_detection = new_latents[-1:].detach().clone().cpu()

        # push into recent window as per-frame tensors on CPU
        for i in range(B):
            t = new_latents[i:i+1].detach().clone()
            self.recent_latents.append(t)

        # Trim recent window and compress overflow into summary
        while len(self.recent_latents) > self.recent_k:
            oldest = self.recent_latents.pop(0)  # (1,C,H,W)
            # compress spatially
            pooled = spatial_pool(oldest, self.summary_hw).squeeze(0)  # (C,h,w)
            # importance score
            score = float(batch_importance_scores(oldest)[0])
            
            if len(self.summary_tokens) < self.max_summary:
                # Always store summary tokens in a consistent 3D shape (C,h,w) on CPU
                self.summary_tokens.append(pooled.cpu())
                self.summary_scores.append(score)
            else:
                # Priority-based replacement: replace lowest-importance token
                min_score_idx = self.summary_scores.index(min(self.summary_scores))
                
                if score > self.summary_scores[min_score_idx]:
                    # New frame is more important, replace
                    logger.debug(f"Replacing summary token {min_score_idx} (score {self.summary_scores[min_score_idx]:.4f} -> {score:.4f})")
                    self.summary_tokens[min_score_idx] = pooled.cpu()
                    self.summary_scores[min_score_idx] = score
                else:
                    # Merge with lowest-importance token using exponential moving average
                    existing = self.summary_tokens[min_score_idx]
                    merged = merge_summary_token(existing, pooled.cpu(), alpha=0.3)
                    # Ensure merged token is also stored as 3D (C,h,w) on CPU to keep list stackable
                    if merged.dim() == 4 and merged.shape[0] == 1:
                        merged = merged.squeeze(0)
                    self.summary_tokens[min_score_idx] = merged.cpu()
                    # Update score with weighted average
                    self.summary_scores[min_score_idx] = 0.7 * self.summary_scores[min_score_idx] + 0.3 * score

        self.total_frames_processed += B
        self.update_step += 1

    def get_context_for_sampler(self, device: Optional[torch.device] = None):
        """
        Returns three tensors (anchor, recent_tensor, summary_tensor)
        - anchor: (1, C, H, W) or None
        - recent_tensor: (K, C, H, W) stacked (on-device if device provided)
        - summary_tensor: (S, C, h, w) stacked or None
        """
        anchor = self.anchor_latents
        if anchor is not None and device is not None:
            anchor = anchor.to(device)
        recent_tensor = None
        if self.recent_latents:
            recent_tensor = torch.cat(self.recent_latents, dim=0)
            if device is not None:
                recent_tensor = recent_tensor.to(device)
        summary_tensor = None
        if self.summary_tokens:
            # Normalize tokens to shape (C, h, w) before stacking so we never mix 3D/4D ranks
            normalized_tokens = []
            for t in self.summary_tokens:
                if t.dim() == 4 and t.shape[0] == 1:
                    normalized_tokens.append(t.squeeze(0))
                elif t.dim() == 3:
                    normalized_tokens.append(t)
                else:
                    raise ValueError("summary token must have shape (C,h,w) or (1,C,h,w)")

            # stack and ensure shape (S, C, h, w)
            summary_tensor = torch.stack(normalized_tokens, dim=0)
            if device is not None:
                summary_tensor = summary_tensor.to(device)
        return anchor, recent_tensor, summary_tensor

    def to_dict(self):
        return {
            'total_frames_processed': self.total_frames_processed,
            'update_step': self.update_step,
            'recent_count': len(self.recent_latents),
            'summary_count': len(self.summary_tokens),
            'meta': self.meta,
        }

    @staticmethod
    def create_empty(recent_k: int = _DEFAULT_RECENT_K, summary_hw: Tuple[int, int] = _DEFAULT_SUMMARY_HW,
                     max_summary: int = _DEFAULT_MAX_SUMMARY, scene_cut_threshold: float = _DEFAULT_SCENE_CUT_THRESHOLD):
        return FramePackMemory(recent_k=recent_k, summary_hw=summary_hw, max_summary=max_summary,
                             scene_cut_threshold=scene_cut_threshold)