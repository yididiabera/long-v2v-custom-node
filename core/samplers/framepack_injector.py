import torch
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class FramePackInjector:
    """
    Implements 'Context Batching' (Hybrid B+C).
    Strategy: Zero-Weight Context Injection via Input Concatenation.
    """
    
    def __init__(self, memory_bank, device="cpu"):
        self.memory_bank = memory_bank
        self.device = device

    def get_context_latents(self) -> Tuple[Optional[torch.Tensor], int]:
        """
        Retrieves memory latents and stacks them for Input Concatenation.
        Returns:
            context_latents: [N, 4, H, W]
            context_count: N
        """
        # 1. Fetch from Memory Bank
        anchor, _, summary = self.memory_bank.get_context_for_sampler(device=self.device)
        
        context_list = []
        
        # 2. Add Anchor (Identity Lock)
        if anchor is not None:
            context_list.append(anchor)
            
        # 3. Add Summary (Long Term Context)
        if summary is not None and summary.shape[0] > 0:
            # Critical Check: Size Mismatch
            # If summary is 16x16 but anchor is 64x64, Context Batching requires resizing.
            if anchor is not None and summary.shape[-1] != anchor.shape[-1]:
                # Upsample summary to match Anchor (bicubic for sharper edges)
                target_size = (anchor.shape[-2], anchor.shape[-1])
                summary_resized = torch.nn.functional.interpolate(
                    summary, size=target_size, mode='bicubic', align_corners=False
                )
                context_list.append(summary_resized)
            else:
                context_list.append(summary)
        
        if not context_list:
            return None, 0
            
        # Stack: [N, 4, H, W]
        try:
            context_latents = torch.cat(context_list, dim=0)
        except RuntimeError as e:
            logger.error(f"FramePack: Failed to concat context. Shapes: {[t.shape for t in context_list]}")
            return None, 0
            
        return context_latents, context_latents.shape[0]

    def apply_patch(self, model, context_latents):
        """
        Wraps the ComfyUI model to handle the Super Batch logic.
        """
        if context_latents is None:
            return model
            
        def framepack_forward_wrapper(apply_model_func, args):
            input_x, timestep, c, cond = args
            
            # 1. Prepare Context
            ctx = context_latents.to(input_x.device).type(input_x.dtype)
            n_ctx = ctx.shape[0]
            
            # 2. Create Super Batch: [Context + Input]
            combined_x = torch.cat([ctx, input_x], dim=0)
            
            # 3. Expand Timesteps
            # Repeat current timestep for context frames
            t_repeat = timestep[0].repeat(n_ctx) if timestep.dim() > 0 else timestep.repeat(n_ctx)
            combined_t = torch.cat([t_repeat, timestep], dim=0)
            
            # 4. Execute Model
            # Note: We rely on standard broadcasting for 'c' or model internal handling
            output = apply_model_func(combined_x, combined_t, c, cond)
            
            # 5. Slice Output (Discard Context)
            return output[n_ctx:]

        m = model.clone()
        m.set_model_sampler_wrapper(framepack_forward_wrapper)
        return m

def get_injector(memory_bank, device="cpu"):
    return FramePackInjector(memory_bank, device)