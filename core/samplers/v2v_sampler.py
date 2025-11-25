import torch
import logging

logger = logging.getLogger(__name__)

def prepare_latent_flow(memory_bank, batch_size, height, width, denoise_strength, device="cpu"):
    """
    Produces the starting latent for the next Video2Video batch.
    
    CRITICAL COMFYUI NOTE:
    We do NOT add noise here. We return the 'Clean' repeated frames.
    The ComfyUI KSampler (in the node) will take this clean latent and 
    the 'denoise' parameter to calculate the correct sigma and add noise itself.
    
    If we added noise here, the KSampler would double-noise the image.
    """
    
    # Standard Latent Dimensions (SD1.5/SDXL = 1/8th scale)
    c_h, c_w = height // 8, width // 8
    latent_shape = (batch_size, 4, c_h, c_w)
    
    # Get Last Frame from Memory
    # We use memory_bank to get the data, maintaining the integration logic
    _, recent, _ = memory_bank.get_context_for_sampler(device=device)
    
    # SCENARIO A: First Batch (Random Generation)
    if recent is None:
        logger.info("[LatentFlow] First batch → Generating random latent.")
        return torch.randn(latent_shape, device=device)

    # SCENARIO B: Continuation (Flow)
    # recent is a stacked tensor (K, C, H, W), get the last frame
    last_frame = recent[-1:]  # Shape: (1, C, H, W)
    last_frame = last_frame.to(device)
    
    # Handle Resolution Changes (Safety)
    if last_frame.shape[-2:] != (c_h, c_w):
         last_frame = torch.nn.functional.interpolate(last_frame, size=(c_h, c_w), mode="bilinear")

    # Repeat for Batch
    # We return the CLEAN last frame repeated N times.
    # The KSampler will see this and say: "Ah, you want me to denoise this image at 0.6 strength."
    logger.info(f"[LatentFlow] Flowing from previous frame (Clean). Denoise={denoise_strength:.2f} handled by Sampler.")
    
    return last_frame.repeat(batch_size, 1, 1, 1)