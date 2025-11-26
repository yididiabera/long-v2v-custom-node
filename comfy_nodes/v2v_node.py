import logging
import torch
from typing import Optional

import comfy.sd
import comfy.sample
import comfy.samplers
import nodes  # ComfyUI node helper wrappers (where available)

from core.framepack.memory_bank import FramePackMemory
from core.samplers.framepack_injector import get_injector
try:
    from core.samplers.v2v_sampler import prepare_latent_flow
except Exception:
    prepare_latent_flow = None

logger = logging.getLogger(__name__)


def auto_tune_recent_k(available_vram_gb: float, resolution: int = 512, model_type: str = "sd15") -> int:
    """
    Automatically calculate optimal recent_k based on available VRAM.
    
    Args:
        available_vram_gb: Available VRAM in GB
        resolution: Image resolution (512, 768, 1024, etc.)
        model_type: "sd15" or "sdxl"
    
    Returns:
        Recommended recent_k value
    """
    # Estimate base usage
    if model_type == "sdxl":
        base_usage_gb = 6.5  # SDXL base model + VAE
    else:
        base_usage_gb = 4.0  # SD1.5 base model + VAE
    
    # Estimate activation memory during sampling (rough approximation)
    if resolution >= 1024:
        activation_gb = 8.0
    elif resolution >= 768:
        activation_gb = 5.0
    else:
        activation_gb = 3.0
    
    # Calculate available headroom for context frames
    headroom_gb = max(0.5, available_vram_gb - base_usage_gb - activation_gb)
    
    # Estimate per-frame latent cost (very rough)
    latent_h = resolution // 8
    latent_w = resolution // 8
    bytes_per_frame = latent_h * latent_w * 4 * 4  # 4 channels, FP32
    gb_per_frame = bytes_per_frame / (1024 ** 3)
    
    # Calculate max frames we can fit
    max_k = int(headroom_gb / gb_per_frame)
    
    # Clamp to reasonable range
    recommended_k = max(1, min(max_k, 16))
    
    logger.info(f"[VRAM Auto-tune] Available: {available_vram_gb:.1f}GB, Recommended recent_k: {recommended_k}")
    return recommended_k


def _safe_vae_encode(vae, pixels: torch.Tensor):
    """
    Robust VAE encode helper — handles multiple Comfy versions:
      - vae.encode(tensor) -> tensor
      - vae.encode(tensor) -> {"samples": tensor}
      - vae.encode returns object with 'latent_dist'
    Returns latents tensor on same device as pixels.
    """
    with torch.no_grad():
        enc = vae.encode(pixels)
    # Common patterns:
    if isinstance(enc, dict):
        # Try typical keys
        for k in ("samples", "sample", "latent", "latents", "z"):
            if k in enc:
                latent = enc[k]
                break
        else:
            # if it contains 'latent_dist' (Diffusers style)
            if "latent_dist" in enc and hasattr(enc["latent_dist"], "mean"):
                latent = enc["latent_dist"].mean
            else:
                # fallback: take first tensor-like item
                vals = [v for v in enc.values() if torch.is_tensor(v)]
                if vals:
                    latent = vals[0]
                else:
                    raise RuntimeError("Unrecognized VAE.encode() output dict structure.")
    elif torch.is_tensor(enc):
        latent = enc
    else:
        # unknown structure
        raise RuntimeError("Unrecognized VAE.encode() return type.")
    return latent


def _safe_vae_decode(vae, latents: torch.Tensor):
    """
    Robust VAE decode helper — handles multiple Comfy versions.
    """
    with torch.no_grad():
        dec = vae.decode(latents)
    if isinstance(dec, dict):
        # typical key is 'samples' or 'images'
        for k in ("samples", "images", "sample"):
            if k in dec:
                imgs = dec[k]
                break
        else:
            vals = [v for v in dec.values() if torch.is_tensor(v)]
            if vals:
                imgs = vals[0]
            else:
                raise RuntimeError("Unrecognized VAE.decode() output dict structure.")
    elif torch.is_tensor(dec):
        imgs = dec
    else:
        raise RuntimeError("Unrecognized VAE.decode() return type.")
    return imgs


def _repeat_conditioning(cond, n_repeat: int):
    """
    Repeat conditioning (which can be a tensor or dict of tensors)
    along the batch dimension n_repeat times at the front.
    This is a shallow duplicator: if cond is a dict, duplicate each tensor.
    """
    if cond is None:
        return None
    if torch.is_tensor(cond):
        # assume shape [B, ...], repeat the first element n_repeat times as additional rows
        if cond.dim() == 0:
            return cond
        first = cond[0:1].repeat(n_repeat, *([1] * (cond.dim() - 1)))
        return torch.cat([first, cond], dim=0)
    if isinstance(cond, dict):
        out = {}
        for k, v in cond.items():
            out[k] = _repeat_conditioning(v, n_repeat)
        return out
    # unknown type: return as-is
    return cond


def _sample_with_framepack_context(
    model,
    base_latents: torch.Tensor,
    positive_cond,
    negative_cond,
    steps: int,
    cfg: float,
    denoise: float,
    seed: int,
    context_latents: Optional[torch.Tensor] = None,
):
    """Run comfy.sample.sample with optional context batching.

    If context_latents is provided, they are concatenated in the batch dimension
    ahead of base_latents, conditioning is repeated for these context frames,
    and the sampler output is sliced to drop the context portion.
    """
    # Prepare latent batch: [N_ctx + B, C, H, W] if context is present
    latent_image = base_latents
    n_ctx = 0
    if context_latents is not None:
        ctx = context_latents.to(latent_image.device).type(latent_image.dtype)
        n_ctx = ctx.shape[0]
        if n_ctx > 0:
            latent_image = torch.cat([ctx, latent_image], dim=0)

            # Duplicate conditioning for context frames (front of batch)
            if positive_cond is not None:
                positive_cond = _repeat_conditioning(positive_cond, n_ctx)
            if negative_cond is not None:
                negative_cond = _repeat_conditioning(negative_cond, n_ctx)

    # Build noise for the (possibly expanded) batch
    noise = comfy.sample.prepare_noise(latent_image, seed)

    # Invoke core Comfy sampler API (stable across versions)
    samples_all = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name="euler",
        scheduler="normal",
        positive=positive_cond,
        negative=negative_cond,
        latent_image=latent_image,
        denoise=denoise,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        noise_mask=None,
        sigmas=None,
        callback=None,
        disable_pbar=False,
        seed=seed,
    )

    # If we prepended context frames, discard their outputs
    if n_ctx > 0:
        if samples_all.shape[0] < n_ctx:
            raise RuntimeError("Sampler returned fewer latents than context frames.")
        return samples_all[n_ctx:]
    return samples_all


class AnimeFramePackNode:
    """
    Production-ready FramePack node for ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image_batch": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0}),
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0}),
                "recent_k": ("INT", {"default": 6, "min": 1, "max": 32}),
                "bypass_mode": ("BOOLEAN", {"default": True}),
                "auto_tune_vram": ("BOOLEAN", {"default": False}),
                "enable_scene_detection": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "prev_history": ("FRAMEPACK_MEM",),
            },
        }

    RETURN_TYPES = ("IMAGE", "FRAMEPACK_MEM")
    RETURN_NAMES = ("processed_images", "history_state")
    FUNCTION = "process_frames"
    CATEGORY = "AnimeProduction/Video"

    def process_frames(
        self,
        model,
        vae,
        image_batch,
        seed,
        steps,
        cfg,
        denoise,
        recent_k,
        bypass_mode,
        auto_tune_vram=False,
        enable_scene_detection=False,
        positive=None,
        negative=None,
        prev_history=None,
    ):
        # 1) Bypass mode (safe for laptop)
        if bypass_mode:
            logger.info("[FramePack] BYPASS MODE active. returning simulated outputs.")
            processed_sim = image_batch.clone().float() * 0.5
            if isinstance(prev_history, FramePackMemory):
                memory = prev_history
            else:
                memory = FramePackMemory(recent_k=recent_k)
            prev_step = int(memory.meta.get("step_count", 0))
            memory.meta["step_count"] = prev_step + 1
            memory.meta["status"] = "mock_data"
            return processed_sim, memory

        # 2) Production mode: check device & prepare memory
        device = comfy.model_management.get_torch_device()
        
        # 2a) Auto-tune recent_k if enabled
        if auto_tune_vram:
            try:
                total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                allocated_vram = torch.cuda.memory_allocated(device) / (1024**3)
                available_vram = total_vram - allocated_vram
                
                # Detect resolution from image batch
                resolution = image_batch.shape[1]  # Assuming square images
                recent_k = auto_tune_recent_k(available_vram, resolution)
                logger.info(f"[FramePack] Auto-tuned recent_k to {recent_k}")
            except Exception as e:
                logger.warning(f"[FramePack] Auto-tune failed: {e}. Using manual recent_k={recent_k}")
        
        # Initialize or resume memory
        if isinstance(prev_history, FramePackMemory):
            memory = prev_history
            logger.info(f"[FramePack] Resuming memory (frames: {memory.total_frames_processed}).")
        else:
            memory = FramePackMemory(recent_k=recent_k)
            logger.info("[FramePack] Initialized new memory bank.")

        # 3) Prepare pixels and encode into latents
        pixels = image_batch.to(device)
        # Ensure we have a 4-channel pixel input if Comfy provided different shape
        # Comfy UI usually uses HWC ordering; assume image_batch is already correct
        try:
            latents = _safe_vae_encode(vae, pixels[:, :, :, :3])
        except Exception as e:
            logger.error("VAE encode failed: %s", e)
            raise

        # Ensure latents are float32 and on device
        latents = latents.to(device).float()

        # 4) Ingest into memory (CPU store for safety)
        # Keep a CPU copy to avoid holding GPU tensors in the workflow
        memory.ingest_batch(latents.detach().cpu(), enable_scene_detection=enable_scene_detection)

        # 5) Prepare base latent for sampling (latent flow or identity)
        if prepare_latent_flow is not None:
            try:
                latent_base = prepare_latent_flow(memory, batch_size=latents.shape[0], height=latents.shape[-2] * 8, width=latents.shape[-1] * 8, denoise_strength=denoise, device=device)
            except Exception:
                latent_base = latents.to(device)
        else:
            latent_base = latents.to(device)

        # 6) Setup injector and context
        injector = get_injector(memory, device=device)
        context_tensors, n_ctx = injector.get_context_latents()
        if context_tensors is not None and n_ctx > 0:
            logger.info(f"[FramePack] Injecting context of {n_ctx} frames (via latent batching).")
        else:
            logger.info("[FramePack] No context frames available for injection.")

        # 7) If no prompts, return passthrough decoded images
        if positive is None or negative is None:
            logger.warning("[FramePack] No conditioning provided — skipping diffusion sampling.")
            out_images = _safe_vae_decode(vae, latent_base)
            return out_images, memory

        # 8) Conditioning: we delegate repetition for context frames to
        # _sample_with_framepack_context, so pass through as-is here.
        positive_exp = positive
        negative_exp = negative

        # 9) Handle latent scale if model expects it (some models require scaling)
        latent_multiplier = 1.0
        if hasattr(model, "latent_scale_factor"):
            try:
                latent_multiplier = float(getattr(model, "latent_scale_factor"))
            except Exception:
                latent_multiplier = 1.0

        sample_latents = latent_base * latent_multiplier

        # 10) Invoke sampler via core comfy.sample API with context batching
        generated_latents = _sample_with_framepack_context(
            model,
            sample_latents,
            positive_exp,
            negative_exp,
            steps,
            cfg,
            denoise,
            seed,
            context_tensors if n_ctx > 0 else None,
        )

        # 11) If sampler returned (samples, extras) tuple, extract first element
        if isinstance(generated_latents, tuple) or isinstance(generated_latents, list):
            if len(generated_latents) >= 1 and torch.is_tensor(generated_latents[0]):
                generated_latents = generated_latents[0]
            else:
                raise RuntimeError("Sampler returned an unexpected structure.")

        # Safety: ensure latents on CPU/GPU expected device
        generated_latents = generated_latents.to(device)

        # 12) Undo latent multiplier if applied
        if latent_multiplier != 1.0:
            generated_latents = generated_latents / latent_multiplier

        # 13) Decode images
        out_images = _safe_vae_decode(vae, generated_latents)

        # 14) Return images and serializable memory metadata (avoid returning GPU tensors inside workflow)
        return out_images, memory
