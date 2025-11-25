import torch
import pytest
import sys
import os

# Path Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.samplers.framepack_injector import FramePackInjector
from core.framepack.memory_bank import FramePackMemory

# --- MOCKS ---

class MockComfyModel:
    """Simulates the ComfyUI Model Object"""
    def __init__(self):
        self.wrapper = None
        
    def clone(self):
        return self
        
    def set_model_sampler_wrapper(self, wrapper):
        self.wrapper = wrapper

def mock_unet_forward(input_x, timestep, c, cond):
    """
    Simulates the actual UNet execution.
    Just returns input_x (pass-through) so we can check shapes.
    """
    return input_x

# --- TESTS ---

def test_context_assembly_simple():
    """Test A: Basic concatenation of Anchor + Summary (Same Size)"""
    print("\n--- Test A: Simple Context Assembly ---")
    
    # 1. Setup Memory
    mem = FramePackMemory(recent_k=2)
    # Anchor is [1, C, H, W]
    mem.anchor_latents = torch.randn(1, 4, 64, 64)
    
    # Summary items are stored as [C, H, W] in the list (3D)
    # so when stacked they become [S, C, H, W]
    mem.summary_tokens = [torch.randn(4, 64, 64)] 
    
    # 2. Run Injector
    injector = FramePackInjector(mem)
    ctx, count = injector.get_context_latents()
    
    # 3. Verify
    # Count should be 2 (1 Anchor + 1 Summary)
    assert count == 2
    # Shape should be [2, 4, 64, 64]
    assert ctx.shape == (2, 4, 64, 64)
    print("✅ Simple Assembly Passed")


def test_context_assembly_hybrid_resize():
    """Test B: The Hybrid Logic (Resize 16x16 Summary to 64x64 Anchor)"""
    print("\n--- Test B: Hybrid Resizing Logic ---")
    
    mem = FramePackMemory(recent_k=2)
    # Anchor: 64x64
    mem.anchor_latents = torch.randn(1, 4, 64, 64)
    
    # Summary: 16x16 (Compressed!)
    # Note: Using 3D shape [4, 16, 16] to match MemoryBank internal storage
    mem.summary_tokens = [torch.randn(4, 16, 16)]
    
    injector = FramePackInjector(mem)
    ctx, count = injector.get_context_latents()
    
    # Verify
    assert count == 2
    # The Summary (Index 1) should have been upscaled to match Anchor (Index 0)
    assert ctx.shape == (2, 4, 64, 64)
    print(f"✅ Resize Logic Passed. Output Shape: {ctx.shape}")


def test_injection_wrapper_flow():
    """Test C: The Full Wrapper Flow (The 'Super Batch')"""
    print("\n--- Test C: Wrapper Execution & Slicing ---")
    
    # 1. Setup Context (1 Anchor)
    mem = FramePackMemory(recent_k=2)
    mem.anchor_latents = torch.randn(1, 4, 64, 64)
    
    # 2. Setup Patch
    injector = FramePackInjector(mem)
    # Pre-calculate context to ensure patch is ready
    ctx, _ = injector.get_context_latents()
    
    # 3. Apply Patch to Mock Model
    model = MockComfyModel()
    patched_model = injector.apply_patch(model, ctx)
    
    # 4. Simulate Sampling Step
    # Input: Batch of 2 frames (Standard Generation)
    input_x = torch.randn(2, 4, 64, 64)
    timestep = torch.tensor([100, 100])
    c, cond = {}, {}
    
    # This is what ComfyUI calls internally:
    wrapper_func = patched_model.wrapper
    
    # We call the wrapper, passing our 'mock_unet_forward' as the runner
    # The wrapper should:
    #   a. Prepend Anchor (1 frame) to Input (2 frames) -> Total 3
    #   b. Call mock_unet_forward with 3 frames
    #   c. Slice the output back to 2 frames
    output = wrapper_func(mock_unet_forward, (input_x, timestep, c, cond))
    
    # 5. Verify Slicing
    print(f"Input Batch Size: {input_x.shape[0]}")
    print(f"Output Batch Size: {output.shape[0]}")
    
    assert output.shape == input_x.shape
    assert output.shape[0] == 2
    
    print("✅ Wrapper Flow & Slicing Passed")

if __name__ == "__main__":
    test_context_assembly_simple()
    test_context_assembly_hybrid_resize()
    test_injection_wrapper_flow()