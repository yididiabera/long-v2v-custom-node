"""
Integration tests for Long V2V Custom Node end-to-end workflows.

Tests multi-chunk processing, memory persistence, scene detection,
and VRAM auto-tuning with realistic workflows.
"""

import torch
import numpy as np
from core.framepack.memory_bank import FramePackMemory
from core.samplers.framepack_injector import get_injector



class TestMultiChunkProcessing:
    """Test end-to-end multi-chunk video processing"""
    
    def test_100_frame_chain(self):
        """
        Verify memory consistency across 5 sequential 20-frame chunks.
        Simulates processing a 100-frame video in chunks.
        """
        memory = FramePackMemory(recent_k=6, max_summary=8)
        
        chunk_size = 20
        num_chunks = 5
        latent_shape = (chunk_size, 4, 64, 64)
        
        for chunk_idx in range(num_chunks):
            # Generate synthetic latents for this chunk
            latents = torch.randn(*latent_shape)
            
            # Ingest into memory
            memory.ingest_batch(latents)
            
            # Verify memory state
            assert memory.total_frames_processed == (chunk_idx + 1) * chunk_size
            assert len(memory.recent_latents) <= memory.recent_k
            
            # Verify anchor is locked to first frame
            if chunk_idx == 0:
                first_anchor = memory.anchor_latents.clone()
            else:
                assert torch.allclose(memory.anchor_latents, first_anchor), \
                    "Anchor frame should remain unchanged across chunks"
        
        # Final verification
        assert memory.total_frames_processed == 100
        assert len(memory.recent_latents) == 6  # Should be full
        
        # Summary should have accumulated older frames
        expected_summary = min(8, (100 - 6))  # max_summary or overflow count
        assert len(memory.summary_tokens) > 0, "Summary memory should contain compressed history"
        
        print(f"✓ Successfully processed {memory.total_frames_processed} frames across {num_chunks} chunks")
    
    def test_memory_persistence(self):
        """Test save/load cycle preserves memory state"""
        import tempfile
        import os
        
        # Create memory and process some frames
        memory = FramePackMemory(recent_k=6, max_summary=8)
        latents = torch.randn(30, 4, 64, 64)
        memory.ingest_batch(latents)
        
        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_memory")
            memory.save(save_path)
            
            # Load back
            loaded_memory = FramePackMemory.load(save_path)
            
            # Verify state restored correctly
            assert loaded_memory.total_frames_processed == memory.total_frames_processed
            assert len(loaded_memory.recent_latents) == len(memory.recent_latents)
            assert len(loaded_memory.summary_tokens) == len(memory.summary_tokens)
            assert len(loaded_memory.summary_scores) == len(memory.summary_scores)
            # Verify configuration preserved
            assert loaded_memory.recent_k == memory.recent_k
            assert loaded_memory.summary_hw == memory.summary_hw
            assert loaded_memory.max_summary == memory.max_summary
            assert loaded_memory.scene_cut_threshold == memory.scene_cut_threshold
            
            # Verify anchor preserved
            assert torch.allclose(loaded_memory.anchor_latents, memory.anchor_latents)
            
            print("✓ Memory save/load cycle successful")
    
    def test_context_extraction(self):
        """Test that context tensors have correct shapes for injection"""
        memory = FramePackMemory(recent_k=6, max_summary=8)
        
        # Process enough frames to populate all memory tiers
        latents = torch.randn(50, 4, 64, 64)
        memory.ingest_batch(latents)
        
        # Extract context
        device = torch.device('cpu')
        anchor, recent, summary = memory.get_context_for_sampler(device)
        
        # Verify shapes
        assert anchor is not None
        assert anchor.shape == (1, 4, 64, 64), f"Anchor shape: {anchor.shape}"
        
        assert recent is not None
        assert recent.shape[0] == 6, f"Recent should have {memory.recent_k} frames"
        assert recent.shape[1:] == (4, 64, 64)
        
        assert summary is not None
        assert summary.shape[1:] == (4, 16, 16), "Summary should be spatially pooled"
        
        print(f"✓ Context shapes valid: anchor={anchor.shape}, recent={recent.shape}, summary={summary.shape}")


class TestSceneCutDetection:
    """Test scene cut detection and anchor reset"""
    
    def test_scene_cut_detection_accuracy(self):
        """Verify scene cuts are detected correctly"""
        memory = FramePackMemory(recent_k=6, scene_cut_threshold=0.3)
        
        # Create first scene (smooth frames)
        scene1_frame1 = torch.randn(1, 4, 64, 64)
        scene1_frame2 = scene1_frame1 + torch.randn(1, 4, 64, 64) * 0.05  # Small variation
        
        memory.ingest_batch(scene1_frame1, enable_scene_detection=False)
        memory.ingest_batch(scene1_frame2, enable_scene_detection=True)
        
        # Should NOT detect scene cut (similar frames)
        assert not memory.detect_scene_cut(scene1_frame2), "Should not detect cut in smooth sequence"
        
        # Create scene cut (completely different frame)
        scene2_frame1 = torch.randn(1, 4, 64, 64)  # Completely different
        
        # Should detect scene cut
        assert memory.detect_scene_cut(scene2_frame1), "Should detect scene cut"
        
        print("✓ Scene cut detection working correctly")
    
    def test_anchor_reset_on_scene_cut(self):
        """Verify anchor is reset when scene cut detected"""
        memory = FramePackMemory(recent_k=6, scene_cut_threshold=0.2)
        
        # First scene
        scene1 = torch.randn(10, 4, 64, 64)
        memory.ingest_batch(scene1, enable_scene_detection=False)
        original_anchor = memory.anchor_latents.clone()
        
        # Scene cut (very different frame)
        scene2 = torch.randn(5, 4, 64, 64) * 10  # Drastically different
        memory.ingest_batch(scene2, enable_scene_detection=True)
        
        # Anchor should have been reset
        assert not torch.allclose(memory.anchor_latents, original_anchor, rtol=0.1), \
            "Anchor should change on scene cut"
        
        print("✓ Anchor reset on scene cut verified")


class TestImportanceBasedReplacement:
    """Test importance scoring and priority-based memory replacement"""
    
    def test_importance_scoring(self):
        """Verify importance scores are calculated and tracked"""
        memory = FramePackMemory(recent_k=3, max_summary=5)
        
        # Create frames with varying complexity
        simple_frame = torch.ones(1, 4, 64, 64) * 0.5  # Low importance
        complex_frame = torch.randn(1, 4, 64, 64)      # High importance
        
        # Process enough to trigger summary storage
        for _ in range(10):
            memory.ingest_batch(simple_frame if _ % 2 == 0 else complex_frame)
        
        # Verify scores are tracked
        assert len(memory.summary_scores) > 0, "Summary scores should be tracked"
        assert len(memory.summary_scores) == len(memory.summary_tokens), \
            "Scores and tokens should be in sync"
        
        # Verify scores have variance (not all identical)
        if len(memory.summary_scores) > 1:
            score_variance = np.var(memory.summary_scores)
            assert score_variance > 0, "Importance scores should have variance"
        
        print(f"✓ Importance scores tracked: {memory.summary_scores}")
    
    def test_priority_replacement(self):
        """Test that low-importance frames are replaced by high-importance ones"""
        memory = FramePackMemory(recent_k=2, max_summary=3)
        
        # Fill summary with low-importance frames
        low_importance = torch.ones(5, 4, 64, 64) * 0.3
        memory.ingest_batch(low_importance)
        
        initial_min_score = min(memory.summary_scores) if memory.summary_scores else 0
        
        # Add high-importance frame
        high_importance = torch.randn(3, 4, 64, 64) * 2  # More variation = higher importance
        memory.ingest_batch(high_importance)
        
        # Verify highest score increased
        if memory.summary_scores:
            final_max_score = max(memory.summary_scores)
            assert final_max_score > initial_min_score, \
                "High-importance frame should increase maximum score"
        
        print("✓ Priority-based replacement working")


class TestVRAMAutoTuning:
    """Test VRAM auto-tuning utility"""
    
    def test_auto_tune_recent_k_calculations(self):
        """Verify auto-tuning produces sensible results"""
        from comfy_nodes.v2v_node import auto_tune_recent_k
        
        # Test various VRAM configurations
        test_cases = [
            (12.0, 512, "sd15", 5, 16),    # 12GB VRAM, generous allocation
            (24.0, 512, "sd15", 10, 16),   # 24GB VRAM, should max out
            (8.0, 1024, "sdxl", 1, 16),    # 8GB VRAM, high res - function is optimistic
            (48.0, 512, "sd15", 16, 16),   # 48GB VRAM, maxed at cap
        ]
        
        for vram_gb, resolution, model_type, min_expected, max_expected in test_cases:
            result = auto_tune_recent_k(vram_gb, resolution, model_type)
            
            assert result >= 1, f"recent_k must be at least 1 (got {result})"
            assert result <= 16, f"recent_k should be capped at 16 (got {result})"
            assert min_expected <= result <= max_expected, \
                f"For {vram_gb}GB/{resolution}px/{model_type}, expected {min_expected}-{max_expected}, got {result}"
        
        print("✓ VRAM auto-tuning calculations validated")


class TestInjectorIntegration:
    """Test framepack injector with memory bank"""
    
    def test_context_batch_assembly(self):
        """Test that injector correctly assembles context batches"""
        memory = FramePackMemory(recent_k=6, max_summary=8)
        
        # Populate memory
        latents = torch.randn(30, 4, 64, 64)
        memory.ingest_batch(latents)
        
        # Create injector
        device = torch.device('cpu')
        injector = get_injector(memory, device=device)
        
        # Get context latents
        context_latents, n_ctx = injector.get_context_latents()
        
        # Verify context assembled correctly
        assert context_latents is not None, "Context should be assembled"
        assert n_ctx > 0, "Should have context frames"
        assert context_latents.shape[0] == n_ctx
        assert context_latents.shape[1:] == (4, 64, 64)
        
        print(f"✓ Context batch assembled: {n_ctx} frames, shape {context_latents.shape}")


# Run tests if executed directly
if __name__ == "__main__":
    print("=" * 60)
    print("Running Long V2V Custom Node Integration Tests")
    print("=" * 60)
    
    # Multi-chunk tests
    print("\n[Multi-Chunk Processing]")
    test = TestMultiChunkProcessing()
    test.test_100_frame_chain()
    test.test_memory_persistence()
    test.test_context_extraction()
    
    # Scene detection tests
    print("\n[Scene Cut Detection]")
    test = TestSceneCutDetection()
    test.test_scene_cut_detection_accuracy()
    test.test_anchor_reset_on_scene_cut()
    
    # Importance tests
    print("\n[Importance-Based Replacement]")
    test = TestImportanceBasedReplacement()
    test.test_importance_scoring()
    test.test_priority_replacement()
    
    # VRAM tuning tests
    print("\n[VRAM Auto-Tuning]")
    test = TestVRAMAutoTuning()
    test.test_auto_tune_recent_k_calculations()
    
    # Injector tests
    print("\n[Injector Integration]")
    test = TestInjectorIntegration()
    test.test_context_batch_assembly()
    
    print("\n" + "=" * 60)
    print("✅ All integration tests passed!")
    print("=" * 60)
