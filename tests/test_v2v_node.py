import sys
import os
import unittest
from unittest.mock import MagicMock
import torch

# ----------------------------------------------------------------------
# 1. SETUP THE MOCK ENVIRONMENT (Critical Step)
# ----------------------------------------------------------------------
# We must mock 'comfy' and 'nodes' BEFORE importing the custom node.

mock_comfy = MagicMock()
mock_nodes = MagicMock()

sys.modules["comfy"] = mock_comfy
sys.modules["comfy.sd"] = mock_comfy.sd
sys.modules["comfy.sample"] = mock_comfy.sample
sys.modules["comfy.samplers"] = mock_comfy.samplers
sys.modules["comfy.model_management"] = mock_comfy.model_management
sys.modules["nodes"] = mock_nodes

# ----------------------------------------------------------------------
# 2. SETUP PATH & IMPORT
# ----------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOW we can import the node
from comfy_nodes.v2v_node import AnimeFramePackNode

# ----------------------------------------------------------------------
# 3. THE TESTS
# ----------------------------------------------------------------------

class TestV2VNode(unittest.TestCase):
    
    def setUp(self):
        """Runs before every test"""
        self.node = AnimeFramePackNode()
        
    def test_bypass_mode_simple(self):
        """Test that Bypass Mode returns dimmed image and mock history"""
        print("\n--- Testing Bypass Mode ---")
        
        # Inputs: 1 Frame, 64x64, 3 channels (Standard Image format)
        image_batch = torch.ones((1, 64, 64, 3))
        
        # Call Node in Bypass Mode with CORRECT parameters
        out_images, out_history = self.node.process_frames(
            model=None, 
            vae=None,
            image_batch=image_batch,
            seed=0,
            steps=20,
            cfg=7.5,
            denoise=0.6,
            recent_k=6,
            bypass_mode=True,
            auto_tune_vram=False,
            enable_scene_detection=False,
            positive=None,
            negative=None,
            prev_history=None
        )
        
        # Verification
        # 1. Image should be dimmed (multiplied by 0.5)
        self.assertTrue(torch.allclose(out_images, image_batch * 0.5))
        
        # 2. History should be FramePackMemory with status
        from core.framepack.memory_bank import FramePackMemory
        self.assertIsInstance(out_history, FramePackMemory)
        self.assertEqual(out_history.meta["status"], "mock_data")
        
        print("✅ Bypass Mode: Passed")

    def test_bypass_mode_chaining(self):
        """Test that Node B can read history from Node A in bypass"""
        print("\n--- Testing Bypass Chaining ---")
        
        from core.framepack.memory_bank import FramePackMemory
        
        # Node A: First call creates initial history
        image_batch_A = torch.zeros((2, 64, 64, 3))
        _, history_A = self.node.process_frames(
            model=None, vae=None,
            image_batch=image_batch_A,
            seed=0, steps=20, cfg=7.5, denoise=0.6, recent_k=6,
            bypass_mode=True,
            prev_history=None
        )
        
        # Verify first call
        self.assertIsInstance(history_A, FramePackMemory)
        initial_step = history_A.meta.get("step_count", 0)
        
        # Node B: Second call uses history from A
        image_batch_B = torch.zeros((3, 64, 64, 3))
        _, history_B = self.node.process_frames(
            model=None, vae=None,
            image_batch=image_batch_B,
            seed=0, steps=20, cfg=7.5, denoise=0.6, recent_k=6,
            bypass_mode=True,
            prev_history=history_A  # <--- Connecting the chain
        )
        
        # Verification: step count should increment
        self.assertEqual(history_B.meta["step_count"], initial_step + 1)
        print("✅ History Chaining: Passed")

    def test_bypass_preserves_memory_object(self):
        """Test that bypass mode preserves FramePackMemory instance"""
        print("\n--- Testing Memory Preservation ---")
        
        from core.framepack.memory_bank import FramePackMemory
        
        image_batch = torch.randn((4, 128, 128, 3))
        _, history = self.node.process_frames(
            model=None, vae=None,
            image_batch=image_batch,
            seed=0, steps=20, cfg=7.5, denoise=0.6, recent_k=6,
            bypass_mode=True
        )
        
        # Verify it's a FramePackMemory instance
        self.assertIsInstance(history, FramePackMemory)
        self.assertEqual(history.recent_k, 6)
        print("✅ Memory Preservation: Passed")

    def test_auto_tune_parameter_exists(self):
        """Test that auto_tune_vram parameter is accepted"""
        print("\n--- Testing Auto-Tune Parameter ---")
        
        image_batch = torch.ones((1, 64, 64, 3))
        
        # Should not raise error with auto_tune_vram=True
        _, history = self.node.process_frames(
            model=None, vae=None,
            image_batch=image_batch,
            seed=0, steps=20, cfg=7.5, denoise=0.6, recent_k=6,
            bypass_mode=True,
            auto_tune_vram=True  # Should be accepted
        )
        
        from core.framepack.memory_bank import FramePackMemory
        self.assertIsInstance(history, FramePackMemory)
        print("✅ Auto-Tune Parameter: Passed")

    def test_scene_detection_parameter_exists(self):
        """Test that enable_scene_detection parameter is accepted"""
        print("\n--- Testing Scene Detection Parameter ---")
        
        image_batch = torch.ones((1, 64, 64, 3))
        
        # Should not raise error with enable_scene_detection=True
        _, history = self.node.process_frames(
            model=None, vae=None,
            image_batch=image_batch,
            seed=0, steps=20, cfg=7.5, denoise=0.6, recent_k=6,
            bypass_mode=True,
            enable_scene_detection=True  # Should be accepted
        )
        
        from core.framepack.memory_bank import FramePackMemory
        self.assertIsInstance(history, FramePackMemory)
        print("✅ Scene Detection Parameter: Passed")

if __name__ == '__main__':
    unittest.main()