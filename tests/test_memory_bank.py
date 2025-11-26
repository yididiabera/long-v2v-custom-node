import torch
import pytest
import sys
import os
from rich.console import Console
from rich.panel import Panel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.framepack.memory_bank import FramePackMemory
from core.framepack.compressor import spatial_pool
from core.framepack.importance import batch_importance_scores

console = Console()

@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Setup test session with rich status display"""
    print() 
    console.print(Panel.fit(
        "[bold blue]Starting FramePack Memory Bank Tests[/bold blue]\n"
        "[dim]Testing memory management, compression, and importance scoring...[/dim]",
        title="Test Session",
        border_style="blue"
    ))
    print()  
    yield
    print()  
    console.print(Panel.fit(
        "[green]✓ All FramePack Memory Bank Tests Completed![/green]",
        title="Test Session Complete",
        border_style="green"
    ))
    print()

def test_memory_initialization():
    console.print("[blue]Testing memory initialization...[/blue]")
    with console.status("Initializing FramePackMemory", spinner="dots"):
        mem = FramePackMemory(recent_k=4)
        assert mem.anchor_latents is None
        assert mem.recent_latents == []
        assert mem.total_frames_processed == 0
        assert mem.update_step == 0
    console.print("[green]✓ Memory initialization test passed[/green]")

def test_memory_ingest_and_anchor():
    console.print("[blue]Testing memory ingestion and anchoring...[/blue]")
    with console.status("Processing batch ingestion", spinner="dots"):
        mem = FramePackMemory(recent_k=4)
        batch = torch.randn(3, 4, 64, 64)

        mem.ingest_batch(batch)

        assert mem.anchor_latents is not None
        assert mem.anchor_latents.shape == (1, 4, 64, 64)
        assert len(mem.recent_latents) == 3
        assert mem.total_frames_processed == 3
    console.print("[green]✓ Memory ingestion and anchoring test passed[/green]")

def test_window_trimming():
    console.print("[blue]Testing window trimming...[/blue]")
    with console.status("Testing window size management", spinner="dots"):
        mem = FramePackMemory(recent_k=2)
        batch1 = torch.randn(3, 4, 64, 64)

        mem.ingest_batch(batch1)

        assert len(mem.recent_latents) == 2
        assert len(mem.summary_tokens) > 0
    console.print("[green]✓ Window trimming test passed[/green]")

def test_context_return_shapes():
    console.print("[blue]Testing context return shapes...[/blue]")
    with console.status("Validating context tensor shapes", spinner="dots"):
        mem = FramePackMemory(recent_k=3)
        batch = torch.randn(3, 4, 64, 64)
        mem.ingest_batch(batch)

        anchor, recent, summary = mem.get_context_for_sampler()

        assert anchor.shape == (1, 4, 64, 64)
        if recent is not None:
            assert recent.dim() == 4
            assert recent.shape[0] == 3
    console.print("[green]✓ Context return shapes test passed[/green]")

def test_spatial_pool_shape():
    console.print("[blue]Testing spatial pooling...[/blue]")
    with console.status("Testing spatial downsampling", spinner="dots"):
        x = torch.randn(1, 4, 64, 64)

        pooled = spatial_pool(x, target_hw=(16, 16))

        assert pooled.shape == (1, 4, 16, 16)
    console.print("[green]✓ Spatial pool shape test passed[/green]")

def test_importance_score():
    console.print("[blue]Testing importance scoring...[/blue]")
    with console.status("Computing importance scores", spinner="dots"):
        x = torch.randn(2, 4, 64, 64)
        scores = batch_importance_scores(x)

        assert isinstance(scores, list)
        assert len(scores) == 2
        assert isinstance(scores[0], float)
    console.print("[green]✓ Importance score test passed[/green]")

def test_memory_persistence(tmp_path):
    console.print("[blue]Testing memory persistence (save/load)...[/blue]")
    with console.status("Testing save/load operations", spinner="dots"):
        # Setup memory
        mem = FramePackMemory(recent_k=2)
        batch = torch.randn(1, 4, 64, 64)
        mem.ingest_batch(batch)
        
        # Save
        save_dir = tmp_path / "checkpoints"
        prefix = str(save_dir / "mem_test")
        paths = mem.save(prefix)
        
        assert os.path.exists(paths['meta'])
        assert os.path.exists(paths['pt'])
        
        # Load
        loaded_mem = FramePackMemory.load(prefix)
        
        assert loaded_mem.total_frames_processed == mem.total_frames_processed
        assert len(loaded_mem.recent_latents) == len(mem.recent_latents)
        # Check tensor content equality
        assert torch.allclose(loaded_mem.recent_latents[0], mem.recent_latents[0])
        
    console.print("[green]✓ Memory persistence test passed[/green]")

def test_summary_overflow_and_merge():
    console.print("[blue]Testing summary overflow and merging...[/blue]")
    with console.status("Testing summary limit logic", spinner="dots"):
        # Max summary of 2 to trigger overflow quickly
        mem = FramePackMemory(recent_k=1, max_summary=2, summary_hw=(8,8))
        
        # Ingest 1st batch -> becomes anchor, added to recent
        mem.ingest_batch(torch.randn(1, 4, 32, 32))
        
        # Ingest 2nd batch -> 1st moves to summary (count=1)
        mem.ingest_batch(torch.randn(1, 4, 32, 32))
        assert len(mem.summary_tokens) == 1
        
        # Ingest 3rd batch -> 2nd moves to summary (count=2) [FULL]
        mem.ingest_batch(torch.randn(1, 4, 32, 32))
        assert len(mem.summary_tokens) == 2
        
        # Ingest 4th batch -> 3rd moves to summary -> Trigger Merge on index 0
        # We want to verify that we didn't just append (len would be 3)
        mem.ingest_batch(torch.randn(1, 4, 32, 32))
        
        assert len(mem.summary_tokens) == 2 # Should stay at max
        
    console.print("[green]✓ Summary overflow test passed[/green]")

def test_utility_methods():
    console.print("[blue]Testing utility methods...[/blue]")
    with console.status("Testing to_dict and create_empty", spinner="dots"):
        mem = FramePackMemory(recent_k=2)
        mem.ingest_batch(torch.randn(1, 4, 32, 32))
        
        # Test to_dict
        data = mem.to_dict()
        assert isinstance(data, dict)
        assert 'total_frames_processed' in data
        assert data['total_frames_processed'] == 1
        
        # Test create_empty
        empty_mem = FramePackMemory.create_empty(recent_k=5)
        assert empty_mem.recent_k == 5
        assert empty_mem.total_frames_processed == 0
        
    console.print("[green]\u2713 Utility methods test passed[/green]")


def test_summary_overflow_context_shapes():
    console.print("[blue]Testing context shapes after summary overflow...[/blue]")
    with console.status("Validating context after repeated ingests", spinner="dots"):
        # Configure small windows to trigger summary merges quickly
        mem = FramePackMemory(recent_k=1, max_summary=2, summary_hw=(8, 8))

        # Ingest multiple batches to force both summary fill and merge path
        for _ in range(6):
            mem.ingest_batch(torch.randn(1, 4, 32, 32))

        # get_context_for_sampler should not raise and must return stackable summary
        anchor, recent, summary = mem.get_context_for_sampler()

        assert anchor is not None
        if summary is not None:
            assert summary.dim() == 4
            # Batch dimension should correspond to number of summary tokens (capped by max_summary)
            assert summary.shape[0] == len(mem.summary_tokens)
    console.print("[green]\u2713 Summary overflow context shapes test passed[/green]")