
# ComfyUI Long Video to Video Custom Node
**Long-Form Video-to-Video Generation with Training-Free Memory Injection**

![Version](https://img.shields.io/badge/Version-v0-blue) ![ComfyUI](https://img.shields.io/badge/Platform-ComfyUI-green)

### Summary
**ComfyUI-Anime-FramePack** is a custom node suite designed to solve the "Amnesia Problem" in AI video generation. It enables **infinite-length video-to-video processing** while maintaining character identity and styling—**without requiring any model training, LoRAs, or adapters.**

By implementing the **FramePack Algorithm** via **Zero-Weight Context Batching**, this node allows standard Stable Diffusion models (SD1.5, SDXL, AnimateDiff) to "see" past frames during generation, ensuring that *Frame 100* looks like *Frame 0*.

---

## Key Features

*   **3-Tier Memory Bank:** Automatically manages **Anchor Frames** (Identity), **Recent History** (Motion), and **Summary Context** (Long-term style) to prevent drift.
*   **Zero-Weight Injection:** Uses **Context Batching** to inject memory directly into the UNet's input stream. No LoRAs, ControlNets, or IP-Adapters required.
*   **Scene Detection:** Automatically detects scene cuts and resets anchor frames to maintain identity across scene transitions.
*   **VRAM Auto-Tuning:** Intelligently calculates optimal memory usage based on your GPU's available VRAM.
*   **Importance-Based Memory:** Prioritizes high-quality frames in summary memory using edge salience and variance scoring.
*   **Hardware Optimized:** Designed for **RTX 3090 (24GB)** workflows. Heavily utilizes CPU offloading for memory storage, only moving active context to VRAM during inference.
*   **Infinite Chaining:** Nodes pass a lightweight `FRAMEPACK_MEM` state object, allowing you to process videos in chunks (e.g., 16 frames at a time) indefinitely.
*   **Mock / Bypass Mode:** Develop workflows on a laptop (CPU-only). The node simulates data flow without loading heavy models, preventing crashes on low-end hardware.

---

## Installation

1.  **Navigate to your ComfyUI custom nodes directory:**
    ```bash
    cd /path/to/ComfyUI/custom_nodes/
    ```
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ComfyUI-Anime-FramePack.git
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r ComfyUI-Anime-FramePack/requirements.txt
    ```
4.  **Restart ComfyUI.**

---

##  The FramePack Architecture

This project implements a specific engineering solution to the **Long-Video Consistency** problem.

### 1. The Memory System
Standard video models lose context after ~24 frames. We implement a persistent state machine:
*   **Anchor Frame:** The very first frame of the scene. The model refers to this constantly to lock facial features and clothing details.
*   **Recent Window:** A sliding FIFO buffer (last `K` frames). Ensures smooth motion flow.
*   **Summary Tokens:** Spatially compressed tokens from older history, preventing long-term style drift.

### 2. The Injection Strategy: Context Batching
We evaluated multiple methods (Projectors, Optical Flow, ControlNet) and selected **Context Batching**:

> **How it works:** Instead of training an adapter to "translate" memory, we concatenate the Anchor and Summary frames to the *Input Batch*. We patch the Model's sampler to force the current generation to attend to these reference pixels. This works natively with pre-trained Checkpoints.

---

## Usage & Workflow

### Node Name: `Anime FramePack (Vid2Vid)`

#### Basic Inputs
*   **model / vae:** Connect your standard Checkpoint loaders.
*   **image_batch:** The chunk of frames you want to process (e.g., loaded via `Load Video` or `Load Image Sequence`).
*   **recent_k:** Size of the sliding window (Default: 6). Higher = smoother motion, more VRAM.

#### The "Infinite Chain" Workflow
To generate long videos, chain multiple nodes together:

1.  **Chunk 1 (Frames 0-16):**
    *   Feed images into **Node A**.
    *   *Input `prev_history`:* Empty (None).
    *   *Output:* `processed_images` and `history_state`.
2.  **Chunk 2 (Frames 17-32):**
    *   Feed images into **Node B**.
    *   *Input `prev_history`:* Connect **Node A**'s `history_state`.
    *   *Result:* Node B "remembers" what Node A generated.

#### Parameters Reference

| Parameter | Default | Range | Description |
| :--- | :--- | :--- | :--- |
| **model** | Required | - | Your SD1.5/SDXL checkpoint |
| **vae** | Required | - | VAE for encoding/decoding |
| **image_batch** | Required | - | Input frames (B, H, W, 3) |
| **seed** | `0` | Any int | Random seed for sampling |
| **steps** | `20` | 1-150 | Sampling steps (20-30 recommended) |
| **cfg** | `7.5` | 1.0-30.0 | Classifier-free guidance scale |
| **denoise** | `0.6` | 0.0-1.0 | V2V strength (0.5-0.7 for consistency) |
| **recent_k** | `6` | 1-32 | Recent frame window size |
| **bypass_mode** | `True` | bool | Enable for workflow testing (no GPU) |
| **auto_tune_vram** | `False` | bool | Auto-calculate optimal recent_k |
| **enable_scene_detection** | `False` | bool | Auto-reset anchor on scene cuts |
| **positive** | Optional | - | Positive conditioning (prompt) |
| **negative** | Optional | - | Negative conditioning (prompt) |
| **prev_history** | Optional | - | Memory state from previous chunk |

---

## Advanced Features & Usage Examples

### Scene Detection

**When to use:** Videos with scene changes (indoor→outdoor, day→night, character changes)

**How to enable:**
```python
# In your ComfyUI workflow:
enable_scene_detection = True
```

**What it does:**
- Detects scene cuts using MSE (Mean Squared Error) between frames
- Automatically resets the anchor frame when a scene cut is detected
- Prevents identity drift across scene boundaries
- Default threshold: 0.3 (configurable in `memory_bank.py`)

**Expected behavior:**
```
Scene 1 (Indoor, frames 0-50):
  └─ Anchor: Frame 0 (character in room)
  
Scene Cut Detected at Frame 51 (Outdoor):
  └─ Anchor reset to Frame 51 (character outside)
  └─ Memory preserved, but identity reference updated
```

**Use cases:**
- Music videos with multiple locations
- Anime episodes with scene transitions
- Narrative videos with cuts between settings

---

### VRAM Auto-Tuning

**When to use:**
- First time running on new hardware
- Switching between SD1.5 and SDXL
- Changing video resolution
- Preventing Out-Of-Memory errors

**How to enable:**
```python
# In your ComfyUI workflow:
auto_tune_vram = True
recent_k = 6  # This will be overridden by auto-tuning
```

**What it does:**
- Estimates base model VRAM usage (4GB for SD1.5, 6.5GB for SDXL)
- Calculates activation memory based on resolution
- Determines optimal `recent_k` to maximize quality without OOM
- Clamps result between 1-16 frames

**Expected behavior by hardware:**

| GPU | VRAM | Resolution | Model | Auto-tuned recent_k |
|-----|------|------------|-------|--------------------|
| RTX 3060 | 12GB | 512px | SD1.5 | 8-12 |
| RTX 3090 | 24GB | 512px | SD1.5 | 16 (max) |
| RTX 4090 | 24GB | 1024px | SDXL | 6-8 |
| RTX 3060 | 12GB | 1024px | SDXL | 2-4 |

**Manual override:**
If auto-tuning is too conservative or aggressive, disable it and set `recent_k` manually.

---

### Parameter Tuning Guide

#### **denoise** (Video Consistency)

| Value | Effect | Use Case |
|-------|--------|----------|
| `0.3-0.4` | Very tight consistency, minimal changes | Static scenes, subtle motion |
| `0.5-0.6` | **Recommended** - Balanced | Most anime/video workflows |
| `0.7-0.8` | More creative freedom | Dynamic action, style changes |
| `0.9-1.0` | Near-random generation | Experimental, heavy transformation |

#### **recent_k** (Motion Smoothness)

| Value | VRAM Impact | Motion Quality | When to Use |
|-------|-------------|----------------|-------------|
| `1-3` | Low | Choppy | VRAM-constrained (8GB) |
| `4-6` | **Medium** | **Smooth** | **Recommended default** |
| `8-12` | High | Very smooth | High-motion scenes, 24GB VRAM |
| `16+` | Very high | Silky smooth | Extreme cases, 48GB VRAM |

#### **cfg** (Prompt Adherence)

| Value | Effect |
|-------|--------|
| `3-5` | Subtle guidance, more natural |
| `7-8` | **Recommended** - Strong adherence |
| `10-15` | Very strong, may oversaturate |

---

### Workflow Examples

#### Example 1: Simple 100-Frame Video (No Scene Changes)

```
[Load Video (frames 0-99)] → [Batch into 5 chunks of 20]
                                      ↓
                              [FramePack Node A]
                              - bypass_mode: False
                              - recent_k: 6
                              - denoise: 0.6
                              - auto_tune_vram: True
                              - enable_scene_detection: False
                                      ↓
                              [Output: 100 frames]
```

#### Example 2: Multi-Scene Video (Scene Detection)

```
[Load Video] → [FramePack Node]
               - enable_scene_detection: True  ← KEY SETTING
               - denoise: 0.55
               - recent_k: 8
                      ↓
               Automatic anchor reset on cuts
                      ↓
               [Output: Consistent within scenes]
```

#### Example 3: Long Video Chain (1000+ frames)

```
[Chunk 1: 0-50]   → [Node A] → history_A
[Chunk 2: 51-100] → [Node B] ← history_A → history_B
[Chunk 3: 101-150]→ [Node C] ← history_B → history_C
                        ...
[Chunk 20: 951-1000] → [Node T] ← history_S → final
```

**Settings for long chains:**
- `denoise: 0.5-0.6` (tighter consistency)
- `recent_k: 8-12` (more temporal context)
- `enable_scene_detection: True` (handle cuts)

---

## Testing & Validation

This project includes a comprehensive test suite to ensure stability.

**Running Tests:**
```bash
# From the custom node root directory
pytest tests/                          # Run all tests
python tests/test_memory_bank.py       # Memory management
python tests/test_framepack_injector.py # Context batching
python tests/test_v2v_node.py          # Node logic
python tests/test_integration.py       # End-to-end workflows
```

---

## Common Issues / FAQ

**Q: I get "Dimension Mismatch" errors.**  
A: Ensure your input video resolution is consistent. The node handles resolution changes automatically via interpolation, but extreme aspect ratio changes may fail. Keep resolution constant within a single video.

**Q: My VRAM usage is too high / Out of Memory errors.**  
A: Try these solutions in order:
1. Enable `auto_tune_vram=True` (automatic optimization)
2. Manually reduce `recent_k` (try 4 instead of 6)
3. Lower your video resolution (768→512)
4. Use SD1.5 instead of SDXL

**Q: The node crashes on my laptop.**  
A: Enable `bypass_mode=True`. Production mode requires ~12GB+ VRAM (SD1.5) or ~20GB+ (SDXL). Bypass mode lets you develop workflows on CPU-only systems.

**Q: Scene detection triggers too often / not enough.**  
A: Adjust the threshold in `core/framepack/memory_bank.py`:
```python
self.scene_cut_threshold = 0.3  # Default
# Lower = more sensitive (0.1-0.2 for subtle cuts)
# Higher = less sensitive (0.5-0.7 for only major cuts)
```

**Q: Video quality degrades over long sequences.**  
A: Try these settings:
- Lower `denoise` to 0.5-0.55 (tighter consistency)
- Increase `recent_k` to 8-12 (more temporal context)
- Enable `enable_scene_detection=True` (prevent drift across cuts)

**Q: Characters change appearance mid-video.**  
A: This indicates anchor drift. Solutions:
- Enable scene detection to reset anchor on cuts
- Use lower denoise values (0.5-0.6)
- Ensure your prompts are consistent across chunks

**Q: How do I know if scene detection is working?**  
A: Check the console logs:
```
[FramePack] Scene cut detected! MSE=0.45 > threshold=0.3
[FramePack] Resetting anchor frame to current frame
```

---

## Performance Tips

- **For best quality:** `denoise=0.6`, `recent_k=8-12`, `steps=25-30`
- **For speed:** `denoise=0.5`, `recent_k=4`, `steps=15-20`
- **For VRAM-constrained:** Enable `auto_tune_vram=True`, use SD1.5, lower resolution
- **For long videos (1000+ frames):** Enable scene detection, use `denoise=0.55`, save checkpoints every 200 frames

---

## Further Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed technical documentation
- **[Integration Tests](tests/test_integration.py)** - Usage examples in code
- **[Memory Bank](core/framepack/memory_bank.py)** - Scene detection implementation
- **[V2V Node](comfy_nodes/v2v_node.py)** - VRAM auto-tuning logic

---