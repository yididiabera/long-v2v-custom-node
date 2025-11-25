from .v2v_node import AnimeFramePackNode

NODE_CLASS_MAPPINGS = {
    "AnimeFramePackNode": AnimeFramePackNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimeFramePackNode": "Anime FramePack (Vid2Vid)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]