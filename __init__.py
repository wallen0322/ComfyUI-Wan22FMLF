# -*- coding: utf-8 -*-
"""
Wan Video Reference Nodes
Multi-frame reference conditioning for Wan2.2 A14B I2V models

Nodes:
1. WanFirstMiddleLastFrameToVideo - 3-frame reference with flexible positioning
2. WanMultiFrameRefToVideo - N-frame universal reference node
"""

from .wan_first_middle_last import WanFirstMiddleLastFrameToVideo
from .wan_multi_frame import WanMultiFrameRefToVideo


NODE_CLASS_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo": WanFirstMiddleLastFrameToVideo,
    "WanMultiFrameRefToVideo": WanMultiFrameRefToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo": "Wan First-Middle-Last Frame 🎬",
    "WanMultiFrameRefToVideo": "Wan Multi-Frame Reference 🎞️",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
