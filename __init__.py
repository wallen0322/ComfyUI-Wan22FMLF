# -*- coding: utf-8 -*-
"""
Wan Video Reference Nodes
Multi-frame reference conditioning for Wan2.2 A14B I2V models

Nodes:
1. WanFirstMiddleLastFrameToVideo - 3-frame reference with flexible positioning
2. WanMultiFrameRefToVideo - N-frame universal reference node
3. WanFourFrameReferenceUltimate - 4-frame reference with adjustable placeholder
"""

from .wan_first_middle_last import WanFirstMiddleLastFrameToVideo
from .wan_multi_frame import WanMultiFrameRefToVideo

# 新增: 4帧参考节点
try:
    from .wan_4_frame_ultimate import WanFourFrameReferenceUltimate
    HAS_4FRAME = True
except ImportError:
    HAS_4FRAME = False
    print("wan_4_frame_ultimate.py not found")


NODE_CLASS_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo": WanFirstMiddleLastFrameToVideo,
    "WanMultiFrameRefToVideo": WanMultiFrameRefToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo": "Wan First-Middle-Last Frame",
    "WanMultiFrameRefToVideo": "Wan Multi-Frame Reference",
}

if HAS_4FRAME:
    NODE_CLASS_MAPPINGS["WanFourFrameReferenceUltimate"] = WanFourFrameReferenceUltimate
    NODE_DISPLAY_NAME_MAPPINGS["WanFourFrameReferenceUltimate"] = "Wan 4-Frame Reference"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
