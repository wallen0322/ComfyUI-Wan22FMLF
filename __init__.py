# -*- coding: utf-8 -*-
"""
Wan Video Reference Nodes
Multi-frame reference conditioning for Wan2.2 A14B I2V models

Nodes:
1. WanFirstMiddleLastFrameToVideo - 3-frame reference with flexible positioning
2. WanMultiFrameRefToVideo - N-frame universal reference node
3. WanFourFrameReferenceUltimate - 4-frame reference with adjustable placeholder
4. WanAdvancedI2V - Ultimate unified node with all features (includes automatic chaining)
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension

from .wan_first_middle_last import WanFirstMiddleLastFrameToVideo
from .wan_multi_frame import WanMultiFrameRefToVideo

HAS_4FRAME = False
HAS_ADVANCED = False

try:
    from .wan_4_frame_ultimate import WanFourFrameReferenceUltimate
    HAS_4FRAME = True
except ImportError:
    print("wan_4_frame_ultimate.py not found")

try:
    from .wan_advanced_i2v import (
        WanAdvancedI2V,
        WanAdvancedExtractLastFrames,
        WanAdvancedExtractLastImages
    )
    HAS_ADVANCED = True
except ImportError:
    print("wan_advanced_i2v.py not found")


class WanVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        nodes = [
            WanFirstMiddleLastFrameToVideo,
            WanMultiFrameRefToVideo,
        ]
        
        if HAS_4FRAME:
            nodes.append(WanFourFrameReferenceUltimate)
        
        if HAS_ADVANCED:
            nodes.extend([
                WanAdvancedI2V,
                WanAdvancedExtractLastFrames,
                WanAdvancedExtractLastImages,
            ])
        
        return nodes


async def comfy_entrypoint():
    return WanVideoExtension()
