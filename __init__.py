# -*- coding: utf-8 -*-
"""
Wan Video Reference Nodes
Multi-frame reference conditioning for Wan2.2 A14B I2V models

Nodes:
1. WanFirstMiddleLastFrameToVideo - 3-frame reference with flexible positioning
2. WanMultiFrameRefToVideo - N-frame universal reference node
3. WanFourFrameReferenceUltimate - 4-frame reference with adjustable placeholder
4. WanAdvancedI2V - Ultimate unified node with all features (includes automatic chaining)
5. WanMultiImageLoader - Load multiple images with UI for batch selection and preview
"""

from .wan_first_middle_last import WanFirstMiddleLastFrameToVideo
from .wan_multi_frame import WanMultiFrameRefToVideo
from .wan_multi_image_loader import WanMultiImageLoader

# 4-frame reference node
try:
    from .wan_4_frame_ultimate import WanFourFrameReferenceUltimate
    HAS_4FRAME = True
except ImportError:
    HAS_4FRAME = False
    print("wan_4_frame_ultimate.py not found")

# Advanced unified node with complete feature set (includes automatic chaining)
try:
    from .wan_advanced_i2v import (
        WanAdvancedI2V,
        WanAdvancedExtractLastFrames,
        WanAdvancedExtractLastImages
    )
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False
    print("wan_advanced_i2v.py not found")


NODE_CLASS_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo": WanFirstMiddleLastFrameToVideo,
    "WanMultiFrameRefToVideo": WanMultiFrameRefToVideo,
    "WanMultiImageLoader": WanMultiImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo": "Wan First-Middle-Last Frame",
    "WanMultiFrameRefToVideo": "Wan Multi-Frame Reference",
    "WanMultiImageLoader": "Wan Multi-Image Loader",
}

if HAS_4FRAME:
    NODE_CLASS_MAPPINGS["WanFourFrameReferenceUltimate"] = WanFourFrameReferenceUltimate
    NODE_DISPLAY_NAME_MAPPINGS["WanFourFrameReferenceUltimate"] = "Wan 4-Frame Reference"

if HAS_ADVANCED:
    NODE_CLASS_MAPPINGS["WanAdvancedI2V"] = WanAdvancedI2V
    NODE_CLASS_MAPPINGS["WanAdvancedExtractLastFrames"] = WanAdvancedExtractLastFrames
    NODE_CLASS_MAPPINGS["WanAdvancedExtractLastImages"] = WanAdvancedExtractLastImages
    
    NODE_DISPLAY_NAME_MAPPINGS["WanAdvancedI2V"] = "Wan Advanced I2V (Ultimate)"
    NODE_DISPLAY_NAME_MAPPINGS["WanAdvancedExtractLastFrames"] = "Wan Extract Last Frames (Latent)"
    NODE_DISPLAY_NAME_MAPPINGS["WanAdvancedExtractLastImages"] = "Wan Extract Last Images"

# Web directory for frontend JS
import os
WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
