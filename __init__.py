from typing_extensions import override
from comfy_api.latest import ComfyExtension

from .wan_first_middle_last import WanFirstMiddleLastFrameToVideo
from .wan_multi_frame import WanMultiFrameRefToVideo
from .wan_multi_image_loader import WanMultiImageLoader

WEB_DIRECTORY = "./js"

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
            WanMultiImageLoader,
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


__all__ = ['WEB_DIRECTORY']
