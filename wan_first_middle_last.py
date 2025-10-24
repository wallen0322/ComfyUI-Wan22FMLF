# -*- coding: utf-8 -*-
"""
Wan First-Middle-Last Frame Node
- Support 3-frame reference with flexible positioning
- Adjustable strength for middle frame
"""

import torch
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
from nodes import MAX_RESOLUTION


class WanFirstMiddleLastFrameToVideo:
    """
    3-frame reference node for Wan2.2 A14B I2V.
    
    Features:
    - First, middle, and last frame reference
    - Flexible middle frame positioning
    - Adjustable strength for middle frame constraint
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "middle_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "middle_frame_ratio": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Middle frame position (0.5=center)"
                }),
                "middle_frame_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Middle frame constraint strength. 0=no constraint, 0.5=balanced (recommended), 1=fully fixed"
                }),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_middle_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "ComfyUI-Wan22FMLF/video"
    DESCRIPTION = "Generate video with first, middle, and last frame references."

    def generate(self, positive, negative, vae, width, height, length, batch_size,
                 start_image=None, middle_image=None, end_image=None,
                 middle_frame_ratio=0.5, middle_frame_strength=0.5,
                 clip_vision_start_image=None, clip_vision_middle_image=None, 
                 clip_vision_end_image=None):
        
        # Get VAE parameters
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        
        device = comfy.model_management.intermediate_device()
        
        # Create empty latent
        latent = torch.zeros([batch_size, latent_channels, latent_t, 
                             height // spacial_scale, width // spacial_scale], 
                             device=device)
        
        # Resize images
        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, 
                "bilinear", "center").movedim(1, -1)
        
        if middle_image is not None:
            middle_image = comfy.utils.common_upscale(
                middle_image[:1].movedim(-1, 1), width, height, 
                "bilinear", "center").movedim(1, -1)
        
        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-length:].movedim(-1, 1), width, height, 
                "bilinear", "center").movedim(1, -1)
        
        # Create timeline
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), 
                         device=device)
        
        # Calculate middle frame position
        middle_idx = int(length * middle_frame_ratio)
        middle_idx = max(1, min(middle_idx, length - 2))
        
        # Place reference frames
        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask[:, :, :start_image.shape[0] + 3] = 0.0
        
        if middle_image is not None:
            image[middle_idx:middle_idx + 1] = middle_image
            # Middle frame uses adjustable strength
            mask_value = 1.0 - middle_frame_strength
            mask[:, :, middle_idx:middle_idx + 4] = mask_value
        
        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            mask[:, :, -end_image.shape[0]:] = 0.0
        
        print(f"📌 Reference frames: start at 0, middle at {middle_idx} (strength={middle_frame_strength:.2f}), end at {length-1}")
        
        # Encode timeline
        concat_latent_image = vae.encode(image[:, :, :, :3])
        
        # Reshape mask
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        
        # Set conditioning
        positive = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask
        })
        negative = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask
        })
        
        # CLIP Vision
        clip_vision_output = self._merge_clip_vision_outputs(
            clip_vision_start_image, 
            clip_vision_middle_image, 
            clip_vision_end_image
        )
        
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, 
                                                           {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, 
                                                           {"clip_vision_output": clip_vision_output})
        
        out_latent = {"samples": latent}
        return (positive, negative, out_latent)

    def _merge_clip_vision_outputs(self, *outputs):
        """Merge multiple CLIP Vision outputs."""
        valid_outputs = [o for o in outputs if o is not None]
        
        if not valid_outputs:
            return None
        
        if len(valid_outputs) == 1:
            return valid_outputs[0]
        
        all_states = [o.penultimate_hidden_states for o in valid_outputs]
        combined_states = torch.cat(all_states, dim=-2)
        
        result = comfy.clip_vision.Output()
        result.penultimate_hidden_states = combined_states
        return result


NODE_CLASS_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo": WanFirstMiddleLastFrameToVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo": "Wan First-Middle-Last Frame 🎬"
}
