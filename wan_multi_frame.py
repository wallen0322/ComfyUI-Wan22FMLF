# -*- coding: utf-8 -*-
"""
Wan Multi-Frame Reference Node
- Support arbitrary number of reference frames (N-frame)
- Flexible positioning (index or ratio)
- Adjustable strength for middle frames
"""

import torch
import json
from typing import List, Tuple
import node_helpers
import comfy
import comfy.utils
from nodes import MAX_RESOLUTION


class WanMultiFrameRefToVideo:
    """
    Universal N-frame reference node for Wan2.2 A14B I2V.
    
    Features:
    - Support 2, 3, 4, or more reference frames
    - Flexible positioning via indices or ratios
    - Automatic distribution if positions not specified
    - Adjustable strength for non-endpoint frames
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
                "ref_images": ("IMAGE",),
                # 🎨 新增: 占位颜色可调节
                "placeholder_gray_level": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Gray level for placeholder. 0.0=black, 0.5=gray, 1.0=white"
                }),
            },
            "optional": {
                "ref_positions": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Frame indices (e.g., '0,40,80') or ratios (e.g., '0,0.5,1.0'). Leave empty for auto distribution."
                }),
                "ref_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Strength for non-endpoint frames. 0=no constraint, 1=fully fixed. Start/end frames always use 1.0."
                }),
                "fade_frames": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of frames to gradually fade out constraint after each reference frame."
                }),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "ComfyUI-Wan22FMLF/video"
    DESCRIPTION = "Generate video with multiple reference frames at flexible positions."

    def generate(self, positive, negative, vae, width, height, length, batch_size,
                 ref_images, placeholder_gray_level=0.5, ref_positions="", ref_strength=0.5, fade_frames=2, clip_vision_output=None):
        
        # Get VAE parameters
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1  # 🔑 VAE时间压缩比 4:1
        
        device = comfy.model_management.intermediate_device()
        
        # Create empty latent
        latent = torch.zeros([batch_size, latent_channels, latent_t, 
                             height // spacial_scale, width // spacial_scale], 
                             device=device)
        
        print(f"📐 Video: {length} frames → Latent: {latent_t} frames (4:1 compression)")
        
        # Process reference images
        imgs = self._resize_images(ref_images, width, height, device)
        n_imgs = imgs.shape[0]
        
        # Parse positions
        positions = self._parse_positions(ref_positions, n_imgs, length)
        
        # 🧮 智能对齐算法: 确保所有位置对齐到latent边界
        def align_position(pos, total_frames):
            """确保对齐到latent边界(4的倍数)"""
            latent_idx = pos // 4
            aligned_pos = latent_idx * 4
            aligned_pos = max(0, min(aligned_pos, total_frames - 1))
            return aligned_pos
        
        aligned_positions = [align_position(int(p), length) for p in positions]
        
        # Create timeline and mask (使用可调节灰度)
        image = torch.ones((length, height, width, 3), device=device) * placeholder_gray_level
        mask = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), 
                         device=device)
        
        print(f"🔧 Aligned positions: {aligned_positions}")
        print(f"🎨 Placeholder gray level: {placeholder_gray_level:.2f}")
        
        # Place reference images
        for i, pos in enumerate(aligned_positions):
            frame_idx = int(pos)
            image[frame_idx:frame_idx + 1] = imgs[i]
            
            # First and last frames are always fully fixed (strength=1.0)
            # Middle frames use ref_strength for smooth transitions
            is_endpoint = (i == 0) or (i == n_imgs - 1)
            strength = 1.0 if is_endpoint else ref_strength
            mask_value = 1.0 - strength
            
            mask[:, :, frame_idx:frame_idx + 4] = mask_value
            
            # Apply fade-out gradient after reference frame (skip for last frame)
            if fade_frames > 0 and i < n_imgs - 1:
                for f in range(fade_frames):
                    fade_start = frame_idx + 4 + (f * 4)
                    fade_end = fade_start + 4
                    
                    if fade_start >= latent_t * 4:
                        break
                    
                    # Linear fade: gradually release constraint
                    fade_ratio = (f + 1) / (fade_frames + 1)
                    fade_value = mask_value + (1.0 - mask_value) * fade_ratio
                    
                    # Clamp and ensure within bounds
                    fade_end = min(fade_end, latent_t * 4)
                    current_mask = mask[:, :, fade_start:fade_end]
                    # Use maximum to avoid overriding stronger constraints
                    mask[:, :, fade_start:fade_end] = torch.maximum(current_mask, 
                                                                     torch.full_like(current_mask, min(fade_value, 1.0)))
        
        # Encode timeline
        concat_latent_image = vae.encode(image[:, :, :, :3])
        
        # Reshape mask: [1, 1, T*4, H', W'] -> [1, 4, T, H', W']
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
        
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, 
                                                           {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, 
                                                           {"clip_vision_output": clip_vision_output})
        
        out_latent = {"samples": latent}
        return (positive, negative, out_latent)

    def _resize_images(self, images, width, height, device):
        """Resize images to target resolution."""
        images = images.to(device)
        b, h, w, c = images.shape
        
        # Convert to BCHW for resize
        x = images.movedim(-1, 1)
        x = comfy.utils.common_upscale(x, width, height, "bilinear", "center")
        x = x.movedim(1, -1)
        
        # Keep only RGB channels
        if x.shape[-1] == 4:
            x = x[..., :3]
        
        return x

    def _parse_positions(self, pos_str: str, n_imgs: int, length: int) -> List[int]:
        """
        Parse position string into frame indices.
        Supports:
        - Indices: "0,40,80"
        - Ratios: "0,0.5,1.0" (values < 2.0 treated as ratios)
        - JSON: "[0, 40, 80]"
        - Empty: auto distribute evenly
        """
        positions = []
        s = (pos_str or "").strip()
        
        if s:
            try:
                # Try JSON first
                if s.startswith("["):
                    positions = json.loads(s)
                else:
                    # Parse comma-separated values
                    positions = [float(x.strip()) for x in s.split(",") if x.strip()]
            except Exception as e:
                print(f"⚠️ Warning: Failed to parse positions '{pos_str}': {e}")
                positions = []
        
        # Auto distribute if empty or invalid
        if not positions:
            if n_imgs <= 1:
                positions = [0]
            else:
                # Even distribution over [0, length-1]
                positions = [i * (length - 1) / (n_imgs - 1) for i in range(n_imgs)]
            print(f"ℹ️ Auto-distributed {n_imgs} frames: {[int(p) for p in positions]}")
        
        # Convert ratios to indices if needed
        converted_positions = []
        for p in positions:
            if 0 <= p < 2.0:  # Likely a ratio
                converted_positions.append(int(p * (length - 1)))
            else:  # Likely an index
                converted_positions.append(int(p))
        
        # Clamp to valid range
        converted_positions = [max(0, min(length - 1, p)) for p in converted_positions]
        
        # Ensure we have the right number
        if len(converted_positions) > n_imgs:
            converted_positions = converted_positions[:n_imgs]
        elif len(converted_positions) < n_imgs:
            # Pad with last position
            converted_positions.extend([converted_positions[-1]] * (n_imgs - len(converted_positions)))
        
        return converted_positions
