# -*- coding: utf-8 -*-

import torch
import json
from typing import List, Tuple, Optional, Any
import node_helpers
import comfy
import comfy.utils
from nodes import MAX_RESOLUTION


class WanMultiFrameRefToVideo:
    """
    Universal N-frame reference node with dual MoE conditioning.
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
            },
            "optional": {
                "ref_positions": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Frame indices or ratios. Leave empty for auto distribution."
                }),
                "ref_strength_high": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "High-noise stage strength for middle frames."
                }),
                "ref_strength_low": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Low-noise stage strength for middle frames."
                }),

                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high_noise", "positive_low_noise", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "ComfyUI-Wan22FMLF/video"

    def generate(self, positive: Tuple[Any, ...], 
                 negative: Tuple[Any, ...],
                 vae: Any,
                 width: int, height: int, length: int, batch_size: int,
                 ref_images: torch.Tensor,
                 ref_positions: str = "",
                 ref_strength_high: float = 0.8,
                 ref_strength_low: float = 0.2,

                 clip_vision_output: Optional[Any] = None) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], dict]:
        
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        device = comfy.model_management.intermediate_device()
        
        latent = torch.zeros([batch_size, latent_channels, latent_t, 
                             height // spacial_scale, width // spacial_scale], device=device)
        
        imgs = self._resize_images(ref_images, width, height, device)
        n_imgs = imgs.shape[0]
        positions = self._parse_positions(ref_positions, n_imgs, length)
        
        def align_position(pos: int, total_frames: int) -> int:
            latent_idx = pos // 4
            aligned_pos = latent_idx * 4
            aligned_pos = max(0, min(aligned_pos, total_frames - 1))
            return aligned_pos
        
        aligned_positions = [align_position(int(p), length) for p in positions]
        
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), device=device)
        
        # 创建双mask
        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()
        
        for i, pos in enumerate(aligned_positions):
            frame_idx = int(pos)
            image[frame_idx:frame_idx + 1] = imgs[i]
            
            is_endpoint = (i == 0) or (i == n_imgs - 1)
            if is_endpoint:
                mask_high_noise[:, :, frame_idx:frame_idx + 4] = 0.0
                mask_low_noise[:, :, frame_idx:frame_idx + 4] = 0.0
            else:
                start_range = max(0, frame_idx)
                end_range = min(length, frame_idx + 4)
                
                # 高噪声mask
                mask_high_value = 1.0 - ref_strength_high
                mask_high_noise[:, :, start_range:end_range] = mask_high_value
                
                # 低噪声mask
                mask_low_value = 1.0 - ref_strength_low
                mask_low_noise[:, :, start_range:end_range] = mask_low_value
            

        
        concat_latent_image = vae.encode(image[:, :, :, :3])
        
        mask_high_reshaped = mask_high_noise.view(1, mask_high_noise.shape[2] // 4, 4, mask_high_noise.shape[3], mask_high_noise.shape[4]).transpose(1, 2)
        mask_low_reshaped = mask_low_noise.view(1, mask_low_noise.shape[2] // 4, 4, mask_low_noise.shape[3], mask_low_noise.shape[4]).transpose(1, 2)
        
        positive_high_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask_high_reshaped
        })
        
        positive_low_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask_low_reshaped
        })
        
        negative_out = negative
        
        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise, 
                                                                   {"clip_vision_output": clip_vision_output})
        
        return (positive_high_noise, positive_low_noise, negative_out, {"samples": latent})

    def _resize_images(self, images: torch.Tensor, width: int, height: int, device: torch.device) -> torch.Tensor:
        images = images.to(device)
        x = images.movedim(-1, 1)
        x = comfy.utils.common_upscale(x, width, height, "bilinear", "center")
        x = x.movedim(1, -1)
        
        if x.shape[-1] == 4:
            x = x[..., :3]
        
        return x

    def _parse_positions(self, pos_str: str, n_imgs: int, length: int) -> List[int]:
        positions = []
        s = (pos_str or "").strip()
        
        if s:
            try:
                if s.startswith("["):
                    positions = json.loads(s)
                else:
                    positions = [float(x.strip()) for x in s.split(",") if x.strip()]
            except Exception:
                positions = []
        
        if not positions:
            if n_imgs <= 1:
                positions = [0]
            else:
                positions = [i * (length - 1) / (n_imgs - 1) for i in range(n_imgs)]
        
        converted_positions = []
        for p in positions:
            if 0 <= p < 2.0:
                converted_positions.append(int(p * (length - 1)))
            else:
                converted_positions.append(int(p))
        
        converted_positions = [max(0, min(length - 1, p)) for p in converted_positions]
        
        if len(converted_positions) > n_imgs:
            converted_positions = converted_positions[:n_imgs]
        elif len(converted_positions) < n_imgs:
            converted_positions.extend([converted_positions[-1]] * (n_imgs - len(converted_positions)))
        
        return converted_positions


NODE_CLASS_MAPPINGS = {"WanMultiFrameRefToVideo": WanMultiFrameRefToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"WanMultiFrameRefToVideo": "Wan Multi-Frame Reference (Dual MoE) 🎭"}
