# -*- coding: utf-8 -*-

import torch
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
from nodes import MAX_RESOLUTION
from typing import Optional, Tuple, Any


class WanFirstMiddleLastFrameToVideo:
    """
    3-frame reference node for Wan2.2 A14B I2V with dual MoE conditioning.
    
    Features:
    - First, middle, and last frame reference
    - Dual conditioning outputs for high-noise and low-noise stages
    - Adjustable constraint strengths for MoE dual-phase sampling
    - Designed for LightX2V蒸馏模型（8步：4步高噪+4步低噪）
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
                }),
                # 双阶段强度控制
                "high_noise_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "高噪声阶段中间帧约束强度（确定动态轨迹）"
                }),
                "low_noise_strength": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "低噪声阶段中间帧约束强度（防止细节闪烁）"
                }),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_middle_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
            },
        }

    # 🎯 三个输出：positive高噪、positive低噪、latent
    # 负向条件使用原始输入（符合ComfyUI习惯）
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high_noise", "positive_low_noise", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "ComfyUI-Wan22FMLF/video"

    def generate(self, positive: Tuple[Any, ...], 
                 negative: Tuple[Any, ...],
                 vae: Any,
                 width: int, 
                 height: int, 
                 length: int, 
                 batch_size: int,
                 start_image: Optional[torch.Tensor] = None,
                 middle_image: Optional[torch.Tensor] = None,
                 end_image: Optional[torch.Tensor] = None,
                 middle_frame_ratio: float = 0.5,
                 high_noise_strength: float = 0.8,
                 low_noise_strength: float = 0.2,
                 clip_vision_start_image: Optional[Any] = None,
                 clip_vision_middle_image: Optional[Any] = None,
                 clip_vision_end_image: Optional[Any] = None) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], dict]:
        
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        
        device = comfy.model_management.intermediate_device()
        
        latent = torch.zeros([batch_size, latent_channels, latent_t, 
                             height // spacial_scale, width // spacial_scale], 
                             device=device)
        
        # 图像预处理
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
        
        # 创建时间线和基础mask
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), 
                              device=device)
        
        def calculate_aligned_position(ratio: float, total_frames: int) -> Tuple[int, int]:
            desired_pixel_idx = int(total_frames * ratio)
            latent_idx = desired_pixel_idx // 4
            aligned_pixel_idx = latent_idx * 4
            aligned_pixel_idx = max(0, min(aligned_pixel_idx, total_frames - 1))
            return aligned_pixel_idx, latent_idx
        
        middle_idx, middle_latent_idx = calculate_aligned_position(middle_frame_ratio, length)
        middle_idx = max(4, min(middle_idx, length - 5))
        
        # 放置参考帧
        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask_base[:, :, :start_image.shape[0] + 3] = 0.0
        
        if middle_image is not None:
            image[middle_idx:middle_idx + 1] = middle_image
            
            # 🎯 创建两个不同的mask
            mask_high_noise = mask_base.clone()
            mask_low_noise = mask_base.clone()
            
            # 高噪声mask（强约束）
            start_range = max(0, middle_idx)
            end_range = min(length, middle_idx + 4)
            high_noise_mask_value = 1.0 - high_noise_strength
            mask_high_noise[:, :, start_range:end_range] = high_noise_mask_value
            
            # 低噪声mask（弱约束）
            low_noise_mask_value = 1.0 - low_noise_strength
            mask_low_noise[:, :, start_range:end_range] = low_noise_mask_value
        
        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            if middle_image is not None:
                mask_high_noise[:, :, -end_image.shape[0]:] = 0.0
                mask_low_noise[:, :, -end_image.shape[0]:] = 0.0
        
        # 编码
        concat_latent_image = vae.encode(image[:, :, :, :3])
        
        # Mask重塑
        mask_high_reshaped = mask_high_noise.view(1, mask_high_noise.shape[2] // 4, 4, mask_high_noise.shape[3], mask_high_noise.shape[4]).transpose(1, 2)
        mask_low_reshaped = mask_low_noise.view(1, mask_low_noise.shape[2] // 4, 4, mask_low_noise.shape[3], mask_low_noise.shape[4]).transpose(1, 2)
        
        # 🎯 创建三种conditioning设置
        # 高噪声阶段：强约束，确定动态轨迹
        positive_high_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask_high_reshaped
        })
        
        # 低噪声阶段：弱约束，防止细节闪烁
        positive_low_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask_low_reshaped
        })
        
        # 负向条件使用原始输入
        negative_out = negative
        
        # CLIP Vision处理（主要用于低噪声阶段的细节优化）
        clip_vision_output = self._merge_clip_vision_outputs(
            clip_vision_start_image, 
            clip_vision_middle_image, 
            clip_vision_end_image
        )
        
        if clip_vision_output is not None:
            # 只在低噪声阶段添加CLIP Vision（更好的细节理解）
            positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise, 
                                                                   {"clip_vision_output": clip_vision_output})
        
        out_latent = {"samples": latent}
        
        return (positive_high_noise, positive_low_noise, negative_out, out_latent)

    def _merge_clip_vision_outputs(self, *outputs: Any) -> Optional[Any]:
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
    "WanFirstMiddleLastFrameToVideo": "Wan First-Middle-Last Frame (Dual MoE) 🎬"
}
