# -*- coding: utf-8 -*-

import torch
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
from nodes import MAX_RESOLUTION
from typing import Optional, Tuple, Any


class WanFourFrameReferenceUltimate:
    """
    4-frame reference node with dual MoE conditioning.
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
                "mode": (["NORMAL", "SINGLE_PERSON"], {
                    "default": "NORMAL",
                    "tooltip": "NORMAL: full control | SINGLE_PERSON: low noise only uses frame 1"
                }),
                "frame_1_image": ("IMAGE",),
                "frame_2_image": ("IMAGE",),
                "frame_2_ratio": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "frame_2_strength_high": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "frame_2_strength_low": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "enable_frame_2": (["disable", "enable"], {"default": "enable"}),
                
                "frame_3_image": ("IMAGE",),
                "frame_3_ratio": ("FLOAT", {"default": 0.67, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "frame_3_strength_high": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "frame_3_strength_low": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "enable_frame_3": (["disable", "enable"], {"default": "enable"}),
                
                "frame_4_image": ("IMAGE",),
                
                "clip_vision_frame_1": ("CLIP_VISION_OUTPUT",),
                "clip_vision_frame_2": ("CLIP_VISION_OUTPUT",),
                "clip_vision_frame_3": ("CLIP_VISION_OUTPUT",),
                "clip_vision_frame_4": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high_noise", "positive_low_noise", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "ComfyUI-Wan22FMLF"

    def generate(self, positive: Tuple[Any, ...], 
                 negative: Tuple[Any, ...],
                 vae: Any,
                 width: int, height: int, length: int, batch_size: int,
                 mode: str = "NORMAL",
                 frame_1_image: Optional[torch.Tensor] = None,
                 frame_2_image: Optional[torch.Tensor] = None,
                 frame_2_ratio: float = 0.33,
                 frame_2_strength_high: float = 0.8,
                 frame_2_strength_low: float = 0.2,
                 enable_frame_2: str = "enable",
                 frame_3_image: Optional[torch.Tensor] = None,
                 frame_3_ratio: float = 0.67,
                 frame_3_strength_high: float = 0.8,
                 frame_3_strength_low: float = 0.2,
                 enable_frame_3: str = "enable",
                 frame_4_image: Optional[torch.Tensor] = None,
                 clip_vision_frame_1: Optional[Any] = None,
                 clip_vision_frame_2: Optional[Any] = None,
                 clip_vision_frame_3: Optional[Any] = None,
                 clip_vision_frame_4: Optional[Any] = None) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], dict]:
        
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        device = comfy.model_management.intermediate_device()
        
        latent = torch.zeros([batch_size, latent_channels, latent_t, 
                             height // spacial_scale, width // spacial_scale], device=device)
        
        if frame_1_image is not None:
            frame_1_image = comfy.utils.common_upscale(frame_1_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if frame_2_image is not None:
            frame_2_image = comfy.utils.common_upscale(frame_2_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if frame_3_image is not None:
            frame_3_image = comfy.utils.common_upscale(frame_3_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if frame_4_image is not None:
            frame_4_image = comfy.utils.common_upscale(frame_4_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), device=device)
        
        def calculate_aligned_position(ratio: float, total_frames: int) -> Tuple[int, int]:
            desired_pixel_idx = int(total_frames * ratio)
            latent_idx = desired_pixel_idx // 4
            aligned_pixel_idx = latent_idx * 4
            aligned_pixel_idx = max(0, min(aligned_pixel_idx, total_frames - 1))
            return aligned_pixel_idx, latent_idx
        
        frame_1_idx = 0
        frame_1_latent_idx = 0
        
        frame_2_idx, frame_2_latent_idx = calculate_aligned_position(frame_2_ratio, length)
        frame_3_idx, frame_3_latent_idx = calculate_aligned_position(frame_3_ratio, length)
        
        frame_4_idx_raw = length - 1
        frame_4_idx, frame_4_latent_idx = calculate_aligned_position(frame_4_idx_raw / length, length)
        
        if frame_2_idx <= frame_1_idx + 4:
            frame_2_idx = frame_1_idx + 4
            frame_2_latent_idx = frame_2_idx // 4
        
        if frame_3_idx <= frame_2_idx + 4:
            frame_3_idx = frame_2_idx + 4
            frame_3_latent_idx = frame_3_idx // 4
        
        if frame_4_idx <= frame_3_idx + 4:
            frame_4_idx = frame_3_idx + 4
            frame_4_latent_idx = frame_4_idx // 4
        
        # 创建双mask
        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()
        
        if frame_1_image is not None:
            image[:frame_1_image.shape[0]] = frame_1_image
            mask_high_noise[:, :, :frame_1_image.shape[0] + 3] = 0.0
            mask_low_noise[:, :, :frame_1_image.shape[0] + 3] = 0.0
        
        if frame_2_image is not None and enable_frame_2 == "enable":
            image[frame_2_idx:frame_2_idx + frame_2_image.shape[0]] = frame_2_image
            start_range = max(0, frame_2_idx)
            end_range = min(length, frame_2_idx + 4)
            
            # 高噪声mask
            mask_high_value = 1.0 - frame_2_strength_high
            mask_high_noise[:, :, start_range:end_range] = mask_high_value
            
            # 低噪声mask
            mask_low_value = 1.0 - frame_2_strength_low
            mask_low_noise[:, :, start_range:end_range] = mask_low_value
        
        if frame_3_image is not None and enable_frame_3 == "enable":
            image[frame_3_idx:frame_3_idx + frame_3_image.shape[0]] = frame_3_image
            start_range = max(0, frame_3_idx)
            end_range = min(length, frame_3_idx + 4)
            
            # 高噪声mask
            mask_high_value = 1.0 - frame_3_strength_high
            mask_high_noise[:, :, start_range:end_range] = mask_high_value
            
            # 低噪声mask
            mask_low_value = 1.0 - frame_3_strength_low
            mask_low_noise[:, :, start_range:end_range] = mask_low_value
        
        if frame_4_image is not None:
            image[frame_4_idx:frame_4_idx + frame_4_image.shape[0]] = frame_4_image
            mask_high_noise[:, :, frame_4_idx:frame_4_idx + 4] = 0.0
            mask_low_noise[:, :, frame_4_idx:frame_4_idx + 4] = 0.0
        
        # Separate latent images for high and low noise stages
        # High noise stage: includes all frames
        concat_latent_image_high = vae.encode(image[:, :, :, :3])
        
        # Low noise stage
        if mode == "SINGLE_PERSON":
            # SINGLE_PERSON mode: low noise only uses frame_1
            # Reset mask_low_noise to all 1.0, then only lock frame_1
            mask_low_noise = mask_base.clone()
            if frame_1_image is not None:
                mask_low_noise[:, :, :frame_1_image.shape[0] + 3] = 0.0
            
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if frame_1_image is not None:
                image_low_only[:frame_1_image.shape[0]] = frame_1_image
            concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
        else:
            # NORMAL mode
            # 低噪声阶段：如果所有中间帧强度都为0则跳过中间帧
            frame_2_strength = frame_2_strength_low if enable_frame_2 == "enable" else 0.0
            frame_3_strength = frame_3_strength_low if enable_frame_3 == "enable" else 0.0
            
            if frame_2_strength == 0.0 and frame_3_strength == 0.0:
                # All middle frame strengths are 0: create latent without middle frames
                image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
                
                # 只放置frame_1和frame_4
                if frame_1_image is not None:
                    image_low_only[:frame_1_image.shape[0]] = frame_1_image
                if frame_4_image is not None:
                    image_low_only[frame_4_idx:frame_4_idx + frame_4_image.shape[0]] = frame_4_image
                
                concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
            else:
                # 有中间帧强度>0：使用完整图像
                concat_latent_image_low = vae.encode(image[:, :, :, :3])
        
        mask_high_reshaped = mask_high_noise.view(1, mask_high_noise.shape[2] // 4, 4, mask_high_noise.shape[3], mask_high_noise.shape[4]).transpose(1, 2)
        mask_low_reshaped = mask_low_noise.view(1, mask_low_noise.shape[2] // 4, 4, mask_low_noise.shape[3], mask_low_noise.shape[4]).transpose(1, 2)
        
        positive_high_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image_high,
            "concat_mask": mask_high_reshaped
        })
        
        positive_low_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image_low,
            "concat_mask": mask_low_reshaped
        })
        
        negative_out = negative
        
        clip_vision_output = self._merge_clip_vision_outputs(
            clip_vision_frame_1, clip_vision_frame_2, 
            clip_vision_frame_3, clip_vision_frame_4
        )
        
        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise, 
                                                                   {"clip_vision_output": clip_vision_output})
        
        return (positive_high_noise, positive_low_noise, negative_out, {"samples": latent})

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


NODE_CLASS_MAPPINGS = {"WanFourFrameReferenceUltimate": WanFourFrameReferenceUltimate}
NODE_DISPLAY_NAME_MAPPINGS = {"WanFourFrameReferenceUltimate": "Wan 4-Frame Reference (Dual MoE)"}
