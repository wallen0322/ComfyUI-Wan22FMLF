# -*- coding: utf-8 -*-

import torch
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
from nodes import MAX_RESOLUTION
from typing import Optional, Tuple, Any


class WanAdvancedI2V:
    """
    Advanced unified node for Wan2.2 A14B I2V with automatic chaining support.
    
    Core Features:
    - Triple frame reference (first, middle, last) for precise control
    - Dual MoE conditioning outputs (high-noise and low-noise stages)
    - Multi-motion frames support for dynamic sequences
    - Automatic video chaining with offset mechanism
    - SVI-SHOT mode for infinite video generation with separate conditioning
    - Adjustable constraint strengths for each stage
    - CLIP Vision integration
    
    Long Video Modes:
    - DISABLED: Standard single-shot generation
    - AUTO_CONTINUE: motion_frames replace start_image at frame 0
    - SVI_SHOT: Stable Video Infinity SHOT mode with separate high/low noise conditioning
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
                    "tooltip": "NORMAL: full control | SINGLE_PERSON: low noise only uses start image"
                }),
                "start_image": ("IMAGE",),
                "middle_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "middle_frame_ratio": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Position of middle frame on timeline (0.5 = center)"
                }),
                "motion_frames": ("IMAGE",),
                "video_frame_offset": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000000,
                    "step": 1,
                    "tooltip": "Frame offset for sequential video generation"
                }),
                "long_video_mode": (["DISABLED", "AUTO_CONTINUE", "SVI_SHOT"], {
                    "default": "DISABLED",
                    "tooltip": "DISABLED=normal | AUTO_CONTINUE=motion at frame 0 | SVI_SHOT=separate high/low noise"
                }),
                "continue_frames_count": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of frames to extract from motion_frames for continuation"
                }),
                "high_noise_mid_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "High-noise stage middle frame constraint strength"
                }),
                "low_noise_start_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Low-noise stage start frame constraint strength"
                }),
                "low_noise_mid_strength": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Low-noise stage middle frame constraint strength"
                }),
                "low_noise_end_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Low-noise stage end frame constraint strength"
                }),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_middle_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
                "enable_middle_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable middle frame constraint in triple frame reference mode"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("positive_high_noise", "positive_low_noise", "negative", "latent", 
                    "trim_latent", "trim_image", "next_offset")
    FUNCTION = "generate"
    CATEGORY = "ComfyUI-Wan22FMLF"

    def generate(self, 
                 positive: Tuple[Any, ...], 
                 negative: Tuple[Any, ...],
                 vae: Any,
                 width: int, 
                 height: int, 
                 length: int, 
                 batch_size: int,
                 mode: str = "NORMAL",
                 start_image: Optional[torch.Tensor] = None,
                 middle_image: Optional[torch.Tensor] = None,
                 end_image: Optional[torch.Tensor] = None,
                 middle_frame_ratio: float = 0.5,
                 motion_frames: Optional[torch.Tensor] = None,
                 video_frame_offset: int = 0,
                 long_video_mode: str = "DISABLED",
                 continue_frames_count: int = 5,
                 high_noise_mid_strength: float = 0.8,
                 low_noise_start_strength: float = 1.0,
                 low_noise_mid_strength: float = 0.2,
                 low_noise_end_strength: float = 1.0,
                 clip_vision_start_image: Optional[Any] = None,
                 clip_vision_middle_image: Optional[Any] = None,
                 clip_vision_end_image: Optional[Any] = None,
                 enable_middle_frame: bool = True) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], dict, int, int, int]:
        
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        
        device = comfy.model_management.intermediate_device()
        
        latent = torch.zeros([batch_size, latent_channels, latent_t, 
                             height // spacial_scale, width // spacial_scale], 
                             device=device)
        
        trim_latent = 0
        trim_image = 0
        next_offset = 0
        
        has_motion_frames = (motion_frames is not None and motion_frames.shape[0] > 0)
        is_pure_triple_mode = (not has_motion_frames and long_video_mode == "DISABLED")
        
        if video_frame_offset >= 0:
            if (long_video_mode == "AUTO_CONTINUE" or long_video_mode == "SVI_SHOT") and has_motion_frames and continue_frames_count > 0:
                actual_count = min(continue_frames_count, motion_frames.shape[0])
                motion_frames = motion_frames[-actual_count:]
                video_frame_offset = max(0, video_frame_offset - motion_frames.shape[0])
                trim_image = motion_frames.shape[0]
            
            if video_frame_offset > 0:
                if start_image is not None and start_image.shape[0] > 1:
                    start_image = start_image[video_frame_offset:] if start_image.shape[0] > video_frame_offset else None
                
                if middle_image is not None and middle_image.shape[0] > 1:
                    middle_image = middle_image[video_frame_offset:] if middle_image.shape[0] > video_frame_offset else None
                
                if end_image is not None and end_image.shape[0] > 1:
                    end_image = end_image[video_frame_offset:] if end_image.shape[0] > video_frame_offset else None
            
            next_offset = video_frame_offset + length
        
        if motion_frames is not None:
            motion_frames = comfy.utils.common_upscale(
                motion_frames.movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
        
        if start_image is not None:
            if is_pure_triple_mode:
                start_image = comfy.utils.common_upscale(
                    start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
            else:
                start_image = comfy.utils.common_upscale(
                    start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
        
        if middle_image is not None:
            middle_image = comfy.utils.common_upscale(
                middle_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        
        if end_image is not None:
            if is_pure_triple_mode:
                end_image = comfy.utils.common_upscale(
                    end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
            else:
                end_image = comfy.utils.common_upscale(
                    end_image[-1:].movedim(-1, 1), width, height, "bilinear", "center"
                ).movedim(1, -1)
        
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), device=device)
        
        middle_idx = self._calculate_aligned_position(middle_frame_ratio, length)[0]
        middle_idx = max(4, min(middle_idx, length - 5))
        
        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()
        
        svi_shot_second_pass = False
        
        if long_video_mode == "SVI_SHOT" and start_image is not None:
            if has_motion_frames:
                svi_shot_second_pass = True
            else:
                image[:start_image.shape[0]] = start_image[:, :, :, :3]
                start_latent_frames = ((start_image.shape[0] - 1) // 4) + 1
                mask_high_noise[:, :, :start_latent_frames * 4] = 0.0
                mask_low_noise[:, :, :start_latent_frames * 4] = max(0.0, 1.0 - low_noise_start_strength)
        
        if has_motion_frames and not (long_video_mode == "SVI_SHOT" and not svi_shot_second_pass):
            image[:motion_frames.shape[0]] = motion_frames[:, :, :, :3]
            
            motion_latent_frames = ((motion_frames.shape[0] - 1) // 4) + 1
            mask_high_noise[:, :, :motion_latent_frames * 4] = 0.0
            
            if not svi_shot_second_pass:
                mask_low_noise[:, :, :motion_latent_frames * 4] = 0.0
            
            if middle_image is not None and enable_middle_frame:
                image[middle_idx:middle_idx + 1] = middle_image
                
                start_range = max(0, middle_idx)
                end_range = min(length, middle_idx + 4)
                
                mask_high_noise[:, :, start_range:end_range] = max(0.0, 1.0 - high_noise_mid_strength)
                mask_low_noise[:, :, start_range:end_range] = max(0.0, 1.0 - low_noise_mid_strength)
            
            if end_image is not None:
                image[-1:] = end_image[:, :, :, :3]
                mask_high_noise[:, :, -1:] = 0.0
                mask_low_noise[:, :, -1:] = max(0.0, 1.0 - low_noise_end_strength)
        else:
            if start_image is not None:
                image[:start_image.shape[0]] = start_image[:, :, :, :3]
                
                if is_pure_triple_mode:
                    mask_range = min(start_image.shape[0] + 3, length)
                    mask_high_noise[:, :, :mask_range] = 0.0
                    mask_low_noise[:, :, :mask_range] = max(0.0, 1.0 - low_noise_start_strength)
                else:
                    start_latent_frames = ((start_image.shape[0] - 1) // 4) + 1
                    mask_high_noise[:, :, :start_latent_frames * 4] = 0.0
                    mask_low_noise[:, :, :start_latent_frames * 4] = max(0.0, 1.0 - low_noise_start_strength)
            
            if middle_image is not None and enable_middle_frame:
                image[middle_idx:middle_idx + 1] = middle_image
                
                start_range = max(0, middle_idx)
                end_range = min(length, middle_idx + 4)
                
                mask_high_noise[:, :, start_range:end_range] = max(0.0, 1.0 - high_noise_mid_strength)
                mask_low_noise[:, :, start_range:end_range] = max(0.0, 1.0 - low_noise_mid_strength)
            
            if end_image is not None:
                image[-end_image.shape[0]:] = end_image[:, :, :, :3]
                
                if is_pure_triple_mode:
                    mask_high_noise[:, :, -end_image.shape[0]:] = 0.0
                    mask_low_noise[:, :, -end_image.shape[0]:] = max(0.0, 1.0 - low_noise_end_strength)
                else:
                    mask_high_noise[:, :, -1:] = 0.0
                    mask_low_noise[:, :, -1:] = max(0.0, 1.0 - low_noise_end_strength)
        
        concat_latent_image = vae.encode(image[:, :, :, :3])
        
        if svi_shot_second_pass:
            image_low = torch.ones((length, height, width, 3), device=device) * 0.5
            
            if mode == "SINGLE_PERSON":
                if start_image is not None:
                    image_low[0] = start_image[0, :, :, :3]
                    mask_low_noise[:, :, 0:4] = 0.0
            else:
                if motion_frames is not None:
                    image_low[0] = motion_frames[0, :, :, :3]
                    mask_low_noise[:, :, 0:1] = 0.0
            
            concat_latent_image_low = vae.encode(image_low[:, :, :, :3])
        elif mode == "SINGLE_PERSON":
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            
            if motion_frames is not None:
                image_low_only[:motion_frames.shape[0]] = motion_frames[:, :, :, :3]
            elif start_image is not None:
                image_low_only[:start_image.shape[0]] = start_image[:, :, :, :3]
            
            concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
        elif low_noise_start_strength == 0.0 or low_noise_mid_strength == 0.0 or low_noise_end_strength == 0.0:
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            
            if motion_frames is not None and low_noise_start_strength > 0.0:
                image_low_only[:motion_frames.shape[0]] = motion_frames[:, :, :, :3]
            elif start_image is not None and low_noise_start_strength > 0.0:
                image_low_only[:start_image.shape[0]] = start_image[:, :, :, :3]
            
            if middle_image is not None and low_noise_mid_strength > 0.0 and enable_middle_frame:
                image_low_only[middle_idx:middle_idx + 1] = middle_image
            
            if end_image is not None and low_noise_end_strength > 0.0:
                if is_pure_triple_mode:
                    image_low_only[-end_image.shape[0]:] = end_image[:, :, :, :3]
                else:
                    image_low_only[-1:] = end_image[:, :, :, :3]
            
            concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
        else:
            concat_latent_image_low = concat_latent_image
        
        mask_high_reshaped = mask_high_noise.view(
            1, mask_high_noise.shape[2] // 4, 4, 
            mask_high_noise.shape[3], mask_high_noise.shape[4]
        ).transpose(1, 2)
        
        mask_low_reshaped = mask_low_noise.view(
            1, mask_low_noise.shape[2] // 4, 4, 
            mask_low_noise.shape[3], mask_low_noise.shape[4]
        ).transpose(1, 2)
        
        positive_high_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask_high_reshaped
        })
        
        positive_low_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image_low,
            "concat_mask": mask_low_reshaped
        })
        
        negative_out = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask_high_reshaped
        })
        
        clip_vision_output = self._merge_clip_vision_outputs(
            clip_vision_start_image, 
            clip_vision_middle_image, 
            clip_vision_end_image
        )
        
        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(
                positive_low_noise, 
                {"clip_vision_output": clip_vision_output}
            )
            negative_out = node_helpers.conditioning_set_values(
                negative_out,
                {"clip_vision_output": clip_vision_output}
            )
        
        out_latent = {"samples": latent}
        
        return (positive_high_noise, positive_low_noise, negative_out, out_latent,
                trim_latent, trim_image, next_offset)
    
    def _calculate_aligned_position(self, ratio: float, total_frames: int) -> Tuple[int, int]:
        desired_pixel_idx = int(total_frames * ratio)
        latent_idx = desired_pixel_idx // 4
        aligned_pixel_idx = latent_idx * 4
        aligned_pixel_idx = max(0, min(aligned_pixel_idx, total_frames - 1))
        return aligned_pixel_idx, latent_idx
    
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


class WanAdvancedExtractLastFrames:
    """Extract last N frames from latent for video stitching"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "num_frames": ("INT", {
                    "default": 9, 
                    "min": 0, 
                    "max": 81, 
                    "step": 1,
                    "tooltip": "Number of frames to extract"
                }),
            },
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("last_frames",)
    FUNCTION = "extract"
    CATEGORY = "ComfyUI-Wan22FMLF"
    
    def extract(self, samples: dict, num_frames: int) -> Tuple[dict]:
        if num_frames == 0:
            out = {"samples": torch.zeros_like(samples["samples"][:, :, :0])}
            return (out,)
        
        latent_frames = ((num_frames - 1) // 4) + 1
        last_latent = samples["samples"][:, :, -latent_frames:].clone()
        out = {"samples": last_latent}
        return (out,)


class WanAdvancedExtractLastImages:
    """Extract last N images for video stitching"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "num_frames": ("INT", {
                    "default": 9, 
                    "min": 0, 
                    "max": 81, 
                    "step": 1,
                    "tooltip": "Number of image frames to extract"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_images",)
    FUNCTION = "extract"
    CATEGORY = "ComfyUI-Wan22FMLF"
    
    def extract(self, images: torch.Tensor, num_frames: int) -> Tuple[torch.Tensor]:
        if num_frames == 0:
            last_images = images[:0].clone()
            return (last_images,)
        
        last_images = images[-num_frames:].clone()
        return (last_images,)


NODE_CLASS_MAPPINGS = {
    "WanAdvancedI2V": WanAdvancedI2V,
    "WanAdvancedExtractLastFrames": WanAdvancedExtractLastFrames,
    "WanAdvancedExtractLastImages": WanAdvancedExtractLastImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAdvancedI2V": "Wan Advanced I2V (Ultimate)",
    "WanAdvancedExtractLastFrames": "Wan Extract Last Frames (Latent)",
    "WanAdvancedExtractLastImages": "Wan Extract Last Images",
}
