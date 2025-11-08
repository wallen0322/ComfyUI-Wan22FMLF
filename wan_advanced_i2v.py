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
    
    Automatic Chaining Features:
    - video_frame_offset: Automatic frame offset tracking
    - long_video_mode: Unified mode selection (DISABLED/AUTO_CONTINUE/SVI_SHOT)
    - trim outputs: Automatic frame trimming information
    
    Long Video Modes:
    - DISABLED: Standard single-shot generation (normal start/middle/end or motion_frames)
    - AUTO_CONTINUE: motion_frames replace start_image at frame 0 for standard continuation
    - SVI_SHOT: Stable Video Infinity SHOT mode with separate high/low noise conditioning
      * 1st generation: start_image at frame 0 (mask=0, locked) - standard I2V
      * 2nd+ generation: 
        - High-noise: motion_frames at frame 0 (locked)
        - Low-noise: start_image at frame 0 (locked)
        - Separate VAE encoding for high/low noise stages
        - Same trim logic as AUTO_CONTINUE (controlled by continue_frames_count)
      * Prevents color accumulation via separate conditioning paths
    
    Designed for both single-shot and infinite video generation workflows.
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
                    "tooltip": "Frame offset for sequential video generation. Connect from previous node's next_offset output for automatic chaining."
                }),
                "long_video_mode": (["DISABLED", "AUTO_CONTINUE", "SVI_SHOT"], {
                    "default": "DISABLED",
                    "tooltip": "DISABLED=normal | AUTO_CONTINUE=motion at frame 0 | SVI_SHOT=1st:start I2V, 2nd+:separate high/low noise"
                }),
                "continue_frames_count": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of frames to extract from motion_frames for continuation (both AUTO_CONTINUE and SVI_SHOT)"
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
        
        use_offset_mode = (video_frame_offset >= 0)
        
        if use_offset_mode:
            if (long_video_mode == "AUTO_CONTINUE" or long_video_mode == "SVI_SHOT") and motion_frames is not None and continue_frames_count > 0:
                actual_count = min(continue_frames_count, motion_frames.shape[0])
                motion_frames = motion_frames[-actual_count:]
                video_frame_offset -= motion_frames.shape[0]
                video_frame_offset = max(0, video_frame_offset)
                trim_image = motion_frames.shape[0]
            
            if video_frame_offset > 0:
                if start_image is not None and start_image.shape[0] > 1:
                    if start_image.shape[0] > video_frame_offset:
                        start_image = start_image[video_frame_offset:]
                    else:
                        start_image = None
                
                if middle_image is not None and middle_image.shape[0] > 1:
                    if middle_image.shape[0] > video_frame_offset:
                        middle_image = middle_image[video_frame_offset:]
                    else:
                        middle_image = None
                
                if end_image is not None and end_image.shape[0] > 1:
                    if end_image.shape[0] > video_frame_offset:
                        end_image = end_image[video_frame_offset:]
                    else:
                        end_image = None
            
            next_offset = video_frame_offset + length
        
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), 
                              device=device)
        
        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()
        
        has_motion_frames = (motion_frames is not None and motion_frames.shape[0] > 0)
        svi_shot_second_pass = False
        
        if long_video_mode == "SVI_SHOT" and start_image is not None:
            if has_motion_frames:
                svi_shot_second_pass = True
            else:
                start_image_proc = comfy.utils.common_upscale(
                    start_image[:1].movedim(-1, 1), width, height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                image[:start_image_proc.shape[0]] = start_image_proc[:, :, :, :3]
                
                start_latent_frames = ((start_image_proc.shape[0] - 1) // 4) + 1
                mask_high_noise[:, :, :start_latent_frames * 4] = 0.0
                
                start_mask_value = max(0.0, 1.0 - low_noise_start_strength)
                mask_low_noise[:, :, :start_latent_frames * 4] = start_mask_value
        
        if has_motion_frames and not (long_video_mode == "SVI_SHOT" and not svi_shot_second_pass):
            motion_frames_proc = comfy.utils.common_upscale(
                motion_frames.movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
            
            image[:motion_frames_proc.shape[0]] = motion_frames_proc[:, :, :, :3]
            
            motion_latent_frames = ((motion_frames_proc.shape[0] - 1) // 4) + 1
            mask_base[:, :, :motion_latent_frames * 4] = 0.0
            mask_high_noise[:, :, :motion_latent_frames * 4] = 0.0
            
            if svi_shot_second_pass:
                if mode == "SINGLE_PERSON":
                    # SINGLE_PERSON: lock first latent (4 frames) for start_image
                    mask_low_noise[:, :, 0:4] = 0.0
                else:
                    # Normal SVI_SHOT: lock only first frame for motion
                    mask_low_noise[:, :, 0:1] = 0.0
            else:
                mask_low_noise[:, :, :motion_latent_frames * 4] = 0.0
            
            if middle_image is not None and enable_middle_frame:
                middle_image_proc = comfy.utils.common_upscale(
                    middle_image[:1].movedim(-1, 1), width, height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                
                middle_idx, middle_latent_idx = self._calculate_aligned_position(
                    middle_frame_ratio, length
                )
                middle_idx = max(4, min(middle_idx, length - 5))
                
                image[middle_idx:middle_idx + 1] = middle_image_proc
                
                start_range = max(0, middle_idx)
                end_range = min(length, middle_idx + 4)
                
                high_noise_mask_value = max(0.0, 1.0 - (high_noise_mid_strength))
                mask_high_noise[:, :, start_range:end_range] = high_noise_mask_value
                
                low_noise_mask_value = max(0.0, 1.0 - (low_noise_mid_strength))
                mask_low_noise[:, :, start_range:end_range] = low_noise_mask_value
            
            if end_image is not None:
                end_image_proc = comfy.utils.common_upscale(
                    end_image[:1].movedim(-1, 1), width, height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                image[-end_image_proc.shape[0]:] = end_image_proc[:, :, :, :3]
                
                end_latent_frames = ((end_image_proc.shape[0] - 1) // 4) + 1
                mask_high_noise[:, :, -end_latent_frames * 4:] = 0.0
                
                end_mask_value = max(0.0, 1.0 - low_noise_end_strength)
                mask_low_noise[:, :, -end_latent_frames * 4:] = end_mask_value
            
        else:
            if start_image is not None:
                start_image_proc = comfy.utils.common_upscale(
                    start_image[:1].movedim(-1, 1), width, height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                image[:start_image_proc.shape[0]] = start_image_proc[:, :, :, :3]
                
                start_latent_frames = ((start_image_proc.shape[0] - 1) // 4) + 1
                mask_high_noise[:, :, :start_latent_frames * 4] = 0.0
                
                start_mask_value = max(0.0, 1.0 - low_noise_start_strength)
                mask_low_noise[:, :, :start_latent_frames * 4] = start_mask_value
            
            if middle_image is not None and enable_middle_frame:
                middle_image_proc = comfy.utils.common_upscale(
                    middle_image[:1].movedim(-1, 1), width, height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                
                middle_idx, middle_latent_idx = self._calculate_aligned_position(
                    middle_frame_ratio, length
                )
                middle_idx = max(4, min(middle_idx, length - 5))
                
                image[middle_idx:middle_idx + 1] = middle_image_proc
                
                start_range = max(0, middle_idx)
                end_range = min(length, middle_idx + 4)
                
                high_noise_mask_value = max(0.0, 1.0 - (high_noise_mid_strength))
                mask_high_noise[:, :, start_range:end_range] = high_noise_mask_value
                
                low_noise_mask_value = max(0.0, 1.0 - (low_noise_mid_strength))
                mask_low_noise[:, :, start_range:end_range] = low_noise_mask_value
            
            if end_image is not None:
                end_image_proc = comfy.utils.common_upscale(
                    end_image[:1].movedim(-1, 1), width, height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                image[-end_image_proc.shape[0]:] = end_image_proc[:, :, :, :3]
                
                end_latent_frames = ((end_image_proc.shape[0] - 1) // 4) + 1
                mask_high_noise[:, :, -end_latent_frames * 4:] = 0.0
                
                end_mask_value = max(0.0, 1.0 - low_noise_end_strength)
                mask_low_noise[:, :, -end_latent_frames * 4:] = end_mask_value
        
        if svi_shot_second_pass:
            concat_latent_image_high = vae.encode(image[:, :, :, :3])
            
            image_low = torch.ones((length, height, width, 3), device=device) * 0.5
            
            if mode == "SINGLE_PERSON":
                # SINGLE_PERSON mode in SVI_SHOT: use start_image for low noise
                if start_image is not None:
                    start_image_proc = comfy.utils.common_upscale(
                        start_image[:1].movedim(-1, 1), width, height, 
                        "bilinear", "center"
                    ).movedim(1, -1)
                    image_low[0] = start_image_proc[0, :, :, :3]
            else:
                # Normal SVI_SHOT: use motion first frame
                if motion_frames is not None and motion_frames.shape[0] > 0:
                    motion_first_frame = comfy.utils.common_upscale(
                        motion_frames[:1].movedim(-1, 1), width, height, "area", "center"
                    ).movedim(1, -1)
                    image_low[0] = motion_first_frame[0, :, :, :3]
            
            concat_latent_image_low = vae.encode(image_low[:, :, :, :3])

        else:
            concat_latent_image_high = vae.encode(image[:, :, :, :3])
            
            if mode == "SINGLE_PERSON":
                # SINGLE_PERSON mode: low noise only uses start_image
                # Create fresh mask_low_noise (all 1.0), then only lock start_image
                mask_low_noise = torch.ones(
                    (1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), 
                    device=device
                )
                if start_image is not None:
                    start_image_proc = comfy.utils.common_upscale(
                        start_image[:1].movedim(-1, 1), width, height, 
                        "bilinear", "center"
                    ).movedim(1, -1)
                    start_mask_value = max(0.0, 1.0 - low_noise_start_strength)
                    start_latent_frames = ((start_image_proc.shape[0] - 1) // 4) + 1
                    mask_low_noise[:, :, :start_latent_frames * 4] = start_mask_value
                
                # 创建灰色背景图像
                image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
                
                # 优先使用motion_frames的第一帧，如果没有再使用start_image
                if has_motion_frames and motion_frames is not None and motion_frames.shape[0] > 0:
                    # 使用motion_frames的第一帧
                    motion_first_frame = comfy.utils.common_upscale(
                        motion_frames[:1].movedim(-1, 1), width, height, "area", "center"
                    ).movedim(1, -1)
                    image_low_only[0] = motion_first_frame[0, :, :, :3]
                elif start_image is not None:
                    # 如果没有motion_frames，使用start_image
                    image_low_only[:start_image_proc.shape[0]] = start_image_proc[:, :, :, :3]
                
                concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
            elif (low_noise_mid_strength == 0.0 and middle_image is not None and 
                enable_middle_frame and not has_motion_frames):
                image_low_only = image.clone()
                middle_idx, _ = self._calculate_aligned_position(middle_frame_ratio, length)
                middle_idx = max(4, min(middle_idx, length - 5))
                image_low_only[middle_idx:middle_idx + 1] = 0.5
                concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
            else:
                concat_latent_image_low = concat_latent_image_high

        
        mask_high_reshaped = mask_high_noise.view(
            1, mask_high_noise.shape[2] // 4, 4, 
            mask_high_noise.shape[3], mask_high_noise.shape[4]
        ).transpose(1, 2)
        
        mask_low_reshaped = mask_low_noise.view(
            1, mask_low_noise.shape[2] // 4, 4, 
            mask_low_noise.shape[3], mask_low_noise.shape[4]
        ).transpose(1, 2)
        
        positive_high_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image_high,
            "concat_mask": mask_high_reshaped
        })
        
        positive_low_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image_low,
            "concat_mask": mask_low_reshaped
        })
        
        negative_out = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent_image_high,
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
