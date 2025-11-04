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
    - Video stitching with prev_latent and overlap frames (Legacy mode)
    - Adjustable constraint strengths for each stage
    - CLIP Vision integration
    
    Automatic Chaining Features:
    - video_frame_offset: Automatic frame offset tracking
    - auto_continue: Automatic frame continuation from previous segment
    - trim outputs: Automatic frame trimming information
    
    Designed for both single-shot and infinite video generation workflows.
    Fully backward compatible with previous workflows.
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
                # Triple Frame Reference
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
                
                # Motion Frames (alternative to single start frame)
                "motion_frames": ("IMAGE",),
                "num_motion_frames": ("INT", {
                    "default": 5, 
                    "min": 0, 
                    "max": 20, 
                    "step": 1,
                    "tooltip": "Number of motion frames to use as sequence start"
                }),
                
                # Video Stitching (Legacy)
                "prev_latent": ("LATENT",),
                "overlap_frames": ("INT", {
                    "default": 9, 
                    "min": 0, 
                    "max": 40, 
                    "step": 1,
                    "tooltip": "Number of overlapping frames for seamless stitching (Legacy mode)"
                }),
                
                # Automatic Chaining (NEW)
                "video_frame_offset": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000000,
                    "step": 1,
                    "tooltip": "Frame offset for sequential video generation. Connect from previous node's next_offset output for automatic chaining."
                }),
                "auto_continue": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically reuse frames from previous segment for smooth continuation. Requires continue_frames input."
                }),
                "continue_frames": ("IMAGE",),
                "continue_frames_count": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of frames to reuse from previous segment (0 = disabled)"
                }),
                
                # MoE Dual-Phase Strength Control
                "high_noise_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "High-noise stage constraint strength (layout and trajectory)"
                }),
                "low_noise_strength": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Low-noise stage constraint strength (detail refinement)"
                }),
                
                # Frame Control Weights
                "start_frame_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Influence weight of start frame"
                }),
                "middle_frame_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Influence weight of middle frame"
                }),
                "end_frame_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Influence weight of end frame"
                }),
                
                # CLIP Vision
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_middle_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
                
                # Mode Control
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
                 start_image: Optional[torch.Tensor] = None,
                 middle_image: Optional[torch.Tensor] = None,
                 end_image: Optional[torch.Tensor] = None,
                 middle_frame_ratio: float = 0.5,
                 motion_frames: Optional[torch.Tensor] = None,
                 num_motion_frames: int = 5,
                 prev_latent: Optional[dict] = None,
                 overlap_frames: int = 9,
                 video_frame_offset: int = 0,
                 auto_continue: bool = False,
                 continue_frames: Optional[torch.Tensor] = None,
                 continue_frames_count: int = 5,
                 high_noise_strength: float = 0.8,
                 low_noise_strength: float = 0.2,
                 start_frame_weight: float = 1.0,
                 middle_frame_weight: float = 1.0,
                 end_frame_weight: float = 1.0,
                 clip_vision_start_image: Optional[Any] = None,
                 clip_vision_middle_image: Optional[Any] = None,
                 clip_vision_end_image: Optional[Any] = None,
                 enable_middle_frame: bool = True) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], dict, int, int, int]:
        
        # Initialize latent space
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        
        device = comfy.model_management.intermediate_device()
        
        latent = torch.zeros([batch_size, latent_channels, latent_t, 
                             height // spacial_scale, width // spacial_scale], 
                             device=device)
        
        # Initialize trim counters
        trim_latent = 0
        trim_image = 0
        
        # Detect mode: offset mode vs legacy mode
        use_offset_mode = (prev_latent is None and video_frame_offset >= 0)
        
        if use_offset_mode:
            # OFFSET MODE: Automatic chaining with frame offset
            
            # Handle auto_continue: automatically reuse frames from previous segment
            if auto_continue and continue_frames is not None and continue_frames_count > 0:
                # Extract last N frames from previous segment
                continue_frames_selected = continue_frames[-continue_frames_count:]
                
                # Compensate offset (these frames are reused, not new)
                video_frame_offset -= continue_frames_selected.shape[0]
                video_frame_offset = max(0, video_frame_offset)
                
                # Use as motion frames automatically
                motion_frames = continue_frames_selected
                num_motion_frames = motion_frames.shape[0]
                
                # Record trim info
                trim_image = num_motion_frames
            
            # Handle reference frame sequences with offset
            if video_frame_offset > 0:
                # Apply offset ONLY to sequences (multi-frame images)
                # Single-frame reference images (shape[0] == 1) are kept as-is
                if start_image is not None and start_image.shape[0] > 1:
                    if start_image.shape[0] > video_frame_offset:
                        start_image = start_image[video_frame_offset:]
                    else:
                        start_image = None  # Beyond range
                
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
            
            # Calculate next offset
            next_offset = video_frame_offset + length
            
        else:
            # LEGACY MODE: Use prev_latent + overlap
            
            # Handle prev_latent for video stitching
            if prev_latent is not None:
                prev_samples = prev_latent["samples"]
                overlap_latent_frames = ((overlap_frames - 1) // 4) + 1 if overlap_frames > 0 else 0
                if overlap_latent_frames > 0 and prev_samples.shape[2] >= overlap_latent_frames:
                    latent[:, :, :overlap_latent_frames] = prev_samples[:, :, -overlap_latent_frames:]
            
            # No offset output in legacy mode
            next_offset = 0
        
        # Create base image and masks
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), 
                              device=device)
        
        # Initialize separate masks for high and low noise
        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()
        
        # Detect if we have motion frames (auto or manual)
        has_motion_frames = (motion_frames is not None and num_motion_frames > 0)
        
        # MODE SELECTION: Motion Frames / Triple Frame Reference
        if has_motion_frames:
            # Motion Frames Mode (manual or auto_continue)
            # Note: start_image is ignored when motion_frames exist (motion_frames serve as start)
            # But middle_image and end_image still work for trajectory control
            
            motion_frames_proc = motion_frames[-num_motion_frames:]
            motion_frames_proc = comfy.utils.common_upscale(
                motion_frames_proc.movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)
            
            image[:motion_frames_proc.shape[0]] = motion_frames_proc[:, :, :, :3]
            
            motion_latent_frames = ((motion_frames_proc.shape[0] - 1) // 4) + 1
            mask_base[:, :, :motion_latent_frames * 4] = 0.0
            mask_high_noise[:, :, :motion_latent_frames * 4] = 0.0
            mask_low_noise[:, :, :motion_latent_frames * 4] = 0.0
            
            # Handle overlap with prev_latent (legacy mode only)
            if not use_offset_mode and prev_latent is not None and overlap_frames > 0:
                overlap_latent_frames = ((overlap_frames - 1) // 4) + 1
                mask_high_noise[:, :, :overlap_latent_frames * 4] = 0.0
                mask_low_noise[:, :, :overlap_latent_frames * 4] = 0.0
            
            # Process middle frame (for trajectory control)
            if middle_image is not None and enable_middle_frame:
                middle_image_proc = comfy.utils.common_upscale(
                    middle_image[:1].movedim(-1, 1), width, height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                
                # Calculate aligned position
                middle_idx, middle_latent_idx = self._calculate_aligned_position(
                    middle_frame_ratio, length
                )
                middle_idx = max(4, min(middle_idx, length - 5))
                
                image[middle_idx:middle_idx + 1] = middle_image_proc
                
                # Dual-phase middle frame masking with weights
                start_range = max(0, middle_idx)
                end_range = min(length, middle_idx + 4)
                
                # High noise: strong constraint for trajectory
                high_noise_mask_value = (1.0 - high_noise_strength) * middle_frame_weight
                mask_high_noise[:, :, start_range:end_range] = high_noise_mask_value
                
                # Low noise: weak constraint for detail refinement
                low_noise_mask_value = (1.0 - low_noise_strength) * middle_frame_weight
                mask_low_noise[:, :, start_range:end_range] = low_noise_mask_value
            
            # Process end image (for ending control)
            if end_image is not None:
                end_image_proc = comfy.utils.common_upscale(
                    end_image[-length:].movedim(-1, 1), width, height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                image[-end_image_proc.shape[0]:] = end_image_proc[:, :, :, :3]
                
                # Apply weight to mask (fixed constraint for end frame)
                end_mask_value = 1.0 - end_frame_weight
                end_latent_frames = ((end_image_proc.shape[0] - 1) // 4) + 1
                mask_high_noise[:, :, -end_latent_frames * 4:] = end_mask_value
                mask_low_noise[:, :, -end_latent_frames * 4:] = end_mask_value
        
        else:
            # Triple Frame Reference Mode
            # Process start frame
            if start_image is not None:
                start_image_proc = comfy.utils.common_upscale(
                    start_image[:length].movedim(-1, 1), width, height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                image[:start_image_proc.shape[0]] = start_image_proc[:, :, :, :3]
                
                # Apply weight to mask (fixed constraint for start frame)
                start_mask_value = 1.0 - start_frame_weight
                start_latent_frames = ((start_image_proc.shape[0] - 1) // 4) + 1
                mask_high_noise[:, :, :start_latent_frames * 4] = start_mask_value
                mask_low_noise[:, :, :start_latent_frames * 4] = start_mask_value
            
            # Process middle frame with alignment and dual-phase control
            if middle_image is not None and enable_middle_frame:
                middle_image_proc = comfy.utils.common_upscale(
                    middle_image[:1].movedim(-1, 1), width, height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                
                # Calculate aligned position
                middle_idx, middle_latent_idx = self._calculate_aligned_position(
                    middle_frame_ratio, length
                )
                middle_idx = max(4, min(middle_idx, length - 5))
                
                image[middle_idx:middle_idx + 1] = middle_image_proc
                
                # Dual-phase middle frame masking with weights
                start_range = max(0, middle_idx)
                end_range = min(length, middle_idx + 4)
                
                # High noise: strong constraint for trajectory
                high_noise_mask_value = (1.0 - high_noise_strength) * middle_frame_weight
                mask_high_noise[:, :, start_range:end_range] = high_noise_mask_value
                
                # Low noise: weak constraint for detail refinement
                low_noise_mask_value = (1.0 - low_noise_strength) * middle_frame_weight
                mask_low_noise[:, :, start_range:end_range] = low_noise_mask_value
            
            # Process end frame
            if end_image is not None:
                end_image_proc = comfy.utils.common_upscale(
                    end_image[-length:].movedim(-1, 1), width, height, 
                    "bilinear", "center"
                ).movedim(1, -1)
                image[-end_image_proc.shape[0]:] = end_image_proc[:, :, :, :3]
                
                # Apply weight to mask (fixed constraint for end frame)
                end_mask_value = 1.0 - end_frame_weight
                end_latent_frames = ((end_image_proc.shape[0] - 1) // 4) + 1
                mask_high_noise[:, :, -end_latent_frames * 4:] = end_mask_value
                mask_low_noise[:, :, -end_latent_frames * 4:] = end_mask_value
            
            # Handle overlap with prev_latent (legacy mode only)
            if not use_offset_mode and prev_latent is not None and overlap_frames > 0:
                overlap_latent_frames = ((overlap_frames - 1) // 4) + 1
                mask_high_noise[:, :, :overlap_latent_frames * 4] = 0.0
                mask_low_noise[:, :, :overlap_latent_frames * 4] = 0.0
        
        # Create separate latent images for high and low noise stages
        # High noise stage: includes all reference frames
        concat_latent_image_high = vae.encode(image[:, :, :, :3])
        
        # Low noise stage: conditionally exclude middle frame if strength is 0
        if (low_noise_strength == 0.0 and middle_image is not None and 
            enable_middle_frame and not has_motion_frames):
            # Create image without middle frame for low noise stage
            # Use the already processed image and just exclude the middle frame
            image_low_only = image.clone()
            
            # Reset middle frame area to neutral (gray)
            middle_idx, _ = self._calculate_aligned_position(middle_frame_ratio, length)
            middle_idx = max(4, min(middle_idx, length - 5))
            # Fill middle frame position with neutral gray
            image_low_only[middle_idx:middle_idx + 1] = 0.5
            
            concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
        else:
            concat_latent_image_low = concat_latent_image_high
        
        # Reshape masks for proper dimensions
        mask_high_reshaped = mask_high_noise.view(
            1, mask_high_noise.shape[2] // 4, 4, 
            mask_high_noise.shape[3], mask_high_noise.shape[4]
        ).transpose(1, 2)
        
        mask_low_reshaped = mask_low_noise.view(
            1, mask_low_noise.shape[2] // 4, 4, 
            mask_low_noise.shape[3], mask_low_noise.shape[4]
        ).transpose(1, 2)
        
        # Create dual conditioning outputs for MoE architecture
        # High noise stage: strong constraints for layout and trajectory
        positive_high_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image_high,
            "concat_mask": mask_high_reshaped
        })
        
        # Low noise stage: refined constraints for detail optimization
        positive_low_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image_low,
            "concat_mask": mask_low_reshaped
        })
        
        # Negative conditioning (shared)
        negative_out = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent_image_high,
            "concat_mask": mask_high_reshaped
        })
        
        # CLIP Vision integration (primarily for low noise stage detail refinement)
        clip_vision_output = self._merge_clip_vision_outputs(
            clip_vision_start_image, 
            clip_vision_middle_image, 
            clip_vision_end_image
        )
        
        if clip_vision_output is not None:
            # Add CLIP Vision to low noise stage for better detail understanding
            positive_low_noise = node_helpers.conditioning_set_values(
                positive_low_noise, 
                {"clip_vision_output": clip_vision_output}
            )
            
            # Optionally add to negative
            negative_out = node_helpers.conditioning_set_values(
                negative_out,
                {"clip_vision_output": clip_vision_output}
            )
        
        out_latent = {"samples": latent}
        
        return (positive_high_noise, positive_low_noise, negative_out, out_latent,
                trim_latent, trim_image, next_offset)
    
    def _calculate_aligned_position(self, ratio: float, total_frames: int) -> Tuple[int, int]:
        """Calculate 4-frame aligned position for middle frame"""
        desired_pixel_idx = int(total_frames * ratio)
        latent_idx = desired_pixel_idx // 4
        aligned_pixel_idx = latent_idx * 4
        aligned_pixel_idx = max(0, min(aligned_pixel_idx, total_frames - 1))
        return aligned_pixel_idx, latent_idx
    
    def _merge_clip_vision_outputs(self, *outputs: Any) -> Optional[Any]:
        """Merge multiple CLIP Vision outputs"""
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
