from typing_extensions import override
from comfy_api.latest import io
import torch
import torch.nn.functional as F
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
import comfy.latent_formats
from typing import Optional, Tuple, Any


class WanAdvancedI2V(io.ComfyNode):
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent", "trim_latent", "trim_image", "next_offset")
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanAdvancedI2V",
            display_name="Wan Advanced I2V (Ultimate)",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=480, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number),
                io.Int.Input("length", default=81, min=1, max=8192, step=4, display_mode=io.NumberDisplay.number),
                io.Int.Input("batch_size", default=1, min=1, max=4096, display_mode=io.NumberDisplay.number),
                io.Combo.Input("mode", ["NORMAL", "SINGLE_PERSON"], default="NORMAL", optional=True),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("middle_image", optional=True),
                io.Image.Input("end_image", optional=True),
                io.Float.Input("middle_frame_ratio", default=0.5, min=0.0, max=1.0, step=0.01, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Image.Input("motion_frames", optional=True),
                io.Int.Input("video_frame_offset", default=0, min=0, max=1000000, step=1, display_mode=io.NumberDisplay.number, optional=True),
                io.Combo.Input("long_video_mode", ["DISABLED", "AUTO_CONTINUE", "SVI", "LATENT_CONTINUE"], default="DISABLED", optional=True),
                io.Int.Input("continue_frames_count", default=5, min=0, max=20, step=1, display_mode=io.NumberDisplay.number, optional=True),
                io.Float.Input("high_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("high_noise_mid_strength", default=0.8, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("low_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("low_noise_mid_strength", default=0.2, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("low_noise_end_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("structural_repulsion_boost", default=1.0, min=1.0, max=2.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True, tooltip="Motion enhancement through spatial gradient conditioning. Only affects high-noise stage."),
                io.ClipVisionOutput.Input("clip_vision_start_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_middle_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_end_image", optional=True),
                io.Boolean.Input("enable_middle_frame", default=True, optional=True),
                io.Latent.Input("prev_latent", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive_high"),
                io.Conditioning.Output(display_name="positive_low"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="trim_latent"),
                io.Int.Output(display_name="trim_image"),
                io.Int.Output(display_name="next_offset"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                mode="NORMAL", start_image=None, middle_image=None, end_image=None,
                middle_frame_ratio=0.5, motion_frames=None, video_frame_offset=0,
                long_video_mode="DISABLED", continue_frames_count=5,
                high_noise_start_strength=1.0, high_noise_mid_strength=0.8, 
                low_noise_start_strength=1.0, low_noise_mid_strength=0.2, low_noise_end_strength=1.0,
                structural_repulsion_boost=1.0,
                clip_vision_start_image=None, clip_vision_middle_image=None,
                clip_vision_end_image=None, enable_middle_frame=True,
                prev_latent=None):
        
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
            if (long_video_mode == "AUTO_CONTINUE" or long_video_mode == "SVI") and has_motion_frames and continue_frames_count > 0:
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
        
        middle_idx = cls._calculate_aligned_position(middle_frame_ratio, length)[0]
        middle_idx = max(4, min(middle_idx, length - 5))
        
        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()
        
        svi_continue_mode = False

        # --- Latent Continue Mode Logic ---
        latent_continue_mode = False
        prev_latent_for_concat = None
        if long_video_mode == 'LATENT_CONTINUE':
            has_prev_latent = (prev_latent is not None and prev_latent.get("samples") is not None)
            if has_prev_latent and continue_frames_count > 0 and start_image is None:
                latent_continue_mode = True
                prev_samples = prev_latent["samples"]
                
                if prev_samples.shape[2] > 0:
                    for b in range(batch_size):
                        latent[b:b+1, :, 0:1, :, :] = prev_samples[:, :, -1:].clone()
                    
                    mask_high_noise[:, :, :4] = 0.0
                    mask_low_noise[:, :, :4] = 0.0
                    
                    prev_latent_for_concat = prev_samples[:, :, -1:].clone()
        # --- End of Latent Continue Mode Logic ---

        # --- SVI Mode Logic ---
        if long_video_mode == 'SVI':
            # SVI mode uses latent-space conditioning similar to WanImageToVideoSVIPro
            spacial_scale = vae.spacial_compression_encode()
            latent_channels = vae.latent_channels
            total_latents = ((length - 1) // 4) + 1
            H = height // spacial_scale
            W = width // spacial_scale
            
            # Calculate middle and end positions
            middle_latent_idx = middle_idx // 4
            end_latent_idx = total_latents - 1
            
            # Check if we have prev_latent for continuation
            has_prev_latent = (prev_latent is not None and prev_latent.get("samples") is not None)
            
            if has_prev_latent and continue_frames_count > 0:
                # SVI Continue: Use prev_latent as continuation reference
                svi_continue_mode = True
                
                # Encode start_image as anchor latent
                if start_image is not None:
                    anchor_latent = vae.encode(start_image[:1, :, :, :3])
                else:
                    # If no start_image, create empty anchor
                    anchor_latent = torch.zeros([1, latent_channels, 1, H, W], 
                                               device=device, dtype=latent.dtype)
                
                # Use prev_latent directly (already encoded)
                prev_samples = prev_latent["samples"]
                # Convert continue_frames_count (image frames) to latent frames
                # 1 latent frame = 4 image frames
                motion_latent_count = min(prev_samples.shape[2], ((continue_frames_count - 1) // 4) + 1)
                motion_latent = prev_samples[:, :, -motion_latent_count:].clone()
                
                # Build image_cond_latent by inserting latents at correct positions
                # Start with padding, then insert anchor, motion, middle, end at their positions
                image_cond_latent = torch.zeros(1, latent_channels, total_latents, H, W, 
                                               dtype=anchor_latent.dtype, device=anchor_latent.device)
                image_cond_latent = comfy.latent_formats.Wan21().process_out(image_cond_latent)
                
                # Insert anchor at position 0
                image_cond_latent[:, :, :1] = anchor_latent
                
                # Insert motion_latent right after anchor (for continuity)
                motion_start = 1
                motion_end = motion_start + motion_latent.shape[2]
                if motion_end <= total_latents:
                    image_cond_latent[:, :, motion_start:motion_end] = motion_latent
                
                # Insert middle_image at middle_latent_idx if provided
                if middle_image is not None and enable_middle_frame:
                    middle_latent = vae.encode(middle_image[:1, :, :, :3])
                    if middle_latent_idx < total_latents:
                        image_cond_latent[:, :, middle_latent_idx:middle_latent_idx+1] = middle_latent
                
                # Insert end_image at end_latent_idx if provided
                if end_image is not None:
                    end_latent = vae.encode(end_image[:1, :, :, :3])
                    image_cond_latent[:, :, end_latent_idx:end_latent_idx+1] = end_latent
                
                # Create masks with strength parameters
                mask_svi_high = torch.ones((1, 1, total_latents, H, W), 
                                          device=device, dtype=anchor_latent.dtype)
                mask_svi_low = torch.ones((1, 1, total_latents, H, W), 
                                         device=device, dtype=anchor_latent.dtype)
                
                # Start frame: apply strength
                mask_svi_high[:, :, :1] = max(0.0, 1.0 - high_noise_start_strength)
                mask_svi_low[:, :, :1] = max(0.0, 1.0 - low_noise_start_strength)
                
                # Middle frame: apply strength
                if middle_image is not None and enable_middle_frame:
                    start_range = max(0, middle_latent_idx)
                    end_range = min(total_latents, middle_latent_idx + 1)
                    mask_svi_high[:, :, start_range:end_range] = max(0.0, 1.0 - high_noise_mid_strength)
                    mask_svi_low[:, :, start_range:end_range] = max(0.0, 1.0 - low_noise_mid_strength)
                
                # End frame: apply strength
                if end_image is not None:
                    mask_svi_high[:, :, end_latent_idx:end_latent_idx+1] = 0.0
                    mask_svi_low[:, :, end_latent_idx:end_latent_idx+1] = max(0.0, 1.0 - low_noise_end_strength)
                
                positive_high_noise = node_helpers.conditioning_set_values(positive, {
                    "concat_latent_image": image_cond_latent,
                    "concat_mask": mask_svi_high
                })
                
                positive_low_noise = node_helpers.conditioning_set_values(positive, {
                    "concat_latent_image": image_cond_latent,
                    "concat_mask": mask_svi_low
                })
                
                negative_out = node_helpers.conditioning_set_values(negative, {
                    "concat_latent_image": image_cond_latent,
                    "concat_mask": mask_svi_high
                })
                
                # Handle clip vision
                clip_vision_output = cls._merge_clip_vision_outputs(
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
                
                return io.NodeOutput(positive_high_noise, positive_low_noise, negative_out, out_latent,
                        trim_latent, trim_image, next_offset)
            
            elif start_image is not None:
                # SVI First: Use start_image as the only anchor
                svi_continue_mode = False
                
                # Encode start_image as anchor latent
                anchor_latent = vae.encode(start_image[:1, :, :, :3])
                
                # Build image_cond_latent by inserting latents at correct positions
                # Start with padding, then insert anchor, middle, end at their positions
                image_cond_latent = torch.zeros(1, latent_channels, total_latents, H, W, 
                                               dtype=anchor_latent.dtype, device=anchor_latent.device)
                image_cond_latent = comfy.latent_formats.Wan21().process_out(image_cond_latent)
                
                # Insert anchor at position 0
                image_cond_latent[:, :, :1] = anchor_latent
                
                # Insert middle_image at middle_latent_idx if provided
                if middle_image is not None and enable_middle_frame:
                    middle_latent = vae.encode(middle_image[:1, :, :, :3])
                    if middle_latent_idx < total_latents:
                        image_cond_latent[:, :, middle_latent_idx:middle_latent_idx+1] = middle_latent
                
                # Insert end_image at end_latent_idx if provided
                if end_image is not None:
                    end_latent = vae.encode(end_image[:1, :, :, :3])
                    image_cond_latent[:, :, end_latent_idx:end_latent_idx+1] = end_latent
                
                # Create masks with strength parameters
                mask_svi_high = torch.ones((1, 1, total_latents, H, W), 
                                          device=device, dtype=anchor_latent.dtype)
                mask_svi_low = torch.ones((1, 1, total_latents, H, W), 
                                         device=device, dtype=anchor_latent.dtype)
                
                # Start frame: apply strength
                mask_svi_high[:, :, :1] = max(0.0, 1.0 - high_noise_start_strength)
                mask_svi_low[:, :, :1] = max(0.0, 1.0 - low_noise_start_strength)
                
                # Middle frame: apply strength
                if middle_image is not None and enable_middle_frame:
                    start_range = max(0, middle_latent_idx)
                    end_range = min(total_latents, middle_latent_idx + 1)
                    mask_svi_high[:, :, start_range:end_range] = max(0.0, 1.0 - high_noise_mid_strength)
                    mask_svi_low[:, :, start_range:end_range] = max(0.0, 1.0 - low_noise_mid_strength)
                
                # End frame: apply strength
                if end_image is not None:
                    mask_svi_high[:, :, end_latent_idx:end_latent_idx+1] = 0.0
                    mask_svi_low[:, :, end_latent_idx:end_latent_idx+1] = max(0.0, 1.0 - low_noise_end_strength)
                
                positive_high_noise = node_helpers.conditioning_set_values(positive, {
                    "concat_latent_image": image_cond_latent,
                    "concat_mask": mask_svi_high
                })
                
                positive_low_noise = node_helpers.conditioning_set_values(positive, {
                    "concat_latent_image": image_cond_latent,
                    "concat_mask": mask_svi_low
                })
                
                negative_out = node_helpers.conditioning_set_values(negative, {
                    "concat_latent_image": image_cond_latent,
                    "concat_mask": mask_svi_high
                })
                
                # Handle clip vision
                clip_vision_output = cls._merge_clip_vision_outputs(
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
                
                return io.NodeOutput(positive_high_noise, positive_low_noise, negative_out, out_latent,
                        trim_latent, trim_image, next_offset)
        # --- End of SVI Mode Logic ---

        # Original logic for other modes (AUTO_CONTINUE, NORMAL)
        if has_motion_frames and long_video_mode != 'SVI' and long_video_mode != 'LATENT_CONTINUE':
            image[:motion_frames.shape[0]] = motion_frames[:, :, :, :3]
            
            motion_latent_frames = ((motion_frames.shape[0] - 1) // 4) + 1
            mask_high_noise[:, :, :motion_latent_frames * 4] = 0.0
            
            if not svi_continue_mode:
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
            if start_image is not None and long_video_mode != 'LATENT_CONTINUE':
                image[:start_image.shape[0]] = start_image[:, :, :, :3]
                
                if is_pure_triple_mode:
                    mask_range = min(start_image.shape[0] + 3, length)
                    mask_high_noise[:, :, :mask_range] = max(0.0, 1.0 - high_noise_start_strength)
                    mask_low_noise[:, :, :mask_range] = max(0.0, 1.0 - low_noise_start_strength)
                else:
                    start_latent_frames = ((start_image.shape[0] - 1) // 4) + 1
                    mask_high_noise[:, :, :start_latent_frames * 4] = max(0.0, 1.0 - high_noise_start_strength)
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
        
        if latent_continue_mode and prev_latent_for_concat is not None:
            concat_latent_image = vae.encode(image[:, :, :, :3])
            concat_latent_image[:, :, 0:1, :, :] = prev_latent_for_concat
        else:
            concat_latent_image = vae.encode(image[:, :, :, :3])
        
        if structural_repulsion_boost > 1.001 and length > 4:
            mask_h, mask_w = mask_high_noise.shape[-2], mask_high_noise.shape[-1]
            boost_factor = structural_repulsion_boost - 1.0
            
            def create_spatial_gradient(img1, img2):
                if img1 is None or img2 is None:
                    return None
                
                motion_diff = torch.abs(img2[0] - img1[0]).mean(dim=-1, keepdim=False)
                motion_diff_4d = motion_diff.unsqueeze(0).unsqueeze(0)
                motion_diff_scaled = F.interpolate(
                    motion_diff_4d,
                    size=(mask_h, mask_w),
                    mode='bilinear',
                    align_corners=False
                )
                
                motion_normalized = (motion_diff_scaled - motion_diff_scaled.min()) / (motion_diff_scaled.max() - motion_diff_scaled.min() + 1e-8)
                
                spatial_gradient = 1.0 - motion_normalized * boost_factor * 2.5
                spatial_gradient = torch.clamp(spatial_gradient, 0.02, 1.0)
                return spatial_gradient[0, 0]
            
            if start_image is not None and middle_image is not None and enable_middle_frame:
                start_img = start_image[0:1].to(device)
                mid_img = middle_image[0:1].to(device)
                
                spatial_gradient_1 = create_spatial_gradient(start_img, mid_img)
                
                if spatial_gradient_1 is not None:
                    start_end = start_image.shape[0] + 3
                    mid_protect_start = max(start_end, middle_idx - 4)
                    mid_protect_end = middle_idx + 5
                    transition_end = min(mid_protect_start, length)
                    
                    for frame_idx in range(start_end, transition_end):
                        current_mask = mask_high_noise[:, :, frame_idx, :, :]
                        mask_high_noise[:, :, frame_idx, :, :] = current_mask * spatial_gradient_1
            
            if middle_image is not None and end_image is not None and enable_middle_frame:
                mid_img = middle_image[0:1].to(device)
                end_img = end_image[-1:].to(device)
                
                spatial_gradient_2 = create_spatial_gradient(mid_img, end_img)
                
                if spatial_gradient_2 is not None:
                    mid_protect_end = middle_idx + 5
                    transition_start = mid_protect_end
                    end_start = length - end_image.shape[0]
                    
                    for frame_idx in range(transition_start, end_start):
                        current_mask = mask_high_noise[:, :, frame_idx, :, :]
                        mask_high_noise[:, :, frame_idx, :, :] = current_mask * spatial_gradient_2
            
            if start_image is not None and end_image is not None and (middle_image is None or not enable_middle_frame):
                start_img = start_image[0:1].to(device)
                end_img = end_image[-1:].to(device)
                
                spatial_gradient = create_spatial_gradient(start_img, end_img)
                
                if spatial_gradient is not None:
                    start_end = start_image.shape[0] + 3
                    end_start = length - end_image.shape[0]
                    
                    for frame_idx in range(start_end, end_start):
                        current_mask = mask_high_noise[:, :, frame_idx, :, :]
                        mask_high_noise[:, :, frame_idx, :, :] = current_mask * spatial_gradient
        
        if svi_continue_mode:
            # Second pass: motion_frames is injected into latent first frame
            # start_image used for low noise conditioning as concat image
            image_low = torch.ones((length, height, width, 3), device=device) * 0.5
            
            if start_image is not None:
                image_low[:start_image.shape[0]] = start_image[:, :, :, :3]
                start_latent_frames = ((start_image.shape[0] - 1) // 4) + 1
                mask_low_noise[:, :, :start_latent_frames * 4] = 0.0
            
            concat_latent_image_low = vae.encode(image_low[:, :, :, :3])
        elif latent_continue_mode:
            # LATENT_CONTINUE mode: concat image and concat latent should be the same
            # Use the same image for both high and low noise conditioning
            concat_latent_image_low = concat_latent_image
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
        
        clip_vision_output = cls._merge_clip_vision_outputs(
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
        
        return io.NodeOutput(positive_high_noise, positive_low_noise, negative_out, out_latent,
                trim_latent, trim_image, next_offset)
    
    @classmethod
    def _calculate_aligned_position(cls, ratio, total_frames):
        desired_pixel_idx = int(total_frames * ratio)
        latent_idx = desired_pixel_idx // 4
        aligned_pixel_idx = latent_idx * 4
        aligned_pixel_idx = max(0, min(aligned_pixel_idx, total_frames - 1))
        return aligned_pixel_idx, latent_idx
    
    @classmethod
    def _merge_clip_vision_outputs(cls, *outputs):
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


class WanAdvancedExtractLastFrames(io.ComfyNode):
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanAdvancedExtractLastFrames",
            display_name="Wan Extract Last Frames (Latent)",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.Latent.Input("samples"),
                io.Int.Input("num_frames", default=9, min=0, max=81, step=1, display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.Latent.Output("last_frames"),
            ],
        )
    
    @classmethod
    def execute(cls, samples, num_frames):
        if num_frames == 0:
            out = {"samples": torch.zeros_like(samples["samples"][:, :, :0])}
            return io.NodeOutput(out)
        
        latent_frames = ((num_frames - 1) // 4) + 1
        last_latent = samples["samples"][:, :, -latent_frames:].clone()
        out = {"samples": last_latent}
        return io.NodeOutput(out)


class WanAdvancedExtractLastImages(io.ComfyNode):
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanAdvancedExtractLastImages",
            display_name="Wan Extract Last Images",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("num_frames", default=9, min=0, max=81, step=1, display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.Image.Output("last_images"),
            ],
        )
    
    @classmethod
    def execute(cls, images, num_frames):
        if num_frames == 0:
            last_images = images[:0].clone()
            return io.NodeOutput(last_images)
        
        last_images = images[-num_frames:].clone()
        return io.NodeOutput(last_images)
