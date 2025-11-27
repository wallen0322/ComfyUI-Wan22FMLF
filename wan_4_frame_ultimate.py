from typing_extensions import override
from comfy_api.latest import io
import torch
import torch.nn.functional as F
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
from typing import Optional, Tuple, Any


class WanFourFrameReferenceUltimate(io.ComfyNode):
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent")
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanFourFrameReferenceUltimate",
            display_name="Wan 4-Frame Reference",
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
                io.Image.Input("frame_1_image", optional=True),
                io.Image.Input("frame_2_image", optional=True),
                io.Float.Input("frame_2_ratio", default=0.33, min=0.0, max=1.0, step=0.01, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("frame_2_strength_high", default=0.8, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("frame_2_strength_low", default=0.2, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Combo.Input("enable_frame_2", ["disable", "enable"], default="enable", optional=True),
                io.Image.Input("frame_3_image", optional=True),
                io.Float.Input("frame_3_ratio", default=0.67, min=0.0, max=1.0, step=0.01, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("frame_3_strength_high", default=0.8, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("frame_3_strength_low", default=0.2, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Combo.Input("enable_frame_3", ["disable", "enable"], default="enable", optional=True),
                io.Image.Input("frame_4_image", optional=True),
                io.Float.Input("structural_repulsion_boost", default=1.0, min=1.0, max=2.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True, tooltip="Motion enhancement through spatial gradient conditioning. Only affects high-noise stage."),
                io.ClipVisionOutput.Input("clip_vision_frame_1", optional=True),
                io.ClipVisionOutput.Input("clip_vision_frame_2", optional=True),
                io.ClipVisionOutput.Input("clip_vision_frame_3", optional=True),
                io.ClipVisionOutput.Input("clip_vision_frame_4", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive_high"),
                io.Conditioning.Output(display_name="positive_low"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                mode="NORMAL", frame_1_image=None, frame_2_image=None, frame_2_ratio=0.33,
                frame_2_strength_high=0.8, frame_2_strength_low=0.2, enable_frame_2="enable",
                frame_3_image=None, frame_3_ratio=0.67, frame_3_strength_high=0.8,
                frame_3_strength_low=0.2, enable_frame_3="enable", frame_4_image=None,
                structural_repulsion_boost=1.0,
                clip_vision_frame_1=None, clip_vision_frame_2=None,
                clip_vision_frame_3=None, clip_vision_frame_4=None):
        
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
        
        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()
        
        if frame_1_image is not None:
            image[:frame_1_image.shape[0]] = frame_1_image
            mask_high_noise[:, :, :frame_1_image.shape[0] + 3] = 0.0
            mask_low_noise[:, :, :frame_1_image.shape[0] + 3] = 0.0
        
        if frame_2_image is not None and enable_frame_2 == "enable":
            image[frame_2_idx:frame_2_idx + frame_2_image.shape[0]] = frame_2_image
            start_range = max(0, frame_2_idx)
            end_range = min(length, frame_2_idx + frame_2_image.shape[0] + 3)
            
            mask_high_value = 1.0 - frame_2_strength_high
            mask_high_noise[:, :, start_range:end_range] = mask_high_value
            
            mask_low_value = 1.0 - frame_2_strength_low
            mask_low_noise[:, :, start_range:end_range] = mask_low_value
        
        if frame_3_image is not None and enable_frame_3 == "enable":
            image[frame_3_idx:frame_3_idx + frame_3_image.shape[0]] = frame_3_image
            start_range = max(0, frame_3_idx)
            end_range = min(length, frame_3_idx + frame_3_image.shape[0] + 3)
            
            mask_high_value = 1.0 - frame_3_strength_high
            mask_high_noise[:, :, start_range:end_range] = mask_high_value
            
            mask_low_value = 1.0 - frame_3_strength_low
            mask_low_noise[:, :, start_range:end_range] = mask_low_value
        
        if frame_4_image is not None:
            image[frame_4_idx:frame_4_idx + frame_4_image.shape[0]] = frame_4_image
            mask_high_noise[:, :, frame_4_idx:frame_4_idx + frame_4_image.shape[0] + 3] = 0.0
            mask_low_noise[:, :, frame_4_idx:frame_4_idx + frame_4_image.shape[0] + 3] = 0.0
        
        concat_latent_image_high = vae.encode(image[:, :, :, :3])
        
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
            
            frames = [
                (frame_1_image, frame_1_idx) if frame_1_image is not None else (None, None),
                (frame_2_image, frame_2_idx) if frame_2_image is not None and enable_frame_2 == "enable" else (None, None),
                (frame_3_image, frame_3_idx) if frame_3_image is not None and enable_frame_3 == "enable" else (None, None),
                (frame_4_image, frame_4_idx) if frame_4_image is not None else (None, None),
            ]
            
            valid_frames = [(img, idx) for img, idx in frames if img is not None and idx is not None]
            
            for i in range(len(valid_frames) - 1):
                img1, pos1 = valid_frames[i]
                img2, pos2 = valid_frames[i + 1]
                
                if pos2 > pos1 + 4:
                    start_end = pos1 + 4
                    end_start = pos2
                    protect_start = pos2 - 4
                    
                    spatial_gradient = create_spatial_gradient(img1[0:1].to(device), img2[0:1].to(device))
                    
                    if spatial_gradient is not None:
                        transition_end = min(protect_start, end_start)
                        
                        for frame_idx in range(start_end, transition_end):
                            current_mask = mask_high_noise[:, :, frame_idx, :, :]
                            mask_high_noise[:, :, frame_idx, :, :] = current_mask * spatial_gradient
        
        if mode == "SINGLE_PERSON":
            mask_low_noise = mask_base.clone()
            if frame_1_image is not None:
                mask_low_noise[:, :, :frame_1_image.shape[0] + 3] = 0.0
            
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if frame_1_image is not None:
                image_low_only[:frame_1_image.shape[0]] = frame_1_image
            concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
        else:
            frame_2_strength = frame_2_strength_low if enable_frame_2 == "enable" else 0.0
            frame_3_strength = frame_3_strength_low if enable_frame_3 == "enable" else 0.0
            
            if frame_2_strength == 0.0 or frame_3_strength == 0.0:
                image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
                
                if frame_1_image is not None:
                    image_low_only[:frame_1_image.shape[0]] = frame_1_image
                
                if frame_2_image is not None and frame_2_strength > 0.0:
                    image_low_only[frame_2_idx:frame_2_idx + frame_2_image.shape[0]] = frame_2_image
                
                if frame_3_image is not None and frame_3_strength > 0.0:
                    image_low_only[frame_3_idx:frame_3_idx + frame_3_image.shape[0]] = frame_3_image
                
                if frame_4_image is not None:
                    image_low_only[frame_4_idx:frame_4_idx + frame_4_image.shape[0]] = frame_4_image
                
                concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
            else:
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
        
        
        negative_out = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent_image_high,
            "concat_mask": mask_high_reshaped
        })
        
        clip_vision_output = cls._merge_clip_vision_outputs(
            clip_vision_frame_1, clip_vision_frame_2, 
            clip_vision_frame_3, clip_vision_frame_4
        )
        
        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise, 
                                                                   {"clip_vision_output": clip_vision_output})
            
            negative_out = node_helpers.conditioning_set_values(negative_out, 
                                                             {"clip_vision_output": clip_vision_output})
        
        return io.NodeOutput(positive_high_noise, positive_low_noise, negative_out, {"samples": latent})

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
