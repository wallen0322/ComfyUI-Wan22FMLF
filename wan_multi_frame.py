from typing_extensions import override
from comfy_api.latest import io
import torch
import torch.nn.functional as F
import json
from typing import List, Tuple, Optional, Any
import node_helpers
import comfy
import comfy.utils


class WanMultiFrameRefToVideo(io.ComfyNode):
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent")
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanMultiFrameRefToVideo",
            display_name="Wan Multi-Frame Reference",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=480, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number),
                io.Int.Input("length", default=81, min=1, max=8192, step=4, display_mode=io.NumberDisplay.number),
                io.Int.Input("batch_size", default=1, min=1, max=4096, display_mode=io.NumberDisplay.number),
                io.Image.Input("ref_images"),
                io.Combo.Input("mode", ["NORMAL", "SINGLE_PERSON"], default="NORMAL", optional=True),
                io.String.Input("ref_positions", default="", optional=True),
                io.Float.Input("ref_strength_high", default=0.8, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("ref_strength_low", default=0.2, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("end_frame_strength_high", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("end_frame_strength_low", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.Float.Input("structural_repulsion_boost", default=1.0, min=1.0, max=2.0, step=0.05, round=0.01, display_mode=io.NumberDisplay.slider, optional=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive_high"),
                io.Conditioning.Output(display_name="positive_low"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, ref_images,
                mode="NORMAL", ref_positions="", ref_strength_high=0.8, ref_strength_low=0.2,
                end_frame_strength_high=1.0, end_frame_strength_low=1.0, structural_repulsion_boost=1.0, clip_vision_output=None):
        
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        device = comfy.model_management.intermediate_device()

        latent = torch.zeros([batch_size, latent_channels, latent_t,
                             height // spacial_scale, width // spacial_scale], device=device)
        
        imgs = cls._resize_images(ref_images, width, height, device)
        n_imgs = imgs.shape[0]
        positions = cls._parse_positions(ref_positions, n_imgs, length)
        
        def align_position(pos: int, total_frames: int) -> int:
            latent_idx = pos // 4
            aligned_pos = latent_idx * 4
            aligned_pos = max(0, min(aligned_pos, total_frames - 1))
            return aligned_pos

        aligned_positions = [align_position(int(p), length) for p in positions]

        for i in range(1, len(aligned_positions)):
            if aligned_positions[i] <= aligned_positions[i-1] + 3:
                aligned_positions[i] = min(aligned_positions[i-1] + 4, length - 1)

        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), device=device)

        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()

        for i, pos in enumerate(aligned_positions):
            frame_idx = int(pos)

            if i == 0:
                image[frame_idx:frame_idx + 1] = imgs[i]
                mask_high_noise[:, :, frame_idx:frame_idx + 4] = 0.0
                mask_low_noise[:, :, frame_idx:frame_idx + 4] = 0.0
            elif i == n_imgs - 1:
                image[-1:] = imgs[i]

                mask_high_value = 1.0 - end_frame_strength_high
                mask_high_noise[:, :, -4:] = mask_high_value

                mask_low_value = 1.0 - end_frame_strength_low
                mask_low_noise[:, :, -4:] = mask_low_value
            else:
                image[frame_idx:frame_idx + 1] = imgs[i]
                start_range = max(0, frame_idx)
                end_range = min(length, frame_idx + 4)

                mask_high_value = 1.0 - ref_strength_high
                mask_high_noise[:, :, start_range:end_range] = mask_high_value

                mask_low_value = 1.0 - ref_strength_low
                mask_low_noise[:, :, start_range:end_range] = mask_low_value

        if mode == "SINGLE_PERSON":
            concat_latent_image_high = vae.encode(image[:, :, :, :3])
        else:
            need_selective_image_high = (ref_strength_high == 0.0) or (end_frame_strength_high == 0.0)

            if need_selective_image_high:
                image_high_only = torch.ones((length, height, width, 3), device=device) * 0.5

                if n_imgs >= 1:
                    frame_idx_first = int(aligned_positions[0])
                    image_high_only[frame_idx_first:frame_idx_first + 1] = imgs[0]

                if ref_strength_high > 0.0:
                    for i in range(1, n_imgs - 1):
                        frame_idx_mid = int(aligned_positions[i])
                        image_high_only[frame_idx_mid:frame_idx_mid + 1] = imgs[i]

                if n_imgs >= 2 and end_frame_strength_high > 0.0:
                    image_high_only[-1:] = imgs[-1]

                concat_latent_image_high = vae.encode(image_high_only[:, :, :, :3])
            else:
                concat_latent_image_high = vae.encode(image[:, :, :, :3])

        if structural_repulsion_boost > 1.001 and length > 4 and n_imgs >= 2:
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
            
            for i in range(n_imgs - 1):
                pos1 = int(aligned_positions[i])
                pos2 = int(aligned_positions[i + 1])
                
                if pos2 > pos1 + 4:
                    start_end = pos1 + 4
                    end_start = pos2
                    protect_start = pos2 - 4
                    
                    img1 = imgs[i:i+1].to(device)
                    img2 = imgs[i+1:i+2].to(device)
                    
                    spatial_gradient = create_spatial_gradient(img1, img2)
                    
                    if spatial_gradient is not None:
                        transition_end = min(protect_start, end_start)
                        
                        for frame_idx in range(start_end, transition_end):
                            current_mask = mask_high_noise[:, :, frame_idx, :, :]
                            mask_high_noise[:, :, frame_idx, :, :] = current_mask * spatial_gradient

        if mode == "SINGLE_PERSON":
            mask_low_noise = mask_base.clone()
            if n_imgs >= 1:
                frame_idx_first = int(aligned_positions[0])
                mask_low_noise[:, :, frame_idx_first:frame_idx_first + 4] = 0.0

            if n_imgs >= 2:
                mask_low_value = 1.0 - end_frame_strength_low
                mask_low_noise[:, :, -4:] = mask_low_value

            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if n_imgs >= 1:
                frame_idx_first = int(aligned_positions[0])
                image_low_only[frame_idx_first:frame_idx_first + 1] = imgs[0]

            if n_imgs >= 2 and end_frame_strength_low > 0.0:
                image_low_only[-1:] = imgs[-1]

            concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
        else:
            need_selective_image = (ref_strength_low == 0.0) or (end_frame_strength_low == 0.0)

            if need_selective_image:
                image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5

                if n_imgs >= 1:
                    frame_idx_first = int(aligned_positions[0])
                    image_low_only[frame_idx_first:frame_idx_first + 1] = imgs[0]

                if ref_strength_low > 0.0:
                    for i in range(1, n_imgs - 1):
                        frame_idx_mid = int(aligned_positions[i])
                        image_low_only[frame_idx_mid:frame_idx_mid + 1] = imgs[i]

                if n_imgs >= 2 and end_frame_strength_low > 0.0:
                    image_low_only[-1:] = imgs[-1]

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
        
        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise,
                                                                   {"clip_vision_output": clip_vision_output})
            
            negative_out = node_helpers.conditioning_set_values(negative_out, 
                                                             {"clip_vision_output": clip_vision_output})
        
        return io.NodeOutput(positive_high_noise, positive_low_noise, negative_out, {"samples": latent})

    @classmethod
    def _resize_images(cls, images, width, height, device):
        images = images.to(device)
        x = images.movedim(-1, 1)
        x = comfy.utils.common_upscale(x, width, height, "bilinear", "center")
        x = x.movedim(1, -1)

        if x.shape[-1] == 4:
            x = x[..., :3]

        return x

    @classmethod
    def _parse_positions(cls, pos_str, n_imgs, length):
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
