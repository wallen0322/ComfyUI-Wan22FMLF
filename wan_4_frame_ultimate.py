# -*- coding: utf-8 -*-
"""
Wan 4-Frame Reference Node - Ultimate Version
- Adjustable gray level for placeholder
- Perfect mask-latent alignment algorithm
- No noise (removed - causes artifacts)
"""

import torch
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
from nodes import MAX_RESOLUTION


class WanFourFrameReferenceUltimate:
    """
    4-frame reference node with perfect alignment.
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
                
                # 🎨 占位颜色 (保留可调节)
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
                "frame_1_image": ("IMAGE",),
                "frame_2_image": ("IMAGE",),
                "frame_2_ratio": ("FLOAT", {
                    "default": 0.33, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Frame 2 position (0.0-1.0)"
                }),
                "frame_2_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Frame 2 strength. 1.0=fully fixed"
                }),
                "enable_frame_2": (["disable", "enable"], {"default": "enable"}),
                
                "frame_3_image": ("IMAGE",),
                "frame_3_ratio": ("FLOAT", {
                    "default": 0.67, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Frame 3 position (0.0-1.0)"
                }),
                "frame_3_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Frame 3 strength. 1.0=fully fixed"
                }),
                "enable_frame_3": (["disable", "enable"], {"default": "enable"}),
                
                "frame_4_image": ("IMAGE",),
                "end_frame_offset": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 64,
                    "step": 4,
                    "tooltip": "Move end frame forward by N frames"
                }),
                
                "clip_vision_frame_1": ("CLIP_VISION_OUTPUT",),
                "clip_vision_frame_2": ("CLIP_VISION_OUTPUT",),
                "clip_vision_frame_3": ("CLIP_VISION_OUTPUT",),
                "clip_vision_frame_4": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "ComfyUI-Wan22FMLF/video"
    DESCRIPTION = "4-frame reference with perfect alignment (no noise)"

    def generate(self, positive, negative, vae, width, height, length, batch_size,
                 placeholder_gray_level=0.5,
                 frame_1_image=None, 
                 frame_2_image=None, frame_2_ratio=0.33, frame_2_strength=0.5, enable_frame_2="enable",
                 frame_3_image=None, frame_3_ratio=0.67, frame_3_strength=0.5, enable_frame_3="enable",
                 frame_4_image=None, end_frame_offset=16,
                 clip_vision_frame_1=None, clip_vision_frame_2=None, 
                 clip_vision_frame_3=None, clip_vision_frame_4=None):
        
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1  # 🔑 VAE时间压缩比 4:1
        device = comfy.model_management.intermediate_device()
        
        latent = torch.zeros([batch_size, latent_channels, latent_t, 
                             height // spacial_scale, width // spacial_scale], device=device)
        
        print(f"📐 Video: {length} frames → Latent: {latent_t} frames (4:1 compression)")
        
        # Resize images
        if frame_1_image is not None:
            frame_1_image = comfy.utils.common_upscale(frame_1_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if frame_2_image is not None:
            frame_2_image = comfy.utils.common_upscale(frame_2_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if frame_3_image is not None:
            frame_3_image = comfy.utils.common_upscale(frame_3_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if frame_4_image is not None:
            frame_4_image = comfy.utils.common_upscale(frame_4_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        
        # 🔑 创建timeline (纯灰色,无噪声)
        image = torch.ones((length, height, width, 3), device=device) * placeholder_gray_level
        mask = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), device=device)
        
        # ═══════════════════════════════════════════════════════════
        # 🧮 智能对齐算法: 确保pixel frame和latent frame对齐
        # ═══════════════════════════════════════════════════════════
        
        def calculate_aligned_position(ratio, total_frames):
            """
            根据ratio计算pixel frame位置,并确保对齐到latent边界
            
            原理:
            - Wan VAE是4:1压缩,每个latent帧对应4个pixel帧
            - 要让参考帧稳定,必须让它对齐到latent帧的起始位置
            - 例如: pixel frame 32 → latent frame 8 ✅
            -       pixel frame 33 → latent frame 8 但mask会错位! ❌
            """
            # 1. 根据ratio计算期望位置
            desired_pixel_idx = int(total_frames * ratio)
            
            # 2. 计算对应的latent索引
            latent_idx = desired_pixel_idx // 4
            
            # 3. 对齐到latent边界 (latent_idx * 4 就是对齐后的pixel位置)
            aligned_pixel_idx = latent_idx * 4
            
            # 4. 确保在有效范围内
            aligned_pixel_idx = max(0, min(aligned_pixel_idx, total_frames - 1))
            
            return aligned_pixel_idx, latent_idx
        
        # 计算各帧位置
        frame_1_idx = 0
        frame_1_latent_idx = 0
        
        frame_2_idx, frame_2_latent_idx = calculate_aligned_position(frame_2_ratio, length)
        frame_3_idx, frame_3_latent_idx = calculate_aligned_position(frame_3_ratio, length)
        
        # 尾帧处理:先计算,然后确保不冲突
        frame_4_idx_raw = length - 1 - end_frame_offset
        frame_4_idx, frame_4_latent_idx = calculate_aligned_position(frame_4_idx_raw / length, length)
        
        # 确保顺序: frame_1 < frame_2 < frame_3 < frame_4
        if frame_2_idx <= frame_1_idx + 4:
            frame_2_idx = frame_1_idx + 4
            frame_2_latent_idx = frame_2_idx // 4
        
        if frame_3_idx <= frame_2_idx + 4:
            frame_3_idx = frame_2_idx + 4
            frame_3_latent_idx = frame_3_idx // 4
        
        if frame_4_idx <= frame_3_idx + 4:
            frame_4_idx = frame_3_idx + 4
            frame_4_latent_idx = frame_4_idx // 4
        
        print(f"🔧 Alignment check:")
        print(f"   Frame 1: pixel[{frame_1_idx}] → latent[{frame_1_latent_idx}] ✅")
        print(f"   Frame 2: pixel[{frame_2_idx}] → latent[{frame_2_latent_idx}] ✅")
        print(f"   Frame 3: pixel[{frame_3_idx}] → latent[{frame_3_latent_idx}] ✅")
        print(f"   Frame 4: pixel[{frame_4_idx}] → latent[{frame_4_latent_idx}] ✅")
        
        # ═══════════════════════════════════════════════════════════
        # 🔵 Frame 1: 放置参考帧
        # ═══════════════════════════════════════════════════════════
        if frame_1_image is not None:
            image[:frame_1_image.shape[0]] = frame_1_image
            mask[:, :, :frame_1_image.shape[0] + 3] = 0.0
            print(f"📌 Frame 1 at pixel[{frame_1_idx}]: mask[0:{frame_1_image.shape[0] + 3}]")
        
        # ═══════════════════════════════════════════════════════════
        # 🟡 Frame 2
        # ═══════════════════════════════════════════════════════════
        if frame_2_image is not None and enable_frame_2 == "enable":
            image[frame_2_idx:frame_2_idx + frame_2_image.shape[0]] = frame_2_image
            mask_value = 1.0 - frame_2_strength
            # 🔑 mask覆盖范围: 从frame_2_idx开始的4帧
            mask[:, :, frame_2_idx:frame_2_idx + 4] = mask_value
            print(f"📌 Frame 2 at pixel[{frame_2_idx}]: latent[{frame_2_latent_idx}], mask[{frame_2_idx}:{frame_2_idx+4}], strength={frame_2_strength:.2f}")
        elif frame_2_image is not None:
            print(f"⚪ Frame 2 (DISABLED)")
        
        # ═══════════════════════════════════════════════════════════
        # 🟠 Frame 3
        # ═══════════════════════════════════════════════════════════
        if frame_3_image is not None and enable_frame_3 == "enable":
            image[frame_3_idx:frame_3_idx + frame_3_image.shape[0]] = frame_3_image
            mask_value = 1.0 - frame_3_strength
            mask[:, :, frame_3_idx:frame_3_idx + 4] = mask_value
            print(f"📌 Frame 3 at pixel[{frame_3_idx}]: latent[{frame_3_latent_idx}], mask[{frame_3_idx}:{frame_3_idx+4}], strength={frame_3_strength:.2f}")
        elif frame_3_image is not None:
            print(f"⚪ Frame 3 (DISABLED)")
        
        # ═══════════════════════════════════════════════════════════
        # 🔴 Frame 4
        # ═══════════════════════════════════════════════════════════
        if frame_4_image is not None:
            image[frame_4_idx:frame_4_idx + frame_4_image.shape[0]] = frame_4_image
            mask[:, :, frame_4_idx:frame_4_idx + 4] = 0.0
            free_frames = length - 1 - frame_4_idx
            print(f"📌 Frame 4 at pixel[{frame_4_idx}]: latent[{frame_4_latent_idx}], mask[{frame_4_idx}:{frame_4_idx+4}], {free_frames} frames after")
        
        print(f"🎨 Placeholder: gray_level={placeholder_gray_level:.2f} (no noise)")
        
        # Encode timeline
        concat_latent_image = vae.encode(image[:, :, :, :3])
        
        # Reshape mask
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        
        # Set conditioning
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        
        # CLIP Vision
        clip_vision_output = self._merge_clip_vision_outputs(clip_vision_frame_1, clip_vision_frame_2, clip_vision_frame_3, clip_vision_frame_4)
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})
        
        return (positive, negative, {"samples": latent})

    def _merge_clip_vision_outputs(self, *outputs):
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
NODE_DISPLAY_NAME_MAPPINGS = {"WanFourFrameReferenceUltimate": "Wan 4-Frame Reference 🎨"}
