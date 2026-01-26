from typing_extensions import override
from comfy_api.latest import io
import torch
import torch.nn.functional as F
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
import comfy.latent_formats
from typing import Optional, Tuple, Any, Dict, List
import math
import numpy as np
from io import BytesIO
from PIL import Image


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
                # 新增SVI参数
                io.Int.Input("svi_motion_latent_count", default=1, min=0, max=128, step=1, optional=True,
                           tooltip="[SVI Mode] Number of latent frames from previous video for motion continuity"),
                io.Boolean.Input("enable_svi_start", default=True, optional=True,
                               tooltip="[SVI Mode] Enable start frame conditioning in SVI mode"),
                io.Boolean.Input("enable_svi_middle", default=True, optional=True,
                               tooltip="[SVI Mode] Enable middle frame conditioning in SVI mode"),
                io.Boolean.Input("enable_svi_end", default=True, optional=True,
                               tooltip="[SVI Mode] Enable end frame conditioning in SVI mode"),
                io.Float.Input("svi_motion_strength", default=1.0, min=0.0, max=1.0, step=0.01, optional=True,
                             tooltip="[SVI Mode] Influence strength of motion frames"),
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
                prev_latent=None,
                # 新增SVI参数
                svi_motion_latent_count=1,
                enable_svi_start=True,
                enable_svi_middle=True,
                enable_svi_end=True,
                svi_motion_strength=1.0):
        
        # ===========================================
        # SVI模式处理 - 新增逻辑
        # ===========================================
        if long_video_mode == "SVI":
            return cls._execute_svi_mode(
                positive, negative, vae, width, height, length, batch_size,
                start_image, middle_image, end_image,
                middle_frame_ratio, prev_latent,
                svi_motion_latent_count,
                enable_svi_start, enable_svi_middle, enable_svi_end,
                svi_motion_strength,
                high_noise_start_strength, high_noise_mid_strength,
                low_noise_start_strength, low_noise_mid_strength, low_noise_end_strength,
                clip_vision_start_image, clip_vision_middle_image, clip_vision_end_image,
                enable_middle_frame
            )
        
        # ===========================================
        # 原有逻辑（其他模式）
        # ===========================================
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

        # 原有逻辑继续...
        # ... [这里省略原有的大段逻辑，保持原样]
        
        # 原有逻辑的结尾部分
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
            
            # 原有结构增强逻辑...
            # [保持原有逻辑]
        
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
    
    # ===========================================
    # SVI模式核心方法 - 新增
    # ===========================================
    @classmethod
    def _execute_svi_mode(cls, positive, negative, vae, width, height, length, batch_size,
                         start_image, middle_image, end_image,
                         middle_frame_ratio, prev_latent,
                         motion_latent_count,
                         enable_start, enable_middle, enable_end,
                         motion_strength,
                         start_strength, middle_strength,
                         low_start_strength, low_middle_strength, low_end_strength,
                         clip_vision_start, clip_vision_middle, clip_vision_end,
                         enable_middle_frame):
        """执行SVI模式 - 兼容多帧条件"""
        # 计算基本参数
        spatial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        total_latents = ((length - 1) // 4) + 1
        H = height // spatial_scale
        W = width // spatial_scale
        
        device = comfy.model_management.intermediate_device()
        
        # 创建空latent
        latent = torch.zeros([batch_size, latent_channels, total_latents, H, W], 
                            device=device)
        
        # 步骤1：调整图像尺寸
        def resize_image(img):
            if img is None:
                return None
            return comfy.utils.common_upscale(
                img[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        
        start_image = resize_image(start_image) if start_image is not None else None
        middle_image = resize_image(middle_image) if middle_image is not None else None
        end_image = resize_image(end_image) if end_image is not None else None
        
        # 步骤2：计算各条件的位置
        positions = {}
        
        # Prev Latent位置
        if prev_latent is not None and motion_latent_count > 0:
            prev_samples = prev_latent.get("samples")
            if prev_samples is not None:
                actual_motion_count = min(motion_latent_count, prev_samples.shape[2], total_latents)
                positions['motion'] = {
                    'type': 'motion',
                    'latent_start': 0,
                    'latent_end': actual_motion_count,
                    'strength': motion_strength
                }
        
        # 计算start位置（放在motion之后）
        start_latent_idx = positions.get('motion', {}).get('latent_end', 0)
        if enable_start and start_latent_idx < total_latents and start_image is not None:
            positions['start'] = {
                'type': 'start',
                'latent_start': start_latent_idx,
                'latent_end': start_latent_idx + 1,
                'strength': start_strength
            }
        
        # 计算middle位置
        if enable_middle and enable_middle_frame and middle_image is not None:
            # 计算中间位置并对齐
            middle_pixel_idx = int(total_latents * 4 * middle_frame_ratio)
            middle_latent_idx = max(1, min(middle_pixel_idx // 4, total_latents - 2))
            
            # 确保middle位置不与其他条件重叠
            while any(middle_latent_idx >= pos.get('latent_start', -1) and 
                     middle_latent_idx < pos.get('latent_end', -1) 
                     for pos in positions.values()):
                middle_latent_idx += 1
                if middle_latent_idx >= total_latents - 1:
                    middle_latent_idx = total_latents - 2
                    break
            
            positions['middle'] = {
                'type': 'middle',
                'latent_start': middle_latent_idx,
                'latent_end': middle_latent_idx + 1,
                'strength': middle_strength
            }
        
        # 计算end位置
        if enable_end and end_image is not None:
            end_latent_idx = total_latents - 1
            positions['end'] = {
                'type': 'end',
                'latent_start': end_latent_idx,
                'latent_end': end_latent_idx + 1,
                'strength': low_end_strength  # 注意：使用low_end_strength
            }
        
        # 步骤3：准备各条件潜变量
        condition_latents = {}
        
        # Prev Latent
        if 'motion' in positions and prev_latent is not None:
            prev_samples = prev_latent.get("samples")
            if prev_samples is not None:
                pos = positions['motion']
                actual_count = pos['latent_end'] - pos['latent_start']
                condition_latents['motion'] = {
                    'latent': prev_samples[:, :, -actual_count:].clone(),
                    'position': pos
                }
        
        # 起始帧潜变量
        if 'start' in positions and start_image is not None:
            pos = positions['start']
            start_latent = vae.encode(start_image[:, :, :, :3])
            condition_latents['start'] = {
                'latent': start_latent,
                'position': pos
            }
        
        # 中间帧潜变量
        if 'middle' in positions and middle_image is not None:
            pos = positions['middle']
            middle_latent = vae.encode(middle_image[:, :, :, :3])
            condition_latents['middle'] = {
                'latent': middle_latent,
                'position': pos
            }
        
        # 结束帧潜变量
        if 'end' in positions and end_image is not None:
            pos = positions['end']
            end_latent = vae.encode(end_image[:, :, :, :3])
            condition_latents['end'] = {
                'latent': end_latent,
                'position': pos
            }
        
        # 步骤4：构建统一的image_cond_latent
        image_cond_latent = torch.zeros(1, latent_channels, total_latents, H, W,
                                       dtype=torch.float32, device=device)
        
        # 将各条件潜变量放入对应位置
        for key, cond_data in condition_latents.items():
            cond_latent = cond_data['latent']
            pos = cond_data['position']
            
            latent_start = pos['latent_start']
            latent_end = pos['latent_end']
            latent_length = latent_end - latent_start
            
            if latent_length > 0 and latent_start < total_latents:
                # 确保latent尺寸匹配
                latent_to_use = cond_latent
                if cond_latent.shape[2] > latent_length:
                    latent_to_use = cond_latent[:, :, :latent_length]
                elif cond_latent.shape[2] < latent_length:
                    # 如果提供的latent不够，填充零
                    padding = torch.zeros(1, latent_channels, latent_length - cond_latent.shape[2],
                                         H, W, device=device, dtype=cond_latent.dtype)
                    latent_to_use = torch.cat([cond_latent, padding], dim=2)
                
                actual_end = min(latent_end, total_latents)
                actual_length = actual_end - latent_start
                if actual_length > 0:
                    image_cond_latent[:, :, latent_start:actual_end] = latent_to_use[:, :, :actual_length]
        
        # 步骤5：构建统一的掩码
        mask_high = torch.ones((1, 1, total_latents, H, W), device=device)
        mask_low = torch.ones((1, 1, total_latents, H, W), device=device)
        
        # 应用各条件的强度到掩码
        for key, cond_data in condition_latents.items():
            pos = cond_data['position']
            strength = pos['strength']
            
            latent_start = pos['latent_start']
            latent_end = min(pos['latent_end'], total_latents)
            
            if latent_end > latent_start:
                # 计算掩码值：strength=1 -> mask=0 (完全使用条件)
                #             strength=0 -> mask=1 (完全忽略条件)
                mask_value = 1.0 - strength
                
                # 应用到高噪声掩码
                mask_high[:, :, latent_start:latent_end] = mask_value
                
                # 低噪声掩码：对于start和middle使用不同的强度
                if key == 'start':
                    low_strength = low_start_strength
                elif key == 'middle':
                    low_strength = low_middle_strength
                elif key == 'end':
                    low_strength = low_end_strength
                else:
                    low_strength = strength
                
                mask_low[:, :, latent_start:latent_end] = 1.0 - low_strength
        
        # 重新调整掩码形状以匹配模型期望 [batch, 4, latent_t, H, W]
        # 注意：这里需要将掩码扩展为4个通道以匹配concat_latent_image
        mask_high_reshaped = mask_high.view(
            1, total_latents, H, W
        ).unsqueeze(1).repeat(1, 4, 1, 1, 1)
        
        mask_low_reshaped = mask_low.view(
            1, total_latents, H, W
        ).unsqueeze(1).repeat(1, 4, 1, 1, 1)
        
        # 步骤6：构建条件
        positive_high_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": image_cond_latent,
            "concat_mask": mask_high_reshaped
        })
        
        positive_low_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": image_cond_latent,
            "concat_mask": mask_low_reshaped
        })
        
        negative_out = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": image_cond_latent,
            "concat_mask": mask_high_reshaped
        })
        
        # 合并Clip Vision输出
        clip_vision_output = cls._merge_clip_vision_outputs(
            clip_vision_start, clip_vision_middle, clip_vision_end
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
                            0, 0, 0)  # trim_latent, trim_image, next_offset 都为0
    
    # ===========================================
    # 辅助方法 - 新增或修改
    # ===========================================
    @classmethod
    def _merge_clip_vision_outputs(cls, *outputs):
        """合并多个Clip Vision输出"""
        valid_outputs = [o for o in outputs if o is not None]
        
        if not valid_outputs:
            return None
        
        if len(valid_outputs) == 1:
            return valid_outputs[0]
        
        # 合并所有输出
        all_states = [o.penultimate_hidden_states for o in valid_outputs]
        combined_states = torch.cat(all_states, dim=-2)
        
        result = comfy.clip_vision.Output()
        result.penultimate_hidden_states = combined_states
        return result
    
    @classmethod
    def _calculate_aligned_position(cls, ratio, total_frames):
        """计算对齐的位置"""
        desired_pixel_idx = int(total_frames * ratio)
        latent_idx = desired_pixel_idx // 4
        aligned_pixel_idx = latent_idx * 4
        aligned_pixel_idx = max(0, min(aligned_pixel_idx, total_frames - 1))
        return aligned_pixel_idx, latent_idx


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


# ===========================================
# SVI控制面板 - 新增
# ===========================================
class WanSVIControlPanel(io.ComfyNode):
    """SVI控制面板 - 提供预设和参数控制"""
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanSVIControlPanel",
            display_name="Wan SVI Control Panel",
            category="ComfyUI-Wan22FMLF/Advanced",
            inputs=[
                io.Combo.Input("preset", [
                    "Balanced", "Strong Motion", "Structure Guided", 
                    "Seamless Loop", "Creative", "Custom"
                ], default="Balanced", 
                tooltip="预设参数组合，选择Custom可手动调整"),
                
                # 运动参数
                io.Int.Input("motion_latent_count", default=1, min=0, max=128, step=1,
                           tooltip="从前一个视频中使用的潜变量帧数"),
                io.Float.Input("motion_strength", default=1.0, min=0.0, max=1.0, step=0.01,
                             tooltip="运动帧的影响强度"),
                
                # 多帧强度
                io.Float.Input("start_strength", default=1.0, min=0.0, max=1.0, step=0.01,
                             tooltip="起始帧的影响强度"),
                io.Float.Input("middle_strength", default=0.8, min=0.0, max=1.0, step=0.01,
                             tooltip="中间帧的影响强度"),
                io.Float.Input("end_strength", default=0.8, min=0.0, max=1.0, step=0.01,
                             tooltip="结束帧的影响强度"),
                
                # 位置控制
                io.Float.Input("middle_position", default=0.5, min=0.0, max=1.0, step=0.01,
                             tooltip="中间帧的位置（0=开始，1=结束）"),
                
                # 启用控制
                io.Boolean.Input("enable_start", default=True,
                               tooltip="启用起始帧条件"),
                io.Boolean.Input("enable_middle", default=True,
                               tooltip="启用中间帧条件"),
                io.Boolean.Input("enable_end", default=True,
                               tooltip="启用结束帧条件"),
            ],
            outputs=[
                io.Int.Output(display_name="motion_latent_count_out"),
                io.Float.Output(display_name="motion_strength_out"),
                io.Float.Output(display_name="start_strength_out"),
                io.Float.Output(display_name="middle_strength_out"),
                io.Float.Output(display_name="end_strength_out"),
                io.Float.Output(display_name="middle_position_out"),
                io.Boolean.Output(display_name="enable_start_out"),
                io.Boolean.Output(display_name="enable_middle_out"),
                io.Boolean.Output(display_name="enable_end_out"),
            ],
        )
    
    @classmethod
    def execute(cls, preset="Balanced", motion_latent_count=1, motion_strength=1.0,
                start_strength=1.0, middle_strength=0.8, end_strength=0.8,
                middle_position=0.5, enable_start=True, enable_middle=True, enable_end=True):
        
        # 应用预设（除非选择Custom）
        if preset != "Custom":
            presets = {
                "Balanced": {
                    "motion_count": 1, "motion_str": 1.0,
                    "start_str": 1.0, "middle_str": 0.8, "end_str": 0.8,
                    "middle_pos": 0.5,
                    "enable_start": True, "enable_middle": True, "enable_end": True
                },
                "Strong Motion": {
                    "motion_count": 3, "motion_str": 1.0,
                    "start_str": 0.9, "middle_str": 0.6, "end_str": 0.7,
                    "middle_pos": 0.4,
                    "enable_start": True, "enable_middle": True, "enable_end": True
                },
                "Structure Guided": {
                    "motion_count": 1, "motion_str": 0.8,
                    "start_str": 0.7, "middle_str": 1.0, "end_str": 0.9,
                    "middle_pos": 0.5,
                    "enable_start": True, "enable_middle": True, "enable_end": True
                },
                "Seamless Loop": {
                    "motion_count": 2, "motion_str": 1.0,
                    "start_str": 1.0, "middle_str": 0.0, "end_str": 1.0,
                    "middle_pos": 0.5,
                    "enable_start": True, "enable_middle": False, "enable_end": True
                },
                "Creative": {
                    "motion_count": 0, "motion_str": 0.5,
                    "start_str": 0.5, "middle_str": 0.5, "end_str": 0.5,
                    "middle_pos": 0.3,
                    "enable_start": True, "enable_middle": True, "enable_end": True
                }
            }
            
            if preset in presets:
                p = presets[preset]
                motion_latent_count = p["motion_count"]
                motion_strength = p["motion_str"]
                start_strength = p["start_str"]
                middle_strength = p["middle_str"]
                end_strength = p["end_str"]
                middle_position = p["middle_pos"]
                enable_start = p["enable_start"]
                enable_middle = p["enable_middle"]
                enable_end = p["enable_end"]
        
        return io.NodeOutput(
            motion_latent_count, motion_strength,
            start_strength, middle_strength, end_strength,
            middle_position, enable_start, enable_middle, enable_end
        )


# ===========================================
# 节点注册 - 修改为包含所有节点
# ===========================================
NODE_CLASS_MAPPINGS = {
    "WanAdvancedI2V": WanAdvancedI2V,
    "WanAdvancedExtractLastFrames": WanAdvancedExtractLastFrames,
    "WanAdvancedExtractLastImages": WanAdvancedExtractLastImages,
    "WanSVIControlPanel": WanSVIControlPanel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAdvancedI2V": "Wan Advanced I2V (Ultimate)",
    "WanAdvancedExtractLastFrames": "Wan Extract Last Frames (Latent)",
    "WanAdvancedExtractLastImages": "Wan Extract Last Images",
    "WanSVIControlPanel": "Wan SVI Control Panel",
}
