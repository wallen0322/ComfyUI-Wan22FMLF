# -*- coding: utf-8 -*-
"""
Wan SVI Pro Advanced I2V Node
专为 SVI (Stable Video Infinity) 无缝衔接优化的节点

解决问题：
1. 高分辨率下SVI动态削弱
2. 拼接处跳帧
3. 界面整洁度

原作者: a1010580415-commits
重构优化: wallen0322
"""

from typing_extensions import override
from comfy_api.latest import io
import torch
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
import comfy.latent_formats
from typing import Optional
import math
import logging

logger = logging.getLogger(__name__)


class WanSVIProAdvancedI2V(io.ComfyNode):
    """SVI Pro Advanced 节点 - 专为无缝视频衔接优化"""
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent", "trim_latent", "trim_image", "next_offset")
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanSVIProAdvancedI2V",
            display_name="Wan SVI Pro Advanced I2V",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                # 基础参数
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16, 
                           display_mode=io.NumberDisplay.number,
                           tooltip="视频宽度（像素）"),
                io.Int.Input("height", default=480, min=16, max=8192, step=16, 
                           display_mode=io.NumberDisplay.number,
                           tooltip="视频高度（像素）"),
                io.Int.Input("length", default=81, min=1, max=8192, step=4, 
                           display_mode=io.NumberDisplay.number,
                           tooltip="视频总帧数"),
                io.Int.Input("batch_size", default=1, min=1, max=4096, 
                           display_mode=io.NumberDisplay.number,
                           tooltip="批次大小"),
                
                # 动态调整参数
                io.Float.Input("motion_influence", default=1.0, min=0.0, max=2.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider,
                             tooltip="动态传递权重\n1.0=正常, <1.0=减弱(高分辨率推荐), >1.0=增强(低分辨率推荐)"),
                io.Int.Input("overlap_frames", default=4, min=4, max=128, step=4, 
                           display_mode=io.NumberDisplay.number,
                           tooltip="重叠帧数（像素帧）\n必须是4的倍数，控制与上一段视频的衔接程度"),
                io.Float.Input("motion_boost", default=1.0, min=0.5, max=3.0, step=0.1, round=0.1,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="动作幅度放大\n<1.0=减小动作幅度, 1.0=正常, >1.0=放大动作幅度"),
                io.Float.Input("detail_boost", default=1.0, min=0.5, max=4.0, step=0.1, round=0.1,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="动态速度/细节增强\n0.5-0.8=平滑过渡, 1.0=平衡\n1.5-2.5=高分辨率推荐, 2.5-4.0=1080p+推荐"),
                
                # 起始帧组
                io.Image.Input("start_image", optional=True,
                             tooltip="起始帧参考图像"),
                io.Boolean.Input("enable_start_frame", default=True, optional=True,
                               tooltip="启用起始帧条件"),
                io.Float.Input("high_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="高噪声阶段起始帧强度"),
                io.Float.Input("low_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="低噪声阶段起始帧强度"),
                
                # 中间帧组
                io.Image.Input("middle_image", optional=True,
                             tooltip="中间帧参考图像"),
                io.Boolean.Input("enable_middle_frame", default=True, optional=True,
                               tooltip="启用中间帧条件"),
                io.Float.Input("middle_frame_ratio", default=0.5, min=0.0, max=1.0, step=0.01, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="中间帧位置比例 (0=开始, 1=结束)"),
                io.Float.Input("high_noise_mid_strength", default=0.8, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="高噪声阶段中间帧强度"),
                io.Float.Input("low_noise_mid_strength", default=0.2, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="低噪声阶段中间帧强度"),
                
                # 结束帧组
                io.Image.Input("end_image", optional=True,
                             tooltip="结束帧参考图像"),
                io.Boolean.Input("enable_end_frame", default=True, optional=True,
                               tooltip="启用结束帧条件"),
                io.Float.Input("low_noise_end_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="低噪声阶段结束帧强度"),
                
                # 其他参数
                io.ClipVisionOutput.Input("clip_vision_start_image", optional=True,
                                        tooltip="起始帧的 CLIP Vision 嵌入"),
                io.ClipVisionOutput.Input("clip_vision_middle_image", optional=True,
                                        tooltip="中间帧的 CLIP Vision 嵌入"),
                io.ClipVisionOutput.Input("clip_vision_end_image", optional=True,
                                        tooltip="结束帧的 CLIP Vision 嵌入"),
                io.Latent.Input("prev_latent", optional=True,
                              tooltip="上一段视频的 latent，用于无缝衔接"),
                io.Int.Input("video_frame_offset", default=0, min=0, max=1000000, step=1, 
                           display_mode=io.NumberDisplay.number, optional=True, 
                           tooltip="视频帧偏移量"),
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

    # ===========================================
    # 辅助方法
    # ===========================================
    
    @classmethod
    def _create_masks(cls, total_latents: int, H: int, W: int, device, dtype):
        """创建高低噪声掩码"""
        mask_high = torch.ones((1, 4, total_latents, H, W), device=device, dtype=dtype)
        mask_low = torch.ones((1, 4, total_latents, H, W), device=device, dtype=dtype)
        return mask_high, mask_low
    
    @classmethod
    def _resize_image(cls, img, width: int, height: int):
        """调整图像尺寸"""
        if img is None:
            return None
        return comfy.utils.common_upscale(
            img[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
    
    @classmethod
    def _calculate_aligned_position(cls, ratio: float, total_frames: int):
        """计算对齐到latent的帧位置"""
        desired_pixel_idx = int(total_frames * ratio)
        latent_idx = desired_pixel_idx // 4
        aligned_pixel_idx = latent_idx * 4
        aligned_pixel_idx = max(0, min(aligned_pixel_idx, total_frames - 1))
        return aligned_pixel_idx, latent_idx
    
    @classmethod
    def _merge_clip_vision_outputs(cls, *outputs):
        """合并多个 CLIP Vision 输出"""
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
    
    @classmethod
    def _calculate_decay_rate(cls, dynamic_strength: float) -> float:
        """根据动态强度计算衰减率"""
        if dynamic_strength <= 1.0:
            decay_rate = 0.9 - (dynamic_strength - 0.5) * 0.4
        else:
            decay_rate = 0.7 - (dynamic_strength - 1.0) * 0.2
        return max(0.05, min(0.9, decay_rate))
    
    @classmethod
    def _apply_motion_amplification(cls, motion_latent: torch.Tensor, amplification: float) -> torch.Tensor:
        """应用动作幅度放大"""
        use_frames = motion_latent.shape[2]
        
        if use_frames < 2 or amplification == 1.0:
            return motion_latent
        
        motion_vectors = []
        for i in range(1, use_frames):
            vector = motion_latent[:, :, i] - motion_latent[:, :, i-1]
            motion_vectors.append(vector)
        
        if not motion_vectors:
            return motion_latent
        
        amplified_vectors = [vec * amplification for vec in motion_vectors]
        amplified_latent = [motion_latent[:, :, 0:1].clone()]
        
        for i in range(len(amplified_vectors)):
            next_frame = amplified_latent[-1] + amplified_vectors[i].unsqueeze(2)
            amplified_latent.append(next_frame)
        
        result = torch.cat(amplified_latent, dim=2)
        logger.info(f"[SVI Pro] Motion amplification applied: {amplification:.1f}x")
        return result
    
    @classmethod
    def _build_image_cond_latent(cls, vae, latent_channels: int, total_latents: int, 
                                  H: int, W: int, device, dtype,
                                  start_image=None, middle_image=None, end_image=None,
                                  enable_start_frame=True, enable_middle_frame=True, enable_end_frame=True,
                                  middle_latent_idx: int = 0, motion_latent=None, 
                                  motion_start: int = 0, motion_end: int = 0):
        """构建 image_cond_latent"""
        if start_image is not None:
            anchor_latent = vae.encode(start_image[:1, :, :, :3])
        else:
            anchor_latent = torch.zeros([1, latent_channels, 1, H, W], device=device, dtype=dtype)
        
        image_cond_latent = torch.zeros(1, latent_channels, total_latents, H, W, 
                                       dtype=anchor_latent.dtype, device=anchor_latent.device)
        image_cond_latent = comfy.latent_formats.Wan21().process_out(image_cond_latent)
        
        if enable_start_frame and start_image is not None:
            image_cond_latent[:, :, :1] = anchor_latent
        
        if motion_latent is not None and motion_end > motion_start:
            motion_to_use = motion_latent[:, :, :motion_end-motion_start]
            image_cond_latent[:, :, motion_start:motion_end] = motion_to_use
        
        actual_middle_idx = middle_latent_idx
        if middle_image is not None and enable_middle_frame:
            middle_latent = vae.encode(middle_image[:1, :, :, :3])
            if middle_latent_idx < total_latents:
                while actual_middle_idx < motion_end and actual_middle_idx < total_latents:
                    actual_middle_idx += 1
                if actual_middle_idx < total_latents:
                    image_cond_latent[:, :, actual_middle_idx:actual_middle_idx+1] = middle_latent
        
        if end_image is not None and enable_end_frame:
            end_latent = vae.encode(end_image[:1, :, :, :3])
            image_cond_latent[:, :, total_latents-1:total_latents] = end_latent
        
        return image_cond_latent, anchor_latent.dtype, actual_middle_idx
    
    @classmethod
    def _apply_mask_strengths(cls, mask_high, mask_low, total_latents: int,
                               enable_start_frame: bool, start_image,
                               high_noise_start_strength: float, low_noise_start_strength: float,
                               enable_middle_frame: bool, middle_image,
                               middle_latent_idx: int, high_noise_mid_strength: float, low_noise_mid_strength: float,
                               enable_end_frame: bool, end_image,
                               low_noise_end_strength: float,
                               motion_start: int = 0, motion_end: int = 0, decay_rate: float = 0.7):
        """应用掩码强度设置"""
        if enable_start_frame and start_image is not None:
            mask_high[:, :, :1] = max(0.0, 1.0 - high_noise_start_strength)
            mask_low[:, :, :1] = max(0.0, 1.0 - low_noise_start_strength)
        
        if motion_end > motion_start:
            for i in range(motion_start, motion_end):
                distance = i - motion_start
                decay = decay_rate ** distance
                mask_high_val = 1.0 - (high_noise_start_strength * decay)
                mask_high[:, :, i:i+1] = max(0.05, min(0.95, mask_high_val))
                mask_low_val = 1.0 - (low_noise_start_strength * decay * 0.7)
                mask_low[:, :, i:i+1] = max(0.1, min(0.95, mask_low_val))
        
        if middle_image is not None and enable_middle_frame and middle_latent_idx < total_latents:
            mask_high[:, :, middle_latent_idx:middle_latent_idx+1] = max(0.0, 1.0 - high_noise_mid_strength)
            mask_low[:, :, middle_latent_idx:middle_latent_idx+1] = max(0.0, 1.0 - low_noise_mid_strength)
        
        if end_image is not None and enable_end_frame:
            mask_high[:, :, total_latents-1:total_latents] = 0.0
            mask_low[:, :, total_latents-1:total_latents] = max(0.0, 1.0 - low_noise_end_strength)
        
        return mask_high, mask_low
    
    @classmethod
    def _build_conditioning_outputs(cls, positive, negative, image_cond_latent, 
                                     mask_high, mask_low, clip_vision_output=None):
        """构建条件输出"""
        positive_high_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": image_cond_latent,
            "concat_mask": mask_high
        })
        
        positive_low_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": image_cond_latent,
            "concat_mask": mask_low
        })
        
        negative_out = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": image_cond_latent,
            "concat_mask": mask_high
        })
        
        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(
                positive_low_noise, {"clip_vision_output": clip_vision_output}
            )
            negative_out = node_helpers.conditioning_set_values(
                negative_out, {"clip_vision_output": clip_vision_output}
            )
        
        return positive_high_noise, positive_low_noise, negative_out

    # ===========================================
    # 主执行方法
    # ===========================================
    
    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                motion_influence=1.0, overlap_frames=4, motion_boost=1.0, detail_boost=1.0,
                start_image=None, enable_start_frame=True,
                high_noise_start_strength=1.0, low_noise_start_strength=1.0,
                middle_image=None, enable_middle_frame=True, middle_frame_ratio=0.5,
                high_noise_mid_strength=0.8, low_noise_mid_strength=0.2,
                end_image=None, enable_end_frame=True, low_noise_end_strength=1.0,
                clip_vision_start_image=None, clip_vision_middle_image=None,
                clip_vision_end_image=None, prev_latent=None, video_frame_offset=0):
        
        spatial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        total_latents = ((length - 1) // 4) + 1
        H = height // spatial_scale
        W = width // spatial_scale
        
        device = comfy.model_management.intermediate_device()
        latent = torch.zeros([batch_size, latent_channels, total_latents, H, W], device=device)
        
        trim_latent = 0
        trim_image = 0
        next_offset = 0
        
        if video_frame_offset > 0:
            if start_image is not None and start_image.shape[0] > video_frame_offset:
                start_image = start_image[video_frame_offset:]
            if middle_image is not None and middle_image.shape[0] > video_frame_offset:
                middle_image = middle_image[video_frame_offset:]
            if end_image is not None and end_image.shape[0] > video_frame_offset:
                end_image = end_image[video_frame_offset:]
            next_offset = video_frame_offset + length
        
        middle_idx = cls._calculate_aligned_position(middle_frame_ratio, length)[0]
        middle_idx = max(4, min(middle_idx, length - 5))
        middle_latent_idx = middle_idx // 4
        
        start_image = cls._resize_image(start_image, width, height)
        middle_image = cls._resize_image(middle_image, width, height)
        end_image = cls._resize_image(end_image, width, height)
        
        has_prev_latent = (prev_latent is not None and prev_latent.get("samples") is not None)
        decay_rate = cls._calculate_decay_rate(detail_boost)
        
        resolution_factor = math.sqrt(width * height) / math.sqrt(832 * 480)
        if resolution_factor > 1.2:
            logger.info(f"[SVI Pro] High resolution detected ({width}x{height}), "
                       f"consider using detail_boost > 1.5 for better motion")
        
        motion_latent = None
        motion_start = 0
        motion_end = 0
        
        if has_prev_latent:
            prev_samples = prev_latent["samples"]
            motion_latent_frames = max(1, overlap_frames // 4)
            adjusted_frames = min(motion_latent_frames, prev_samples.shape[2])
            
            if detail_boost > 1.0:
                use_frames = min(int(adjusted_frames * detail_boost), prev_samples.shape[2])
            else:
                use_frames = max(1, int(adjusted_frames * detail_boost))
            
            motion_latent = prev_samples[:, :, -use_frames:].clone()
            motion_latent = cls._apply_motion_amplification(motion_latent, motion_boost)
            
            if motion_influence != 1.0:
                motion_latent = motion_latent * motion_influence
            
            motion_start = 1 if enable_start_frame else 0
            motion_end = min(motion_start + use_frames, total_latents)
        
        image_cond_latent, anchor_dtype, actual_middle_idx = cls._build_image_cond_latent(
            vae, latent_channels, total_latents, H, W, device, latent.dtype,
            start_image, middle_image, end_image,
            enable_start_frame, enable_middle_frame, enable_end_frame,
            middle_latent_idx, motion_latent, motion_start, motion_end
        )
        
        mask_high, mask_low = cls._create_masks(total_latents, H, W, device, anchor_dtype)
        
        mask_high, mask_low = cls._apply_mask_strengths(
            mask_high, mask_low, total_latents,
            enable_start_frame, start_image,
            high_noise_start_strength, low_noise_start_strength,
            enable_middle_frame, middle_image,
            actual_middle_idx, high_noise_mid_strength, low_noise_mid_strength,
            enable_end_frame, end_image,
            low_noise_end_strength,
            motion_start, motion_end, decay_rate
        )
        
        clip_vision_output = cls._merge_clip_vision_outputs(
            clip_vision_start_image if enable_start_frame else None, 
            clip_vision_middle_image if enable_middle_frame else None, 
            clip_vision_end_image if enable_end_frame else None
        )
        
        positive_high_noise, positive_low_noise, negative_out = cls._build_conditioning_outputs(
            positive, negative, image_cond_latent, mask_high, mask_low, clip_vision_output
        )
        
        out_latent = {"samples": latent}
        
        return io.NodeOutput(positive_high_noise, positive_low_noise, negative_out, out_latent,
                trim_latent, trim_image, next_offset)


# ===========================================
# 节点注册
# ===========================================
NODE_CLASS_MAPPINGS = {
    "WanSVIProAdvancedI2V": WanSVIProAdvancedI2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSVIProAdvancedI2V": "Wan SVI Pro Advanced I2V",
}
