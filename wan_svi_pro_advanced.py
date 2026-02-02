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


class WanSVIProAdvancedI2V(io.ComfyNode):
    
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
                io.Int.Input("width", default=832, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number,
                           tooltip="Width of the generated video in pixels"),
                io.Int.Input("height", default=480, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number,
                           tooltip="Height of the generated video in pixels"),
                io.Int.Input("length", default=81, min=1, max=8192, step=4, display_mode=io.NumberDisplay.number,
                           tooltip="Total number of frames in the generated video"),
                io.Int.Input("batch_size", default=1, min=1, max=4096, display_mode=io.NumberDisplay.number,
                           tooltip="Batch size (number of videos to generate)"),
                
                # 动态调整参数
                io.Float.Input("motion_boost", default=1.0, min=0.5, max=3.0, step=0.1, round=0.1,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Motion amplitude amplification\n<1.0: Reduce movement amplitude\n1.0: Normal (default)\n>1.0: Amplify movement amplitude\nFor bigger actions like punches, slaps, large gestures"),
                io.Float.Input("detail_boost", default=1.0, min=0.5, max=4.0, step=0.1, round=0.1,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Motion dynamic strength\n0.5-0.8: Smooth transitions (continuity first)\n1.0: Balanced (default)\n1.2-1.5: Dynamic motion priority\n1.6-2.5: Strong motion for HD/720p+\n2.6-4.0: Very strong motion for 1080p+/high-res"),
                io.Float.Input("motion_influence", default=1.0, min=0.0, max=2.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider,
                             tooltip="Influence strength of motion latent from previous video\n1.0 = normal, <1.0 = weaker, >1.0 = stronger"),
                
                # 重叠帧参数
                io.Int.Input("overlap_frames", default=4, min=4, max=128, step=4, display_mode=io.NumberDisplay.number,
                           tooltip="Number of overlapping frames (pixel frames)\nMust be multiple of 4 (4 pixel frames = 1 latent frame)\nControls how much to continue from previous video"),
                
                # 起始帧组
                io.Image.Input("start_image", optional=True,
                             tooltip="First frame reference image (anchor for the video)"),
                io.Boolean.Input("enable_start_frame", default=True, optional=True,
                               tooltip="Enable start frame conditioning"),
                io.Float.Input("high_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for start frame in high-noise stage\n0.0 = no conditioning, 1.0 = full conditioning"),
                io.Float.Input("low_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for start frame in low-noise stage"),
                
                # 中间帧组
                io.Image.Input("middle_image", optional=True,
                             tooltip="Middle frame reference image for better consistency"),
                io.Boolean.Input("enable_middle_frame", default=True, optional=True,
                               tooltip="Enable middle frame conditioning"),
                io.Float.Input("middle_frame_ratio", default=0.5, min=0.0, max=1.0, step=0.01, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Position of middle frame (0=start, 1=end)"),
                io.Float.Input("high_noise_mid_strength", default=0.8, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for middle frame in high-noise stage"),
                io.Float.Input("low_noise_mid_strength", default=0.2, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for middle frame in low-noise stage"),
                
                # 结束帧组
                io.Image.Input("end_image", optional=True,
                             tooltip="Last frame reference image (target ending)"),
                io.Boolean.Input("enable_end_frame", default=True, optional=True,
                               tooltip="Enable end frame conditioning"),
                io.Float.Input("low_noise_end_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for end frame in low-noise stage"),
                
                # 其他参数
                io.ClipVisionOutput.Input("clip_vision_start_image", optional=True,
                                        tooltip="CLIP vision embedding for start frame (for better semantic consistency)"),
                io.ClipVisionOutput.Input("clip_vision_middle_image", optional=True,
                                        tooltip="CLIP vision embedding for middle frame"),
                io.ClipVisionOutput.Input("clip_vision_end_image", optional=True,
                                        tooltip="CLIP vision embedding for end frame"),
                io.Latent.Input("prev_latent", optional=True,
                              tooltip="Previous video latent for seamless continuation"),
                io.Int.Input("video_frame_offset", default=0, min=0, max=1000000, step=1, display_mode=io.NumberDisplay.number, 
                           optional=True, tooltip="Video frame offset (advanced, usually set to 0)\nSkip this many frames from input images"),
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
                motion_boost=1.0, detail_boost=1.0, motion_influence=1.0,
                overlap_frames=4,
                start_image=None, enable_start_frame=True,
                high_noise_start_strength=1.0, low_noise_start_strength=1.0,
                middle_image=None, enable_middle_frame=True, middle_frame_ratio=0.5,
                high_noise_mid_strength=0.8, low_noise_mid_strength=0.2,
                end_image=None, enable_end_frame=True, low_noise_end_strength=1.0,
                clip_vision_start_image=None, clip_vision_middle_image=None,
                clip_vision_end_image=None, prev_latent=None, video_frame_offset=0):
        
        # 重命名变量以保持代码一致性
        motion_amplification = motion_boost
        dynamic_strength = detail_boost
        
        # 计算基本参数
        spatial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        total_latents = ((length - 1) // 4) + 1  # 1个潜变量帧 = 4个图像帧
        H = height // spatial_scale
        W = width // spatial_scale
        
        device = comfy.model_management.intermediate_device()
        
        # 创建空latent
        latent = torch.zeros([batch_size, latent_channels, total_latents, H, W], 
                            device=device)
        
        trim_latent = 0
        trim_image = 0
        next_offset = 0
        
        # 应用视频帧偏移（如果启用）
        if video_frame_offset > 0:
            if start_image is not None and start_image.shape[0] > 1:
                start_image = start_image[video_frame_offset:] if start_image.shape[0] > video_frame_offset else None
            
            if middle_image is not None and middle_image.shape[0] > 1:
                middle_image = middle_image[video_frame_offset:] if middle_image.shape[0] > video_frame_offset else None
            
            if end_image is not None and end_image.shape[0] > 1:
                end_image = end_image[video_frame_offset:] if end_image.shape[0] > video_frame_offset else None
            
            next_offset = video_frame_offset + length
        
        # 计算中间位置
        middle_idx = cls._calculate_aligned_position(middle_frame_ratio, length)[0]
        middle_idx = max(4, min(middle_idx, length - 5))
        middle_latent_idx = middle_idx // 4
        
        # 调整图像尺寸
        def resize_image(img):
            if img is None:
                return None
            return comfy.utils.common_upscale(
                img[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        
        start_image = resize_image(start_image) if start_image is not None else None
        middle_image = resize_image(middle_image) if middle_image is not None else None
        end_image = resize_image(end_image) if end_image is not None else None
        
        # ===========================================
        # SVI无缝衔接模式核心逻辑
        # ===========================================
        
        # 检查是否有prev_latent用于继续
        has_prev_latent = (prev_latent is not None and prev_latent.get("samples") is not None)
        
        if has_prev_latent:
            # SVI Continue: 使用prev_latent作为延续参考
            prev_samples = prev_latent["samples"]
            
            # 将像素帧转换为潜变量帧
            motion_latent_frames = max(1, overlap_frames // 4)  # 至少1个潜变量帧
            
            # 根据动态强度调整使用的帧数
            # dynamic_strength越高，使用越多帧保持动态
            adjusted_frames = min(motion_latent_frames, prev_samples.shape[2])
            
            # 动态强度影响使用的帧数
            if dynamic_strength > 1.0:
                # 动态强时使用更多帧
                use_frames = min(int(adjusted_frames * dynamic_strength), prev_samples.shape[2])
            else:
                # 动态弱时使用较少帧
                use_frames = max(1, int(adjusted_frames * dynamic_strength))
            
            # 提取重叠潜变量帧
            motion_latent = prev_samples[:, :, -use_frames:].clone()
            
            # ===========================================
            # 核心改进：动作幅度放大
            # ===========================================
            
            # 如果有多于1帧且需要放大动作幅度
            if use_frames >= 2 and motion_amplification != 1.0:
                # 计算运动向量（帧间差异）
                motion_vectors = []
                for i in range(1, use_frames):
                    # 当前帧与前帧的差异
                    vector = motion_latent[:, :, i] - motion_latent[:, :, i-1]
                    motion_vectors.append(vector)
                
                # 应用动作幅度放大
                if motion_vectors:
                    # 放大运动向量
                    amplified_vectors = [vec * motion_amplification for vec in motion_vectors]
                    
                    # 重建放大的运动潜变量
                    # 保持第一帧不变，后续帧根据放大后的运动向量重建
                    amplified_latent = [motion_latent[:, :, 0:1].clone()]
                    
                    for i in range(len(amplified_vectors)):
                        # 基于前一帧和放大后的运动向量计算下一帧
                        next_frame = amplified_latent[-1] + amplified_vectors[i].unsqueeze(2)
                        amplified_latent.append(next_frame)
                    
                    # 组合所有帧
                    motion_latent = torch.cat(amplified_latent, dim=2)
                    
                    print(f"[SVI Pro] Motion amplification applied: {motion_amplification:.1f}x")
            
            # 应用运动强度（motion_influence）
            if motion_influence != 1.0:
                motion_latent = motion_latent * motion_influence
            
            # 根据分辨率自动调整动态强度
            # 高分辨率需要更强的动态
            resolution_factor = math.sqrt(width * height) / math.sqrt(832 * 480)  # 相对于832x480的因子
            if resolution_factor > 1.2:  # 如果分辨率显著高于默认
                print(f"[SVI Pro] High resolution detected ({width}x{height}), consider using dynamic_strength > 1.5 for better motion")
            
            # ===========================================
            # 构建统一的image_cond_latent
            # ===========================================
            
            # 编码起始图像作为锚点潜变量
            if start_image is not None:
                anchor_latent = vae.encode(start_image[:1, :, :, :3])
            else:
                # 如果没有起始图像，创建空锚点
                anchor_latent = torch.zeros([1, latent_channels, 1, H, W], 
                                           device=device, dtype=latent.dtype)
            
            # 创建基础image_cond_latent
            image_cond_latent = torch.zeros(1, latent_channels, total_latents, H, W, 
                                           dtype=anchor_latent.dtype, device=anchor_latent.device)
            image_cond_latent = comfy.latent_formats.Wan21().process_out(image_cond_latent)
            
            # 在位置0插入锚点（如果起始帧启用）
            if enable_start_frame:
                image_cond_latent[:, :, :1] = anchor_latent
            
            # ===========================================
            # 将运动潜变量放入合适位置
            # ===========================================
            
            # 运动潜变量放在锚点之后
            motion_start = 1 if enable_start_frame else 0
            motion_end = min(motion_start + use_frames, total_latents)
            
            if motion_end > motion_start:
                # 确保尺寸匹配
                motion_to_use = motion_latent[:, :, :motion_end-motion_start]
                image_cond_latent[:, :, motion_start:motion_end] = motion_to_use
            
            # 插入中间图像（如果提供）
            if middle_image is not None and enable_middle_frame:
                middle_latent = vae.encode(middle_image[:1, :, :, :3])
                if middle_latent_idx < total_latents:
                    # 确保中间位置不与其他条件重叠
                    actual_middle_idx = middle_latent_idx
                    while (actual_middle_idx < motion_end and actual_middle_idx < total_latents):
                        actual_middle_idx += 1
                    
                    if actual_middle_idx < total_latents:
                        image_cond_latent[:, :, actual_middle_idx:actual_middle_idx+1] = middle_latent
                        middle_latent_idx = actual_middle_idx
            
            # 插入结束图像（如果提供）
            if end_image is not None and enable_end_frame:
                end_latent = vae.encode(end_image[:1, :, :, :3])
                image_cond_latent[:, :, total_latents-1:total_latents] = end_latent
            
            # ===========================================
            # 根据dynamic_strength计算衰减率
            # ===========================================
            
            # 动态衰减率映射：
            # dynamic_strength=0.5: 衰减率0.9（快速衰减，平滑衔接）
            # dynamic_strength=1.0: 衰减率0.7（平衡，与原版相同）
            # dynamic_strength=2.0: 衰减率0.3（强动态）
            # dynamic_strength=4.0: 衰减率0.1（极强动态，用于高分辨率）
            
            if dynamic_strength <= 1.0:
                # 0.5-1.0范围，衰减率从0.9到0.7（平滑到平衡）
                decay_rate = 0.9 - (dynamic_strength - 0.5) * 0.4  # 0.9->0.7
            else:
                # 1.0-4.0范围，衰减率从0.7到0.1（平衡到极强动态）
                decay_rate = 0.7 - (dynamic_strength - 1.0) * 0.2  # 0.7->0.1
            
            # 确保衰减率在合理范围内
            decay_rate = max(0.05, min(0.9, decay_rate))
            
            # ===========================================
            # 创建无缝衔接掩码
            # ===========================================
            
            # 创建掩码
            mask_high = torch.ones((1, 4, total_latents, H, W), 
                                  device=device, dtype=anchor_latent.dtype)
            mask_low = torch.ones((1, 4, total_latents, H, W), 
                                 device=device, dtype=anchor_latent.dtype)
            
            # 无缝衔接模式：锚点和运动潜变量使用强条件
            if enable_start_frame and start_image is not None:
                # 锚点使用高条件强度
                mask_high[:, :, :1] = max(0.0, 1.0 - high_noise_start_strength)
                mask_low[:, :, :1] = max(0.0, 1.0 - low_noise_start_strength)
            
            # 运动潜变量：根据dynamic_strength使用渐进衰减
            if motion_end > motion_start:
                for i in range(motion_start, motion_end):
                    # 计算衰减（距离锚点越远，条件越弱）
                    distance = i - motion_start
                    decay = decay_rate ** distance
                    
                    # 高噪声掩码：渐进衰减
                    mask_high_val = 1.0 - (high_noise_start_strength * decay)
                    mask_high[:, :, i:i+1] = max(0.05, min(0.95, mask_high_val))  # 扩大范围到0.05-0.95
                    
                    # 低噪声掩码：保持较强条件但也要衰减
                    mask_low_val = 1.0 - (low_noise_start_strength * decay * 0.7)
                    mask_low[:, :, i:i+1] = max(0.1, min(0.95, mask_low_val))
            
            # 中间帧：使用指定强度
            if middle_image is not None and enable_middle_frame and middle_latent_idx < total_latents:
                mask_high[:, :, middle_latent_idx:middle_latent_idx+1] = max(
                    0.0, 1.0 - high_noise_mid_strength
                )
                mask_low[:, :, middle_latent_idx:middle_latent_idx+1] = max(
                    0.0, 1.0 - low_noise_mid_strength
                )
            
            # 结束帧：使用指定强度
            if end_image is not None and enable_end_frame:
                # 注意：结束帧在高噪声阶段完全使用（掩码为0.0）
                mask_high[:, :, total_latents-1:total_latents] = 0.0  # 完全使用结束帧
                mask_low[:, :, total_latents-1:total_latents] = max(
                    0.0, 1.0 - low_noise_end_strength
                )
            
            # 构建条件
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
            
            # 处理clip vision
            clip_vision_output = cls._merge_clip_vision_outputs(
                clip_vision_start_image if enable_start_frame else None, 
                clip_vision_middle_image if enable_middle_frame else None, 
                clip_vision_end_image if enable_end_frame else None
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
            # 如果没有prev_latent，仅使用起始图像
            # 编码start_image作为锚点潜变量
            anchor_latent = vae.encode(start_image[:1, :, :, :3])
            
            # 构建image_cond_latent
            image_cond_latent = torch.zeros(1, latent_channels, total_latents, H, W, 
                                           dtype=anchor_latent.dtype, device=anchor_latent.device)
            image_cond_latent = comfy.latent_formats.Wan21().process_out(image_cond_latent)
            
            # 在位置0插入锚点（如果起始帧启用）
            if enable_start_frame:
                image_cond_latent[:, :, :1] = anchor_latent
            
            # 插入中间图像（如果提供）
            if middle_image is not None and enable_middle_frame:
                middle_latent = vae.encode(middle_image[:1, :, :, :3])
                if middle_latent_idx < total_latents:
                    image_cond_latent[:, :, middle_latent_idx:middle_latent_idx+1] = middle_latent
            
            # 插入结束图像（如果提供）
            if end_image is not None and enable_end_frame:
                end_latent = vae.encode(end_image[:1, :, :, :3])
                image_cond_latent[:, :, total_latents-1:total_latents] = end_latent
            
            # 创建掩码
            mask_high = torch.ones((1, 4, total_latents, H, W), 
                                  device=device, dtype=anchor_latent.dtype)
            mask_low = torch.ones((1, 4, total_latents, H, W), 
                                 device=device, dtype=anchor_latent.dtype)
            
            # 应用起始帧强度
            if enable_start_frame:
                mask_high[:, :, :1] = max(0.0, 1.0 - high_noise_start_strength)
                mask_low[:, :, :1] = max(0.0, 1.0 - low_noise_start_strength)
            
            # 应用中间帧强度
            if middle_image is not None and enable_middle_frame:
                mask_high[:, :, middle_latent_idx:middle_latent_idx+1] = max(
                    0.0, 1.0 - high_noise_mid_strength
                )
                mask_low[:, :, middle_latent_idx:middle_latent_idx+1] = max(
                    0.0, 1.0 - low_noise_mid_strength
                )
            
            # 应用结束帧强度
            if end_image is not None and enable_end_frame:
                # 注意：结束帧在高噪声阶段完全使用（掩码为0.0）
                mask_high[:, :, total_latents-1:total_latents] = 0.0
                mask_low[:, :, total_latents-1:total_latents] = max(
                    0.0, 1.0 - low_noise_end_strength
                )
            
            # 构建条件
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
            
            # 处理clip vision
            clip_vision_output = cls._merge_clip_vision_outputs(
                clip_vision_start_image if enable_start_frame else None, 
                clip_vision_middle_image if enable_middle_frame else None, 
                clip_vision_end_image if enable_end_frame else None
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
        else:
            # 如果没有起始图像和prev_latent，创建基本的空条件
            # 这应该很少发生，但提供一个安全的回退
            
            # 创建基础image_cond_latent
            image_cond_latent = torch.zeros(1, latent_channels, total_latents, H, W, 
                                           device=device, dtype=latent.dtype)
            image_cond_latent = comfy.latent_formats.Wan21().process_out(image_cond_latent)
            
            # 创建掩码
            mask = torch.ones((1, 4, total_latents, H, W), 
                            device=device, dtype=latent.dtype)
            
            # 构建条件
            positive_high_noise = node_helpers.conditioning_set_values(positive, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask
            })
            
            negative_out = node_helpers.conditioning_set_values(negative, {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask
            })
            
            out_latent = {"samples": latent}
            
            return io.NodeOutput(positive_high_noise, positive_high_noise, negative_out, out_latent,
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


# ===========================================
# 节点注册
# ===========================================
NODE_CLASS_MAPPINGS = {
    "WanSVIProAdvancedI2V": WanSVIProAdvancedI2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSVIProAdvancedI2V": "Wan SVI Pro Advanced I2V",
}
