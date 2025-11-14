# -*- coding: utf-8 -*-

import torch
import json
from typing import List, Tuple, Optional, Any
import node_helpers
import comfy
import comfy.utils
from nodes import MAX_RESOLUTION


class WanMultiFrameRefToVideo:
    """
    Universal N-frame reference node with dual MoE conditioning and motion enhancement.
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
                "ref_images": ("IMAGE",),
            },
            "optional": {
                "mode": (["NORMAL", "SINGLE_PERSON"], {
                    "default": "NORMAL",
                    "tooltip": "NORMAL: full control | SINGLE_PERSON: low noise only uses first frame"
                }),
                "ref_positions": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Frame indices or ratios. Leave empty for auto distribution."
                }),
                "ref_strength_high": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "High-noise stage strength for middle frames."
                }),
                "ref_strength_low": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Low-noise stage strength for middle frames."
                }),
                "end_frame_strength_high": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "High-noise stage strength for end frame."
                }),
                "end_frame_strength_low": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Low-noise stage strength for end frame."
                }),
                "motion_amplitude": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "High-noise motion enhancement. Amplifies inter-frame differences (1.0=off, 1.15=recommended)"
                }),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high_noise", "positive_low_noise", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "ComfyUI-Wan22FMLF"

    def generate(self, positive: Tuple[Any, ...],
                 negative: Tuple[Any, ...],
                 vae: Any,
                 width: int, height: int, length: int, batch_size: int,
                 ref_images: torch.Tensor,
                 mode: str = "NORMAL",
                 ref_positions: str = "",
                 ref_strength_high: float = 0.8,
                 ref_strength_low: float = 0.2,
                 end_frame_strength_high: float = 1.0,
                 end_frame_strength_low: float = 1.0,
                 motion_amplitude: float = 1.0,
                 clip_vision_output: Optional[Any] = None) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], dict]:

        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1
        device = comfy.model_management.intermediate_device()

        latent = torch.zeros([batch_size, latent_channels, latent_t,
                             height // spacial_scale, width // spacial_scale], device=device)

        imgs = self._resize_images(ref_images, width, height, device)
        n_imgs = imgs.shape[0]
        positions = self._parse_positions(ref_positions, n_imgs, length)

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

        if motion_amplitude > 1.0 and concat_latent_image_high.shape[2] > 1:
            n_frames = concat_latent_image_high.shape[2]

            frame_diffs = []
            for i in range(1, n_frames):
                prev_frame = concat_latent_image_high[:, :, i-1:i]
                curr_frame = concat_latent_image_high[:, :, i:i+1]

                diff = curr_frame - prev_frame
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff_centered = diff - diff_mean

                enhanced_diff = diff_centered * motion_amplitude + diff_mean
                frame_diffs.append(enhanced_diff)

            enhanced_latents = [concat_latent_image_high[:, :, 0:1]]
            for enhanced_diff in frame_diffs:
                next_frame = enhanced_latents[-1] + enhanced_diff
                next_frame = torch.clamp(next_frame, -6, 6)
                enhanced_latents.append(next_frame)

            concat_latent_image_high = torch.cat(enhanced_latents, dim=2)

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

        # 修复：negative也设置图像条件（使用high noise的条件）
        negative_out = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent_image_high,
            "concat_mask": mask_high_reshaped
        })

        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise,
                                                                   {"clip_vision_output": clip_vision_output})
            # 修复：negative也应用clip_vision_output
            negative_out = node_helpers.conditioning_set_values(negative_out,
                                                             {"clip_vision_output": clip_vision_output})

        return (positive_high_noise, positive_low_noise, negative_out, {"samples": latent})

    def _resize_images(self, images: torch.Tensor, width: int, height: int, device: torch.device) -> torch.Tensor:
        images = images.to(device)
        x = images.movedim(-1, 1)
        x = comfy.utils.common_upscale(x, width, height, "bilinear", "center")
        x = x.movedim(1, -1)

        if x.shape[-1] == 4:
            x = x[..., :3]

        return x

    def _parse_positions(self, pos_str: str, n_imgs: int, length: int) -> List[int]:
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


NODE_CLASS_MAPPINGS = {"WanMultiFrameRefToVideo": WanMultiFrameRefToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"WanMultiFrameRefToVideo": "Wan Multi-Frame Reference (Dual MoE + Motion)"}
