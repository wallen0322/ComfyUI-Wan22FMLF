# -*- coding: utf-8 -*-

import torch
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
from nodes import MAX_RESOLUTION
from typing import Optional, Tuple, Any


class WanFirstMiddleLastFrameToVideo:
    """
    3-frame reference node for Wan2.x I2V with dual MoE conditioning.
    Supports Wan2.1 and Wan2.2 models.
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
                "start_image": ("IMAGE",),
                "middle_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "middle_frame_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "high_noise_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                "low_noise_strength": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                "start_frame_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "middle_frame_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "end_frame_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_middle_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high_noise", "positive_low_noise", "negative", "latent")
    FUNCTION = "generate"
    CATEGORY = "ComfyUI-Wan22FMLF"

    def generate(
        self,
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
        high_noise_strength: float = 0.8,
        low_noise_strength: float = 0.2,
        start_frame_weight: float = 1.0,
        middle_frame_weight: float = 1.0,
        end_frame_weight: float = 1.0,
        clip_vision_start_image: Optional[Any] = None,
        clip_vision_middle_image: Optional[Any] = None,
        clip_vision_end_image: Optional[Any] = None
    ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...], dict]:

        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1

        device = comfy.model_management.intermediate_device()

        latent = torch.zeros(
            [batch_size, latent_channels, latent_t, height // spacial_scale, width // spacial_scale],
            device=device
        )

        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center"
            ).movedim(1, -1)

        if middle_image is not None:
            middle_image = comfy.utils.common_upscale(
                middle_image[:1].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center"
            ).movedim(1, -1)

        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-length:].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center"
            ).movedim(1, -1)

        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones(
            (1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]),
            device=device
        )

        middle_idx = self._calculate_aligned_position(middle_frame_ratio, length)
        middle_idx = max(4, min(middle_idx, length - 5))

        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()

        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            start_mask_value = max(0.0, 1.0 - start_frame_weight)
            mask_high_noise[:, :, :start_image.shape[0]] = start_mask_value
            mask_low_noise[:, :, :start_image.shape[0]] = start_mask_value

        if middle_image is not None:
            image[middle_idx:middle_idx + 1] = middle_image

            start_range = max(0, middle_idx)
            end_range = min(length, middle_idx + 4)

            high_noise_mask_value = max(0.0, 1.0 - (high_noise_strength * middle_frame_weight))
            mask_high_noise[:, :, start_range:end_range] = high_noise_mask_value

            low_noise_mask_value = max(0.0, 1.0 - (low_noise_strength * middle_frame_weight))
            mask_low_noise[:, :, start_range:end_range] = low_noise_mask_value

        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            end_mask_value = max(0.0, 1.0 - end_frame_weight)
            mask_high_noise[:, :, -end_image.shape[0]:] = end_mask_value
            mask_low_noise[:, :, -end_image.shape[0]:] = end_mask_value

        concat_latent_image = vae.encode(image[:, :, :, :3])

        if low_noise_strength == 0.0 and middle_image is not None:
            image_low_only = image.clone()
            image_low_only[middle_idx:middle_idx + 1] = 0.5
            concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
        else:
            concat_latent_image_low = concat_latent_image

        mask_high_reshaped = mask_high_noise.view(
            1,
            mask_high_noise.shape[2] // 4,
            4,
            mask_high_noise.shape[3],
            mask_high_noise.shape[4]
        ).transpose(1, 2)

        mask_low_reshaped = mask_low_noise.view(
            1,
            mask_low_noise.shape[2] // 4,
            4,
            mask_low_noise.shape[3],
            mask_low_noise.shape[4]
        ).transpose(1, 2)

        positive_high_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image,
            "concat_mask": mask_high_reshaped
        })

        positive_low_noise = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image_low,
            "concat_mask": mask_low_reshaped
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

        out_latent = {"samples": latent}

        return (positive_high_noise, positive_low_noise, negative, out_latent)

    def _calculate_aligned_position(self, ratio: float, total_frames: int) -> int:
        desired_idx = int(total_frames * ratio)
        latent_idx = desired_idx // 4
        aligned_idx = latent_idx * 4
        aligned_idx = max(0, min(aligned_idx, total_frames - 1))
        return aligned_idx

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


NODE_CLASS_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo": WanFirstMiddleLastFrameToVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanFirstMiddleLastFrameToVideo": "Wan First-Middle-Last Frame to Video"
}
