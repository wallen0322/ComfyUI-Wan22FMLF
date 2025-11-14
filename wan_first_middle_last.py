from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import torch
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision


class WanFirstMiddleLastFrameToVideo(io.ComfyNode):
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanFirstMiddleLastFrameToVideo",
            display_name="Wan First-Middle-Last Frame to Video",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input(
                    "width",
                    default=832,
                    min=16,
                    max=8192,
                    step=16,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "height",
                    default=480,
                    min=16,
                    max=8192,
                    step=16,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "length",
                    default=81,
                    min=1,
                    max=8192,
                    step=4,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=4096,
                    display_mode=io.NumberDisplay.number,
                ),
                io.Combo.Input("mode", ["NORMAL", "SINGLE_PERSON"], default="NORMAL", optional=True),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("middle_image", optional=True),
                io.Image.Input("end_image", optional=True),
                io.Float.Input(
                    "middle_frame_ratio",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    round=0.01,
                    display_mode=io.NumberDisplay.slider,
                    optional=True,
                ),
                io.Float.Input(
                    "high_noise_mid_strength",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    round=0.01,
                    display_mode=io.NumberDisplay.slider,
                    optional=True,
                ),
                io.Float.Input(
                    "low_noise_start_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    round=0.01,
                    display_mode=io.NumberDisplay.slider,
                    optional=True,
                ),
                io.Float.Input(
                    "low_noise_mid_strength",
                    default=0.2,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    round=0.01,
                    display_mode=io.NumberDisplay.slider,
                    optional=True,
                ),
                io.Float.Input(
                    "low_noise_end_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    round=0.01,
                    display_mode=io.NumberDisplay.slider,
                    optional=True,
                ),
                io.ClipVisionOutput.Input("clip_vision_start_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_middle_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_end_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output("positive_high_noise"),
                io.Conditioning.Output("positive_low_noise"),
                io.Conditioning.Output("negative_out"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        mode="NORMAL",
        start_image=None,
        middle_image=None,
        end_image=None,
        middle_frame_ratio=0.5,
        high_noise_mid_strength=0.8,
        low_noise_start_strength=1.0,
        low_noise_mid_strength=0.2,
        low_noise_end_strength=1.0,
        clip_vision_start_image=None,
        clip_vision_middle_image=None,
        clip_vision_end_image=None,
    ):
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

        middle_idx = cls._calculate_aligned_position(middle_frame_ratio, length)
        middle_idx = max(4, min(middle_idx, length - 5))

        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()

        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask_high_noise[:, :, :start_image.shape[0] + 3] = 0.0
            
            low_start_mask_value = 1.0 - low_noise_start_strength
            mask_low_noise[:, :, :start_image.shape[0] + 3] = low_start_mask_value

        if middle_image is not None:
            image[middle_idx:middle_idx + 1] = middle_image

            start_range = max(0, middle_idx)
            end_range = min(length, middle_idx + 4)

            high_noise_mask_value = 1.0 - high_noise_mid_strength
            mask_high_noise[:, :, start_range:end_range] = high_noise_mask_value

            low_middle_mask_value = 1.0 - low_noise_mid_strength
            mask_low_noise[:, :, start_range:end_range] = low_middle_mask_value

        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            mask_high_noise[:, :, -end_image.shape[0]:] = 0.0
            
            low_end_mask_value = 1.0 - low_noise_end_strength
            mask_low_noise[:, :, -end_image.shape[0]:] = low_end_mask_value

        concat_latent_image = vae.encode(image[:, :, :, :3])

        if mode == "SINGLE_PERSON":
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5
            if start_image is not None:
                image_low_only[:start_image.shape[0]] = start_image
            concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
        elif low_noise_start_strength == 0.0 or low_noise_mid_strength == 0.0 or low_noise_end_strength == 0.0:
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5

            if start_image is not None and low_noise_start_strength > 0.0:
                image_low_only[:start_image.shape[0]] = start_image
            
            if middle_image is not None and low_noise_mid_strength > 0.0:
                image_low_only[middle_idx:middle_idx + 1] = middle_image
            
            if end_image is not None and low_noise_end_strength > 0.0:
                image_low_only[-end_image.shape[0]:] = end_image

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

        # 修复：negative也设置图像条件（使用high noise的条件）
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
            # 修复：negative也应用clip_vision_output
            negative_out = node_helpers.conditioning_set_values(
                negative_out,
                {"clip_vision_output": clip_vision_output}
            )

        out_latent = {"samples": latent}

        return (positive_high_noise, positive_low_noise, negative_out, out_latent)

    @classmethod
    def _calculate_aligned_position(cls, ratio, total_frames):
        desired_idx = int(total_frames * ratio)
        latent_idx = desired_idx // 4
        aligned_idx = latent_idx * 4
        aligned_idx = max(0, min(aligned_idx, total_frames - 1))
        return aligned_idx

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

