from typing_extensions import override
from comfy_api.latest import io
import torch
import numpy as np
from PIL import Image, ImageOps
import json
import os

from folder_paths import get_input_directory, get_temp_directory, get_output_directory

WEB_DIRECTORY = "./js"

class WanMultiImageLoader(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanMultiImageLoader",
            display_name="Wan Multi-Image Loader",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=999,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                ),
                io.String.Input("images_data", optional=True),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def _get_base_dir(cls, dir_type: str):
        if dir_type == "temp":
            return get_temp_directory()
        if dir_type == "output":
            return get_output_directory()
        return get_input_directory()

    @classmethod
    def execute(cls, index, images_data=None):
        if not images_data:
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy,)

        try:
            data = json.loads(images_data)
        except Exception as e:
            print(f"WanMultiImageLoader: failed to parse images_data: {e}")
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy,)

        if not data or len(data) == 0:
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy,)

        actual_index = max(0, min(index, len(data) - 1))

        try:
            info = data[actual_index]
            name = info.get("name")
            dir_type = info.get("type", "input")
            subfolder = info.get("subfolder", "") or ""

            if not name:
                raise ValueError("image name missing in images_data")

            base_dir = cls._get_base_dir(dir_type)
            full_dir = os.path.join(base_dir, os.path.normpath(subfolder))
            filepath = os.path.join(full_dir, name)

            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"image file not found: {filepath}")

            img = Image.open(filepath)
            img = ImageOps.exif_transpose(img)

            if img.mode == "I":
                img = img.point(lambda i: i * (1 / 255))
            img = img.convert("RGB")

            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)[None, ...]

            return (img_tensor,)

        except Exception as e:
            print(f"WanMultiImageLoader: Error loading image: {e}")
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy,)
