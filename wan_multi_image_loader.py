from typing_extensions import override
from comfy_api.latest import io
import torch
import numpy as np
from PIL import Image, ImageOps
import base64
import io as python_io
import os


WEB_DIRECTORY = "./js"


class WanMultiImageLoader(io.ComfyNode):
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanMultiImageLoader",
            display_name="Wan Multi-Image Loader",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.Int.Input("index", default=0, min=0, max=999, step=1, display_mode=io.NumberDisplay.number),
                io.String.Input("images_data", optional=True),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )
    
    @classmethod
    def execute(cls, index, images_data=None):
        import json
        
        if not images_data:
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy,)
        
        try:
            data = json.loads(images_data)
        except:
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy,)
        
        if not data or len(data) == 0:
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy,)
        
        actual_index = max(0, min(index, len(data) - 1))
        
        try:
            img_data = base64.b64decode(data[actual_index]['data'].split(',')[1])
            img = Image.open(python_io.BytesIO(img_data))
            img = ImageOps.exif_transpose(img)
            
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))
            img = img.convert("RGB")
            
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)[None,]
            
            return (img_tensor,)
            
        except Exception as e:
            print(f"Error loading image: {e}")
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy,)
