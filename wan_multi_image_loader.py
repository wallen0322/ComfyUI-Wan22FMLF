# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image, ImageOps
import os
import base64
import io


class WanMultiImageLoader:
    """
    Load multiple images with custom UI widget for batch selection and preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 999, 
                    "step": 1,
                }),
            },
            "hidden": {
                "images_data": "STRING",
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_images"
    CATEGORY = "ComfyUI-Wan22FMLF"

    def load_images(self, index: int, images_data: str):
        """
        Load images and return selected one with ORIGINAL size - no resizing!
        """
        import json
        
        try:
            data = json.loads(images_data)
        except:
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy,)
        
        if not data or len(data) == 0:
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy,)
        
        # Clamp index
        actual_index = max(0, min(index, len(data) - 1))
        
        # Load ONLY the selected image - original size
        try:
            img_data = base64.b64decode(data[actual_index]['data'].split(',')[1])
            img = Image.open(io.BytesIO(img_data))
            img = ImageOps.exif_transpose(img)
            
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))
            img = img.convert("RGB")
            
            # Convert to tensor - ORIGINAL SIZE, no padding!
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)[None,]
            
            return (img_tensor,)
            
        except Exception as e:
            print(f"Error loading image: {e}")
            dummy = torch.zeros((1, 64, 64, 3))
            return (dummy,)


NODE_CLASS_MAPPINGS = {"WanMultiImageLoader": WanMultiImageLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"WanMultiImageLoader": "Wan Multi-Image Loader"}
