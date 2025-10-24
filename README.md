# ComfyUI-Wan22FMLF

Multi-frame reference conditioning nodes for Wan2.2 A14B I2V models.

---

## 🎬 Nodes

### 1. Wan First-Middle-Last Frame 🎬

Generate videos with 3-frame reference: start, middle, and end frames.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positive` | CONDITIONING | Required | Positive prompt conditioning |
| `negative` | CONDITIONING | Required | Negative prompt conditioning |
| `vae` | VAE | Required | VAE model for encoding |
| `width` | INT | 832 | Video width (multiple of 16) |
| `height` | INT | 480 | Video height (multiple of 16) |
| `length` | INT | 81 | Total frames (multiple of 4 + 1) |
| `batch_size` | INT | 1 | Number of videos to generate |
| `start_image` | IMAGE | Optional | First frame reference |
| `middle_image` | IMAGE | Optional | Middle frame reference |
| `end_image` | IMAGE | Optional | Last frame reference |
| `middle_frame_ratio` | FLOAT | 0.5 | Middle frame position (0.0-1.0) |
| `middle_frame_strength` | FLOAT | 0.5 | Middle frame constraint (0=loose, 1=fixed) |
| `clip_vision_start_image` | CLIP_VISION_OUTPUT | Optional | CLIP Vision for start frame |
| `clip_vision_middle_image` | CLIP_VISION_OUTPUT | Optional | CLIP Vision for middle frame |
| `clip_vision_end_image` | CLIP_VISION_OUTPUT | Optional | CLIP Vision for end frame |

---

### 2. Wan Multi-Frame Reference 🎞️

Universal N-frame reference node supporting 2, 3, 4, or more reference frames with flexible positioning.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positive` | CONDITIONING | Required | Positive prompt conditioning |
| `negative` | CONDITIONING | Required | Negative prompt conditioning |
| `vae` | VAE | Required | VAE model for encoding |
| `width` | INT | 832 | Video width (multiple of 16) |
| `height` | INT | 480 | Video height (multiple of 16) |
| `length` | INT | 81 | Total frames (multiple of 4 + 1) |
| `batch_size` | INT | 1 | Number of videos to generate |
| `ref_images` | IMAGE | Required | Reference frame images |
| `ref_positions` | STRING | "" (auto) | Frame positions: "0,40,80" or "0,0.5,1.0" |
| `ref_strength` | FLOAT | 0.5 | Constraint for middle frames (0-1) |
| `fade_frames` | INT | 2 | Fade-out gradient frames (0-8) |
| `clip_vision_output` | CLIP_VISION_OUTPUT | Optional | CLIP Vision output |

**Position Format:**
- **Indices**: `"0,40,80"` - Specific frame numbers
- **Ratios**: `"0,0.5,1.0"` - Relative positions (0.0-1.0)
- **JSON**: `"[0, 40, 80]"` - Array format
- **Auto**: Leave empty for even distribution
