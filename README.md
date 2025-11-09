# ComfyUI-Wan22FMLF

Multi-frame reference conditioning nodes for Wan2.2 A14B I2V models.

# 注意事项

尽量使用官方模型，不要使用量化模型，新增单人模式可有效杜绝颜色累积和亮度闪烁，
低分辨率推荐：480x832/832x480/576x1024
高分辨率对剑：704x1280/1280x704
注意：720*1280会导致中间帧闪烁问题，

# 差异比较大的场景：
如果是变化场景较大，（如变身等）可以切换为normal模式按以下参数设置，使用normal模式，并且低噪lightx2v的Lora权重需要降低到0.6左右，不然低噪会破坏掉你的变化效果。

<img width="347" height="461" alt="ead76d24f88d3e773bdbfb34994addf8" src="https://github.com/user-attachments/assets/a2da0900-7439-4e57-a105-b6c772d5f6af" />

# 如果你想做无限续杯多图参考长视频，推荐以下参数：

<img width="539" height="833" alt="a95eec0897a6ac258917b989a0620bab" src="https://github.com/user-attachments/assets/86a2aaed-efd5-4e11-9bca-0518f9239c8f" />




# WanMultiFrameRefToVideo中ref_positions 参数使用说明

## 概述
`ref_positions` 用于指定参考帧在视频中的位置，支持多种格式。

---

## 格式说明

### 1. 留空（自动分布）
```
ref_positions: ""
```
- **效果**: 参考帧在视频中均匀分布
- **示例**: 
  - 3张图片，length=81 → 位置：0, 40, 80
  - 6张图片，length=81 → 位置：0, 16, 32, 48, 64, 80

---

### 2. 比例值（推荐）
```
ref_positions: "0, 0.2, 0.5, 0.8, 1.0"
```
- **范围**: 0.0 - 1.0（0%到100%）
- **效果**: 按视频长度的比例定位
- **计算**: `实际位置 = 比例 × (length - 1)`
- **示例**: 
  - length=81时
  - 0.0 → 帧0
  - 0.5 → 帧40
  - 1.0 → 帧80

---

### 3. 绝对帧索引
```
ref_positions: "0, 20, 40, 60, 80"
```
- **范围**: 大于等于2的整数
- **效果**: 直接指定帧位置
- **注意**: 超出范围会自动裁剪到 [0, length-1]

---

### 4. JSON 数组格式
```
ref_positions: "[0, 0.25, 0.5, 0.75, 1.0]"
```
- **格式**: 标准JSON数组
- **支持**: 比例值或绝对值混用
- **示例**: `[0, 20, 0.5, 60, 1.0]`

---

## 实际应用示例

### 示例1：3帧视频（首-中-尾）
```
length: 81
ref_images: 3张图片
ref_positions: ""           → 自动分布到 0, 40, 80
ref_positions: "0, 0.5, 1"  → 精确定位到 0, 40, 80
```

### 示例2：5帧视频
```
length: 81
ref_images: 5张图片
ref_positions: "0, 0.25, 0.5, 0.75, 1"  → 位置: 0, 20, 40, 60, 80
```

### 示例3：6帧视频（自定义）
```
length: 81
ref_images: 6张图片
ref_positions: "0, 10, 25, 45, 65, 80"  → 绝对位置
ref_positions: "0, 0.12, 0.31, 0.56, 0.81, 1"  → 比例位置
```

### 示例4：混合格式（不推荐但支持）
```
ref_positions: "[0, 20, 0.5, 60, 1.0]"  → 0, 20, 40, 60, 80
```

---

## 重要提示

### 自动对齐
- 所有位置会自动对齐到4的倍数（latent对齐）
- 例如：帧15 → 对齐到帧12

### 帧间距保护
- 相邻帧自动保持至少4帧间距
- 例如：如果帧16和帧18冲突 → 自动调整为16和20

### 数量匹配
- 如果位置数量少于图片数量：重复最后一个位置
- 如果位置数量多于图片数量：截断多余位置

---

## 推荐用法

**最简单**: 留空，让系统自动分布
```
ref_positions: ""
```

**最灵活**: 使用比例值（0-1）
```
ref_positions: "0, 0.33, 0.67, 1"
```

**最精确**: 使用绝对帧索引
```
ref_positions: "0, 20, 40, 60, 80"
```

---

## 常见问题

**Q: 我有6张图片，怎么均匀分布？**
A: 留空即可，或使用 `"0, 0.2, 0.4, 0.6, 0.8, 1"`

**Q: 比例值0.5和1哪个区别？**
A: 
- 0.5 = 50%位置 = 帧40（length=81时）
- 1 = 100%位置 = 帧80（length=81时）
- 1.0 也是100%

**Q: 可以让某些帧更密集吗？**
A: 可以，使用自定义位置：`"0, 10, 15, 20, 50, 80"`

**Q: 位置会自动排序吗？**
A: 不会，请按顺序输入位置值

---
#DONT WORRY BE HAPPY
建议高噪2步就够了，高噪步数太多会增加中间帧闪的概率

20251031重大更新
解决中间帧闪烁问题，请使用我更新的示例工作流，节点做了较大改变！
建议高噪中间帧强度：06-0.8，低噪中间帧强度：0.2左右，如果复杂场景，低噪中间帧可直接设置为0


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
