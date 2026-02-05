# ComfyUI-Wan22FMLF_svi-fix

## 📝 更新日志 (svi-fix)

### 2025-02-01 SVI 新增节点

新增svipro专用节点wan_svi_pro_advanced.py

用于解决如下问题：高分辨率下svi动态削弱，拼接处跳帧，以及界面整洁度

#### 测试结论：在示例工作流的参数下，1920x1080的分辨率也可以保持住动态。
<img width="13031" height="4457" alt="workflow (82)" src="https://github.com/user-attachments/assets/73ce87c5-9d49-4c04-a021-5cda89e7010f" />

（附示例工作流截图)

#### 新参数：

motion influence ：用于控制传递下来的动态权重，默认1  （为了保持动态：低分辨率建议调高，高分辨率建议调低）

overlap frames : 提供给motion inluence的图像帧数量（=latent帧x4）

motion boost ：提高动态，增加动作幅度。 `原理：计算连续帧之间的差异，放大运动向量`

detail boost ：提高速度和细节，但是可能会让画面不稳定。 `原理：在创建掩码和条件时，调整衰减率和使用的帧数`

搭配1：高传递高动态，motion influence 2 overlap frames 16 motion boost 1 detail boost 0.5 ， 视频17帧重叠可无缝


搭配2：低传递高动态，motion influence 0.7 overlap frames 4 motion boost 2.5 detail boost 3.0  视频5帧重叠可无缝

具体使用方式，请见svipro boost工作流示例
---


> Multi-frame reference conditioning nodes for Wan2.2 A14B I2V models.

一个为 Wan2.2 A14B I2V 模型提供多帧参考条件控制的 ComfyUI 自定义节点集合。

---

## 📋 目录

- [更新日志](#更新日志)
- [快速开始](#快速开始)
- [节点说明](#节点说明)
- [参数配置](#参数配置)
- [使用建议](#使用建议)
- [常见问题](#常见问题)

---

## 📝 更新日志

### 2025-01-27 SVI 模式增强

- ✅ **修复 SVI 模式 mask 维度问题**
  - 修复 `concat_mask` 维度从 `(1, 1, T, H, W)` 到 `(1, 4, T, H, W)`
  - 与 non-SVI 模式格式保持一致，解决拼接方向错误

- ✅ **新增 `svi_motion_strength` 参数**
  - 控制 SVI 模式下的动态传递强度
  - 参数范围：0.0-2.0，默认 1.0
  - `<1.0` = 更稳定的效果，`>1.0` = 更夸张的动态效果

- ✅ **新增三个参考帧开关**
  - `enable_start_frame`：控制是否启用起始帧参考
  - `enable_middle_frame`：控制是否启用中间帧参考（已有）
  - `enable_end_frame`：控制是否启用结束帧参考

**感谢**：[@a1010580415-commits](https://github.com/a1010580415-commits) 在 [PR #29](https://github.com/wallen0322/ComfyUI-Wan22FMLF/pull/29) 中的贡献和建议

---

### SVI PRO - 连续性优化

**SVI 项目地址**：https://github.com/vita-epfl/Stable-Video-Infinity

**SVI 模式第二次采样逻辑优化**
- ✅ `motion_frames`（上一次采样的最后一帧）现在直接注入到 latent 的第一帧，确保帧间连续性
- ✅ `start_image` 作为 concat image 注入条件，提供视觉引导
- ✅ 优化了低噪声阶段的处理逻辑

**技术变更**：
- 第二次采样时：`motion_frames` 的第一帧编码后注入 `latent` 的第一帧（不注入条件）
- `start_image` 作为 concat image 注入条件
- 优化了 `image_low` 的处理，确保低噪声阶段一致性

**修复问题**：
- 修复多次采样时帧间不连续的问题
- 优化 latent 和条件注入的时机

---

### 最新更新 - 高性能图片选择节点

- ✅ **重大性能优化**：改用服务器文件存储，不再在前端存储 base64 数据
  - 避免 LocalStorage 配额限制（QuotaExceededError）
  - 大幅减少工作流文件大小
  - 提升节点加载和响应速度
- ✅ 使用 ComfyUI 标准 `/upload/image` 和 `/view` 接口
- ✅ 修复图片排序功能，支持手动排序
- ✅ 优化代码结构，提升稳定性

---

### 运动增强功能

- ✅ **新增 `structural_repulsion_boost` 参数**
  - 通过空间梯度条件注入增强运动效果
  - **仅影响高噪阶段**，保护低噪阶段的颜色稳定性
  - 参数范围：1.0-2.0，默认 1.0（无增强）
  - 推荐值：1.2-1.5 可获得明显效果
  
**工作原理：**
- 在相邻参考帧之间创建空间梯度
- 运动区域的 mask 值降低，增强 `concat_latent_image` 的影响
- 自动保护参考帧附近的区域，避免颜色和亮度偏移
- 适用于所有多帧参考节点

**使用建议：**
- 默认值 1.0 = 不应用增强
- 1.2-1.3 = 轻微增强，适合大多数场景
- 1.4-1.5 = 中等增强，适合需要明显动态的场景
- 1.6-2.0 = 强烈增强，注意可能影响颜色稳定性

---

### 2025-10-31 重大更新

- ✅ **解决中间帧闪烁问题**
- ✅ 节点做了较大改变，请使用更新的示例工作流
- ✅ 建议参数：
  - 高噪中间帧强度：0.6-0.8
  - 低噪中间帧强度：0.2 左右
  - 复杂场景，低噪中间帧可直接设置为 0

---

## 🚀 快速开始

### 系统要求

- 尽量使用**官方模型**，不要使用量化模型
- 新增**单人模式**可有效杜绝颜色累积和亮度闪烁

### 推荐分辨率

#### 低分辨率推荐
- `480×832`
- `832×480`
- `576×1024`

#### 高分辨率推荐
- `704×1280`
- `1280×704`

⚠️ **注意**：`720×1280` 会导致中间帧闪烁问题，不推荐使用。

---

## 🎬 节点说明

### 1. Wan First-Middle-Last Frame 🎬

生成带有 3 帧参考的视频：起始帧、中间帧和结束帧。

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `positive` | CONDITIONING | **必需** | 正向提示词条件 |
| `negative` | CONDITIONING | **必需** | 负向提示词条件 |
| `vae` | VAE | **必需** | 用于编码的 VAE 模型 |
| `width` | INT | 832 | 视频宽度（16 的倍数） |
| `height` | INT | 480 | 视频高度（16 的倍数） |
| `length` | INT | 81 | 总帧数（4 的倍数 + 1） |
| `batch_size` | INT | 1 | 生成视频的数量 |
| `start_image` | IMAGE | 可选 | 起始帧参考图 |
| `middle_image` | IMAGE | 可选 | 中间帧参考图 |
| `end_image` | IMAGE | 可选 | 结束帧参考图 |
| `middle_frame_ratio` | FLOAT | 0.5 | 中间帧位置比例（0.0-1.0） |
| `high_noise_mid_strength` | FLOAT | 0.8 | 高噪中间帧约束强度（0=宽松，1=严格） |
| `low_noise_start_strength` | FLOAT | 1.0 | 低噪起始帧约束强度 |
| `low_noise_mid_strength` | FLOAT | 0.2 | 低噪中间帧约束强度 |
| `low_noise_end_strength` | FLOAT | 1.0 | 低噪结束帧约束强度 |
| `structural_repulsion_boost` | FLOAT | 1.0 | 运动增强系数（1.0-2.0），仅影响高噪阶段 |
| `clip_vision_start_image` | CLIP_VISION_OUTPUT | 可选 | 起始帧的 CLIP Vision 输出 |
| `clip_vision_middle_image` | CLIP_VISION_OUTPUT | 可选 | 中间帧的 CLIP Vision 输出 |
| `clip_vision_end_image` | CLIP_VISION_OUTPUT | 可选 | 结束帧的 CLIP Vision 输出 |

---

### 2. Wan Multi-Frame Reference 🎞️

支持 2、3、4 或更多参考帧的通用多帧参考节点，具有灵活的位置配置。

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `positive` | CONDITIONING | **必需** | 正向提示词条件 |
| `negative` | CONDITIONING | **必需** | 负向提示词条件 |
| `vae` | VAE | **必需** | 用于编码的 VAE 模型 |
| `width` | INT | 832 | 视频宽度（16 的倍数） |
| `height` | INT | 480 | 视频高度（16 的倍数） |
| `length` | INT | 81 | 总帧数（4 的倍数 + 1） |
| `batch_size` | INT | 1 | 生成视频的数量 |
| `ref_images` | IMAGE | **必需** | 参考帧图片 |
| `ref_positions` | STRING | `""` (自动) | 帧位置：`"0,40,80"` 或 `"0,0.5,1.0"` |
| `ref_strength` | FLOAT | 0.5 | 中间帧约束强度（0-1） |
| `fade_frames` | INT | 2 | 淡出渐变帧数（0-8） |
| `clip_vision_output` | CLIP_VISION_OUTPUT | 可选 | CLIP Vision 输出 |

---

## ⚙️ 参数配置

### `ref_positions` 参数使用说明

`ref_positions` 用于指定参考帧在视频中的位置，支持多种格式。

#### 格式说明

##### 1️⃣ 留空（自动分布）⭐ 推荐

```
ref_positions: ""
```

- **效果**：参考帧在视频中均匀分布
- **示例**：
  - 3 张图片，length=81 → 位置：0, 40, 80
  - 6 张图片，length=81 → 位置：0, 16, 32, 48, 64, 80

---

##### 2️⃣ 比例值（推荐）

```
ref_positions: "0, 0.2, 0.5, 0.8, 1.0"
```

- **范围**：0.0 - 1.0（0% 到 100%）
- **效果**：按视频长度的比例定位
- **计算**：`实际位置 = 比例 × (length - 1)`
- **示例**（length=81 时）：
  - `0.0` → 帧 0
  - `0.5` → 帧 40
  - `1.0` → 帧 80

---

##### 3️⃣ 绝对帧索引

```
ref_positions: "0, 20, 40, 60, 80"
```

- **范围**：大于等于 2 的整数
- **效果**：直接指定帧位置
- **注意**：超出范围会自动裁剪到 `[0, length-1]`

---

##### 4️⃣ JSON 数组格式

```
ref_positions: "[0, 0.25, 0.5, 0.75, 1.0]"
```

- **格式**：标准 JSON 数组
- **支持**：比例值或绝对值混用
- **示例**：`[0, 20, 0.5, 60, 1.0]`

---

#### 实际应用示例

##### 示例 1：3 帧视频（首-中-尾）

```yaml
length: 81
ref_images: 3张图片
ref_positions: ""           # → 自动分布到 0, 40, 80
ref_positions: "0, 0.5, 1"  # → 精确定位到 0, 40, 80
```

##### 示例 2：5 帧视频

```yaml
length: 81
ref_images: 5张图片
ref_positions: "0, 0.25, 0.5, 0.75, 1"  # → 位置: 0, 20, 40, 60, 80
```

##### 示例 3：6 帧视频（自定义）

```yaml
length: 81
ref_images: 6张图片
ref_positions: "0, 10, 25, 45, 65, 80"          # → 绝对位置
ref_positions: "0, 0.12, 0.31, 0.56, 0.81, 1"   # → 比例位置
```

---

#### 重要提示

##### 自动对齐

- 所有位置会自动对齐到 4 的倍数（latent 对齐）
- **示例**：帧 15 → 对齐到帧 12

##### 帧间距保护

- 相邻帧自动保持至少 4 帧间距
- **示例**：如果帧 16 和帧 18 冲突 → 自动调整为 16 和 20

##### 数量匹配

- 如果位置数量少于图片数量：重复最后一个位置
- 如果位置数量多于图片数量：截断多余位置

---

#### 推荐用法

**最简单** ⭐ 留空，让系统自动分布

```
ref_positions: ""
```

**最灵活** ⭐ 使用比例值（0-1）

```
ref_positions: "0, 0.33, 0.67, 1"
```

**最精确** ⭐ 使用绝对帧索引

```
ref_positions: "0, 20, 40, 60, 80"
```

---

## 💡 使用建议

### 场景配置

#### 差异较大的场景（如变身等）

如果场景变化较大，可以切换到 **normal 模式**，使用以下参数设置：

- 使用 normal 模式
- LightX2V 的 Lora 权重需要降低到 **0.6 左右**
- 否则低噪会破坏你的变化效果

![差异场景配置示例](https://github.com/user-attachments/assets/a2da0900-7439-4e57-a105-b6c772d5f6af)

---

#### 无限续杯多图参考长视频

推荐以下参数配置：

![长视频配置示例](https://github.com/user-attachments/assets/86a2aaed-efd5-4e11-9bca-0518f9239c8f)

**✨ 更新**：3 图循环工作流，增加可视化选择图片的节点，不用再去建文件夹和改图片名字了，更新立刻享受！

![可视化选择节点](https://github.com/user-attachments/assets/1e3665f4-a664-408e-a6b4-06ff9bfa0c8b)

---

### 噪点强度建议

#### 高噪设置

- **步数**：2 步就够了
- ⚠️ 高噪步数太多会增加中间帧闪烁的概率

#### 中间帧强度建议

| 场景类型 | 高噪中间帧强度 | 低噪中间帧强度 |
|----------|---------------|---------------|
| 普通场景 | 0.6-0.8 | 0.2 左右 |
| 复杂场景 | 0.6-0.8 | 0（可直接设置为 0） |

---

## ❓ 常见问题

### 关于 `ref_positions` 参数

**Q: 我有 6 张图片，怎么均匀分布？**

A: 留空即可，或使用 `"0, 0.2, 0.4, 0.6, 0.8, 1"`

---

**Q: 比例值 0.5 和 1 有什么区别？**

A: 
- `0.5` = 50% 位置 = 帧 40（length=81 时）
- `1` 或 `1.0` = 100% 位置 = 帧 80（length=81 时）

---

**Q: 可以让某些帧更密集吗？**

A: 可以，使用自定义位置：`"0, 10, 15, 20, 50, 80"`

---

**Q: 位置会自动排序吗？**

A: 不会，请按顺序输入位置值

---

## 📚 示例工作流

项目提供了多个示例工作流供参考：

- `Long video + segmented prompt words.json` - 长视频 + 分段提示词
- `Wan22FMLF-1109update.json` - 最新更新版本
- `长视频-SVI-shot+三图（Long Video Service with Unlimited 3-Image Continuation）.json` - 长视频服务（无限 3 图续接）

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

查看项目许可证文件以获取更多信息。

---

**🎉 Don't worry, be happy!**
