# SVI PRO 更新日志

## SVI 模式连续性优化

### 主要改进

**SVI 模式第二次采样逻辑优化**
- `motion_frames`（上一次采样的最后一帧）现在直接注入到 latent 的第一帧，确保帧间连续性
- `start_image` 作为 concat image 注入条件，提供视觉引导
- 优化了低噪声阶段的处理逻辑

### 技术变更

**第二次采样时**：
- `motion_frames` 的第一帧编码后注入 `latent` 的第一帧（不注入条件）
- `start_image` 作为 concat image 注入条件
- 优化了 `image_low` 的处理，确保低噪声阶段一致性

**代码优化**：
- 清理冗余注释
- 优化 SVI 模式逻辑流程

### 修复问题

- 修复多次采样时帧间不连续的问题
- 优化 latent 和条件注入的时机

### 文件变更

- `wan_advanced_i2v.py` - SVI 模式核心逻辑优化

