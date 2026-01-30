# 变更日志

## 2026-01-30

### ✅ VQ-VAE f=8（32×32 tokens）支持
- VQ-VAE 支持可配置 downsample_factor（默认 16，新增 8）
- 训练脚本支持 `--downsample-factor` 并在 checkpoint 中记录该参数
- tokens 导出将 downsample_factor 写入 npz（与 f=16 并行共存）
- 文档补充 f=8 训练与 tokens 导出示例

## 2026-01-29

### ✅ 并行采集与数据集升级
- 新增 10 端口并行采集脚本 `bin/run_collect_10.sh`（Phase A/B 分布）
- 新动作相关性数据集：`data/raw_action_corr_v3`
- 采集脚本增加连接重试/超时配置（client-timeout / connect-retries）
- 采集质量约束保持：collision / lane / stuck 过滤与预览视频输出

### ✅ 模型与训练进度同步
- VQ-VAE v3 训练输出到 `checkpoints/vqvae_action_corr_v2`
- Tokens 固定为 `data/tokens_action_corr_v2/tokens_actions.npz`
- World Model v5 + Scheduled Sampling v5/v5_fast 产出
- `train/config.py` 已更新到 A-XL 配置（32层/18heads）

### ✅ 工具与文档更新
- `bin/model_tools.sh` 默认优先选择 v5/v5_ss_fast 与 vqvae_action_corr_v2
- README/QUICKSTART/PROJECT_STRUCTURE 对齐最新数据与脚本
- 旧版脚本归档到 `legacy/`（保留实验记录）

## 2026-01-16

### ✅ 文档同步
- README/QUICKSTART/PROJECT_STRUCTURE 对齐当前脚本与目录结构
- 补充 model_tools 自动选择 tokens/模型的规则说明
- 新增 CARLA 0.9.16 服务器安装文档 `docs/INSTALL_SERVER.md`

## 2026-01-14

### ✅ 动作-视觉相关性采集
- 新增动作相关性数据采集脚本（collect_data_action_correlated.py）
- 采集时实时计算动作-视觉相关性，不达标自动重采
- 使用低速稳定段+高速转向脉冲，提高相关性稳定度
- 质量门控加入 burst 段速度检查，减少误拒

### ✅ 采集验证与训练数据
- verify_data_action_focused.py 支持自定义数据目录与分支分析
- 导出 tokens 增加 episode_ids，训练序列避免跨 episode
- 数据集加载器按 episode_ids 过滤跨场景序列

### ✅ 清理采集脚本
- 删除 collect_data_action_branching.py
- 删除 collect_data_action_focused.py

## 2026-01-13

### ✅ 脚本整理
- 精简 bin/ 脚本从12个减少到6个
- 删除冗余脚本（quick_test, quick_eval, check_training, run_pipeline, start_training, quick_start）
- 统一入口：model_tools.sh 提供 status/eval/diagnose/video/analyze/figures
- 三层架构：bin/（Shell脚本）→ tools/（分析工具）→ utils/（核心库）
- 统一输出目录：outputs/（evaluations, videos, analysis, figures）

### ✅ 项目整理
- 精简文档从11个减少到6个
- 删除冗余文档（CLEANUP_GUIDE, COMPARISON, STATUS, WASD_SUMMARY）
- 创建简洁的主文档结构
- 更新 README.md 和 PROJECT_STRUCTURE.md

### ✅ WASD动作控制
- 实现WASD按键到动作的映射
- 支持从文本文件读取动作序列
- 基于训练数据分布优化映射值
- 添加WASD使用文档和测试脚本

### 🔄 Scheduled Sampling训练
- 开始SS训练（从TF模型预训练）
- 配置：linear schedule, max_prob=0.5
- 当前状态：Epoch 0，Loss从0.0712降至0.0500

## 2026-01-12

### ✅ 评估系统
- 实现完整的评估系统
- 单步预测评估（Token准确率、PSNR/SSIM/LPIPS）
- 自回归生成评估（长期稳定性指标）
- 稳定性指标：崩溃点、半衰期、衰减率

### ✅ World Model v2训练
- 完成150 epochs训练
- 模型：238M参数，16层Transformer
- 最终loss: ~0.05
- 保存最佳模型

## 2026-01-11

### ✅ VQ-VAE v2优化
- 更深的编码器/解码器
- 更大的codebook（1024×256）
- 训练loss: 0.0089
- 重建质量显著提升

### ✅ 数据采集
- 采集10,000帧数据（Town03）
- 图像尺寸：256×256
- 动作：[steering, throttle]
- 70%大转向，30%直行

### ✅ 环境配置
- 配置CARLA 0.9.15服务器
- 配置PyTorch 2.5.1 + CUDA 12.4
- 验证同步模式数据采集

## 技术要点

### 模型架构
- **VQ-VAE v2**: 1024 codebook, 256 embed_dim
- **World Model**: 238M参数, 16层Transformer
- **上下文**: 4帧历史
- **动作**: 2维 (steering, throttle)

### 训练策略
- **混合精度**: bf16
- **Teacher Forcing**: 初始训练
- **Scheduled Sampling**: 缓解误差累积
- **标签平滑**: warmup策略

### 评估指标
- **图像质量**: PSNR, SSIM, LPIPS
- **Token准确率**: 单步预测
- **稳定性**: 崩溃点, 半衰期, 衰减率

## 已知问题

### 训练日志过大
- **原因**: tqdm进度条输出
- **解决**: 从checkpoint读取信息

### 误差累积
- **原因**: Teacher Forcing训练
- **解决**: Scheduled Sampling训练

## 下一步

- [ ] 继续完善 Scheduled Sampling 版本（统一目录命名）
- [ ] 对比 v4 与 v4_ss 的评估结果
- [ ] 清理旧 checkpoint 与冗余日志

## 参考

- 查看 [README.md](README.md) 了解项目概况
- 查看 [QUICKSTART.md](QUICKSTART.md) 开始使用
- 查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解结构
- 查看 [INSTALL_SERVER.md](INSTALL_SERVER.md) 配置 CARLA 服务器
