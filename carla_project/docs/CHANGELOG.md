# 变更日志

## 2026-02-11

### ✅ World Model 稳定训练与抗崩溃更新
- `train_world_model.py` 引入 **short rollout loss**（基于模型自回归预测回灌）以抑制 free-run 漂移
- `utils/dataset.py` 支持返回 `future_tokens` / `future_actions`（`rollout_steps`）
- `train/config.py` 增加 rollout 课程参数并调整主线默认超参数（更稳收敛）
- 训练日志新增 `Rollout` / `Rollout Weight` 字段，便于定位“loss 降但视频崩”的问题

### ✅ 评估增加崩溃可观测性
- `evaluate_world_model.py` 新增清晰度/纹理熵统计与 collapse 指标：
  - `blur_collapse_frame`
  - `texture_collapse_frame`
  - `sharpness_ratio_last`
  - `entropy_ratio_last`

### ✅ 文档同步
- `README` / `QUICKSTART` / `PROJECT_STRUCTURE` 同步 rollout 主线与新评估输出

## 2026-02-09

### ✅ 训练收敛主线重构（f=8）
- World Model 条件注入升级为 **AdaLN-Zero**（支持与 FiLM 切换，默认 AdaLN-Zero）
- 新增 **ActionAux** 辅助头与损失（显式约束动作语义学习）
- 训练策略改为“先稳定 CE，再逐步动作约束”：
  - contrast 延迟开启 + warmup
  - 动作注入/对比权重可调度
  - 梯度裁剪（`max_grad_norm`）
- 恢复训练时强制使用当前配置学习率，避免继续沿用旧 optimizer lr

### ✅ 评估能力增强
- `evaluate_world_model.py` 集成/完善 **FID 与 FVD** 评估开关与参数
- 支持输出包含像素级与感知/时序级的综合结果（PSNR/SSIM/LPIPS/FID/FVD）

### ✅ 文档与路径对齐
- 文档主线更新为：`raw_action_corr_f8 -> tokens_action_corr_f8 -> vqvae_action_corr_f8 -> world_model_f8_adaln_aux`
- 日志目录按子目录归档（`logs/train_wm`、`logs/train_vqvae`、`logs/train_ss`、`logs/data_collect`）
- 输出目录同步为 `outputs/evaluations`、`outputs/videos`、`outputs/analysis`、`outputs/debug`

## 2026-01-30

### ✅ VQ-VAE f=8（32×32 tokens）支持
- VQ-VAE 支持可配置 downsample_factor（默认 16，新增 8）
- 训练脚本支持 `--downsample-factor` 并在 checkpoint 中记录该参数
- tokens 导出将 downsample_factor 写入 npz（与 f=16 并行共存）
- 文档补充 f=8 训练与 tokens 导出示例

## 2026-01-29

### ✅ 并行采集与数据集升级
- 新增 10 端口并行采集脚本 `bin/run_collect_10.sh`（Phase A/B 分布）
- 新动作相关性数据集：`data/raw_action_corr_f8`
- 采集脚本增加连接重试/超时配置（client-timeout / connect-retries）
- 采集质量约束保持：collision / lane / stuck 过滤与预览视频输出

### ✅ 模型与训练进度同步
- VQ-VAE v3 训练输出到 `checkpoints/vqvae/vqvae_action_corr_v2`
- Tokens 固定为 `data/tokens_action_corr_v2/tokens_actions.npz`
- World Model v5 + Scheduled Sampling v5/v5_fast 产出
- `train/config.py` 更新为大模型配置并启用 f=8 路线

### ✅ 工具与文档更新
- `bin/model_tools.sh` 统一状态/评估/视频/诊断入口
- README/QUICKSTART/PROJECT_STRUCTURE 对齐当时版本路径
- 旧版脚本归档到 `legacy/`（保留实验记录）

## 2026-01-16

### ✅ 文档同步
- README/QUICKSTART/PROJECT_STRUCTURE 对齐脚本与目录结构
- 新增 CARLA 0.9.16 服务器安装文档 `docs/INSTALL_SERVER.md`

## 2026-01-14

### ✅ 动作-视觉相关性采集
- 新增动作相关性采集脚本（`collect_data_action_correlated.py`）
- 采集时实时计算动作-视觉相关性，不达标自动重采
- 导出 tokens 增加 `episode_ids`，训练序列避免跨 episode

### ✅ 采集脚本清理
- 删除 `collect_data_action_branching.py`
- 删除 `collect_data_action_focused.py`

## 2026-01-13

### ✅ 脚本与文档整理
- 精简入口脚本与文档，统一日常操作入口
- 输出目录统一到 `outputs/`
- WASD 动作控制流程文档化

## 2026-01-12

### ✅ 评估系统
- 建立单步/自回归评估流程
- 引入长期稳定性相关统计指标

## 2026-01-11

### ✅ VQ-VAE 与基础训练
- 完成 v2 路线训练与基础数据采集
- 建立最小可运行端到端流程
