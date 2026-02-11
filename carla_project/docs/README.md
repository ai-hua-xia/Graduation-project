# CARLA World Model

基于 CARLA 的动作条件世界模型项目，当前主线是 **f=8 token 管线 + AdaLN-Zero 条件注入 + ActionAux 辅助监督 + 短 rollout 抗漂移训练**。

## 当前推荐主线

- 原始数据：`data/raw_action_corr_f8/`
- Tokens：`data/tokens_action_corr_f8/tokens_actions.npz`
- VQ-VAE：`checkpoints/vqvae/vqvae_action_corr_f8/best.pth`
- World Model（新主线）：`checkpoints/wm/world_model_f8_adaln_aux/`
- 训练日志：`logs/train_wm/train_world_model_f8_adaln_aux.log`

> 说明：`checkpoints/wm/world_model_f8/` 保留为旧版基线（可回退对比）。

## 关键更新（当前代码）

- 条件注入从传统 FiLM 扩展为 **AdaLN-Zero**（默认启用）。
- 新增 **ActionAux** 头，辅助学习动作语义，缓解动作条件弱响应。
- 新增 **短 rollout loss**，显式惩罚 free-run 漂移（抑制“轻微转向后快速崩画面”）。
- 训练引入更稳定策略：
  - 动作对比损失延迟开启与 warmup
  - 梯度裁剪
  - 更稳的学习率与课程注入
- 评估支持感知与时序指标：**PSNR / SSIM / LPIPS / FID / FVD**，并新增崩溃指标：
  - `blur_collapse_frame`
  - `texture_collapse_frame`
  - `sharpness_ratio_last` / `entropy_ratio_last`

## 快速开始

### 1) 训练 VQ-VAE（f=8）

```bash
python train/train_vqvae_v3.py \
  --data-path data/raw_action_corr_f8 \
  --save-dir checkpoints/vqvae/vqvae_action_corr_f8 \
  --downsample-factor 8 \
  --batch-size 32
```

### 2) 导出 tokens（f=8）

```bash
python utils/export_tokens_v2.py \
  --data-path data/raw_action_corr_f8 \
  --vqvae-checkpoint checkpoints/vqvae/vqvae_action_corr_f8/best.pth \
  --output data/tokens_action_corr_f8/tokens_actions.npz
```

### 3) 训练 World Model（AdaLN-Zero + ActionAux）

```bash
python train/train_world_model.py \
  --token-path data/tokens_action_corr_f8/tokens_actions.npz \
  --save-dir checkpoints/wm/world_model_f8_adaln_aux \
  --pretrained checkpoints/wm/world_model_f8/best.pth \
  --allow-missing-keys \
  --batch-size 2 \
  --device cuda
```

### 4) 评估（含 FID/FVD）

```bash
python evaluate/evaluate_world_model.py \
  --vqvae-checkpoint checkpoints/vqvae/vqvae_action_corr_f8/best.pth \
  --world-model-checkpoint checkpoints/wm/world_model_f8_adaln_aux/best.pth \
  --token-file data/tokens_action_corr_f8/tokens_actions.npz \
  --output outputs/evaluations/wm_f8_adaln_aux_eval.json \
  --num-samples 100 \
  --num-sequences 16 \
  --sequence-length 16 \
  --fvd-clip-len 16 \
  --fvd-max-videos 32 \
  --device cuda
```

## 常用命令

```bash
# 查看 WM 训练摘要（epoch 级）
rg -n "Epoch [0-9]+:|  Loss:|  CE:|  Contrast:|  ActionAux:|  Rollout:|Rollout Weight:" logs/train_wm/train_world_model_f8_adaln_aux.log | tail -n 160

# 生成视频
python utils/generate_videos.py \
  --vqvae-checkpoint checkpoints/vqvae/vqvae_action_corr_f8/best.pth \
  --world-model-checkpoint checkpoints/wm/world_model_f8_adaln_aux/best.pth \
  --token-file data/tokens_action_corr_f8/tokens_actions.npz \
  --output-dir outputs/videos \
  --num-videos 1 \
  --num-frames 64 \
  --fps 10 \
  --prediction-only
```

## 目录概览（核心）

```text
carla_project/
├── bin/
│   ├── model_tools.sh
│   ├── run_collect_10.sh
│   └── start_carla_server.sh
├── collect/
├── train/
├── evaluate/
├── utils/
├── checkpoints/
│   ├── vqvae/vqvae_action_corr_f8/
│   └── wm/world_model_f8_adaln_aux/
├── data/
│   ├── raw_action_corr_f8/
│   └── tokens_action_corr_f8/
├── logs/
│   ├── train_wm/
│   ├── train_vqvae/
│   ├── train_ss/
│   └── data_collect/
└── docs/
```

## 文档索引

- `docs/QUICKSTART.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/CHANGELOG.md`
- `docs/INSTALL_SERVER.md`

---

最后更新：2026-02-11
