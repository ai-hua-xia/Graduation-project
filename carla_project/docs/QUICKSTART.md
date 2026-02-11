# 快速开始指南

本指南按当前主线流程：**f=8 VQ-VAE + f=8 tokens + AdaLN-Zero + ActionAux + rollout World Model**。

## 0. 约定

- 在项目根目录 `carla_project/` 执行。
- Conda 环境默认 `voyager`。
- 日志使用当前目录结构：`logs/train_wm/`、`logs/train_vqvae/`、`logs/train_ss/`、`logs/data_collect/`。

## 1. 环境与服务

```bash
pip install -r requirements_carla.txt
./bin/start_carla_server.sh
```

## 2. 数据采集（并行推荐）

```bash
# 10 端口并行采集（Phase A/B）
./bin/run_collect_10.sh
```

采集输出默认在 `data/raw_action_corr_f8/`（按你当前脚本配置）。

## 3. 训练 VQ-VAE（f=8）

```bash
python train/train_vqvae_v3.py \
  --data-path data/raw_action_corr_f8 \
  --save-dir checkpoints/vqvae/vqvae_action_corr_f8 \
  --downsample-factor 8 \
  --batch-size 32
```

训练日志建议：`logs/train_vqvae/train_vqvae_f8.log`

## 4. 导出 f=8 tokens

```bash
python utils/export_tokens_v2.py \
  --data-path data/raw_action_corr_f8 \
  --vqvae-checkpoint checkpoints/vqvae/vqvae_action_corr_f8/best.pth \
  --output data/tokens_action_corr_f8/tokens_actions.npz
```

## 5. 训练 World Model（主线）

```bash
python train/train_world_model.py \
  --token-path data/tokens_action_corr_f8/tokens_actions.npz \
  --save-dir checkpoints/wm/world_model_f8_adaln_aux \
  --pretrained checkpoints/wm/world_model_f8/best.pth \
  --allow-missing-keys \
  --batch-size 2 \
  --device cuda
```

推荐 tmux 启动：

```bash
tmux new-session -d -s wm_f8_1gpu "cd /home/llb/HunyuanWorld-Voyager/bishe/carla_project && \
source ~/miniconda3/etc/profile.d/conda.sh && conda activate voyager && \
CUDA_VISIBLE_DEVICES=0 python train/train_world_model.py \
  --token-path data/tokens_action_corr_f8/tokens_actions.npz \
  --save-dir checkpoints/wm/world_model_f8_adaln_aux \
  --pretrained checkpoints/wm/world_model_f8/best.pth \
  --allow-missing-keys \
  --batch-size 2 \
  --device cuda \
  > logs/train_wm/train_world_model_f8_adaln_aux.log 2>&1"
```

## 6. 训练监控与停训判断

查看 epoch 级趋势：

```bash
rg -n "Epoch [0-9]+:|  Loss:|  CE:|  Contrast:|  ActionAux:|  Rollout:|Action Scale:|Contrast Weight:|Rollout Weight:" \
  logs/train_wm/train_world_model_f8_adaln_aux.log | tail -n 160
```

建议规则：
- 连续 3-5 个 epoch，`CE` 有持续下降：继续训练。
- 连续 8-10 个 epoch，`CE` 降幅 < 0.01：停训并调整。
- `CE` 下降但视频仍快速发糊时，优先查看 `Rollout` 是否>0 且随 epoch 下降。

## 7. 评估（FID/FVD + 崩溃指标）

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

可选关闭指标：
- `--no-fid`
- `--no-fvd`

评估结果会额外输出：
- `blur_collapse_frame`
- `texture_collapse_frame`
- `sharpness_ratio_last`
- `entropy_ratio_last`

## 8. 生成视频

```bash
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

## 9. 兼容与旧路径说明

- `checkpoints/wm/world_model_f8/`：旧版基线，可保留 `best.pth` 做迁移初始化。
- `checkpoints/wm/world_model_f8_adaln_aux/`：当前主线训练输出。

---

最后更新：2026-02-11
