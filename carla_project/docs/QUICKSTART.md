# 快速开始指南

## 0. 约定

- 下面命令默认在项目根目录执行（`carla_project/`）。
- `bin/setup_env.sh` 和 `bin/activate.sh` 依赖本机 conda 环境（默认 `voyager`），如需使用请先按需修改脚本路径。

## 1. 环境准备

### 系统要求
- Ubuntu 20.04/22.04
- NVIDIA GPU（建议 16GB+ 显存）
- Python 3.8+
- CUDA 11.8+（根据驱动情况调整）

### 安装依赖
```bash
pip install -r requirements_carla.txt
```

### 安装并启动 CARLA 服务器
- 参考 `INSTALL_SERVER.md` 安装 CARLA 0.9.16
- 启动：
```bash
./bin/start_carla_server.sh
```

## 2. 快速体验（已有模型/数据）

```bash
# 查看训练状态摘要
./bin/model_tools.sh status

# 快速评估
./bin/model_tools.sh eval

# 生成对比视频
./bin/model_tools.sh video 30

# 纯预测视频
./bin/model_tools.sh video 100 1990 --pred-only

# WASD 梦境视频
./bin/model_tools.sh dream actions.txt --show-controls
```

说明：`model_tools.sh` 会自动选择 tokens 与 checkpoint，优先 `tokens_action_corr_v2` 与 `world_model_v5_ss_fast`（若存在）。

## 3. 完整流程（可选）

### Step 1: 数据采集
```bash
# 基础采集（legacy，固定 Town03 配置，路径固定为 data/raw）
python legacy/collect/collect_data.py

# 动作相关性采集（单进程）
python collect/collect_data_action_correlated.py \
    --episodes 20 \
    --data-dir data/raw_action_corr_v3

# 动作相关性采集（推荐：10端口并行，Phase A/B）
./bin/run_collect_10.sh
```

### Step 2: 训练 VQ-VAE v2
```bash
python train/train_vqvae_v3.py \
    --data-path data/raw_action_corr_v3 \
    --save-dir checkpoints/vqvae/vqvae_action_corr_v2

# 可选：f=8（32×32 tokens）
# python train/train_vqvae_v3.py \
#     --data-path data/raw_action_corr_v3 \
#     --save-dir checkpoints/vqvae/vqvae_action_corr_f8 \
#     --downsample-factor 8
```

### Step 3: 导出 tokens
```bash
python utils/export_tokens_v2.py \
    --data-path data/raw_action_corr_v3 \
    --vqvae-checkpoint checkpoints/vqvae/vqvae_action_corr_v2/best.pth \
    --output data/tokens_action_corr_v2/tokens_actions.npz

# 可选：f=8 tokens
# python utils/export_tokens_v2.py \
#     --data-path data/raw_action_corr_v3 \
#     --vqvae-checkpoint checkpoints/vqvae/vqvae_action_corr_f8/best.pth \
#     --output data/tokens_action_corr_f8/tokens_actions.npz
```

### Step 4: 训练 World Model
```bash
python train/train_world_model.py \
    --token-path data/tokens_action_corr_v2/tokens_actions.npz \
    --save-dir checkpoints/wm/world_model_v5
```

多卡示例：
```bash
torchrun --nproc_per_node=2 train/train_world_model.py \
    --token-path data/tokens_action_corr_v2/tokens_actions.npz \
    --save-dir checkpoints/wm/world_model_v5
```

### Step 5: Scheduled Sampling（可选）
```bash
python train/train_world_model_ss.py \
    --token-path data/tokens_action_corr_v2/tokens_actions.npz \
    --pretrained checkpoints/wm/world_model_v5/best.pth \
    --save-dir checkpoints/wm_ss/world_model_v5_ss_fast
```

### Step 6: 生成视频
```bash
python utils/generate_videos.py \
    --vqvae-checkpoint checkpoints/vqvae/vqvae_action_corr_v2/best.pth \
    --world-model-checkpoint checkpoints/wm_ss/world_model_v5_ss_fast/best.pth \
    --token-file data/tokens_action_corr_v2/tokens_actions.npz \
    --output-dir outputs/videos \
    --num-videos 1 \
    --num-frames 150 \
    --fps 10 \
    --temperature 1.0
```

### Step 7: 评估模型
```bash
python evaluate/evaluate_world_model.py \
    --vqvae-checkpoint checkpoints/vqvae/vqvae_action_corr_v2/best.pth \
    --world-model-checkpoint checkpoints/wm_ss/world_model_v5_ss_fast/best.pth \
    --token-file data/tokens_action_corr_v2/tokens_actions.npz \
    --output outputs/evaluations/eval.json \
    --num-samples 100 \
    --num-sequences 10 \
    --sequence-length 50 \
    --device cuda
```

## 4. WASD 动作文件

### 1) 创建动作文件
```bash
cat > actions.txt << 'EOF_ACTIONS'
W
W
A
A
N
D
D
S
EOF_ACTIONS
```

### 2) 生成梦境视频
```bash
./bin/model_tools.sh dream actions.txt --show-controls
```

## 5. 训练监控

```bash
# 查看训练日志
ls logs

# 示例：查看 v5 训练日志
tail -f logs/train_wm/train_world_model_v5.log
```

查看 checkpoint 信息：
```bash
python - << 'PY'
import torch
ckpt = torch.load('checkpoints/wm/world_model_v5/best.pth', map_location='cpu', weights_only=False)
print(f"Epoch: {ckpt.get('epoch')}, Loss: {ckpt.get('loss')}")
PY
```

## 6. 常见问题

### Q: CARLA 连接失败
A: 确保 CARLA 服务器已启动，端口 2000 可用；必要时重启 `./bin/start_carla_server.sh`

### Q: 显存不足
A: 降低 batch size 或使用更小的 seq_len；多卡训练可用 `torchrun`

### Q: 生成视频质量差
A: 使用 Scheduled Sampling 权重并尝试降低 `--temperature`

## 7. 下一步

- 查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解目录结构
- 查看 [CHANGELOG.md](CHANGELOG.md) 了解更新记录
- 查看 [INSTALL_SERVER.md](INSTALL_SERVER.md) 完成 CARLA 安装
