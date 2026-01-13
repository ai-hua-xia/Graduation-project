# 快速开始指南

## 环境准备

### 1. 系统要求
- Ubuntu 20.04/22.04
- NVIDIA GPU (16GB+ 显存)
- Python 3.8+
- CUDA 11.8+

### 2. 安装依赖
```bash
# 安装PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r carla_project/requirements_carla.txt
```

### 3. 启动CARLA服务器
```bash
bash carla_project/script/start_carla_server.sh
```

## 完整流程

### 方式1: 一键运行
```bash
bash carla_project/script/run_pipeline.sh
```

包含步骤：
1. 数据采集（如需要）
2. 训练VQ-VAE v2
3. 导出tokens
4. 训练World Model
5. 生成视频
6. 评估模型

### 方式2: 分步执行

#### Step 1: 数据采集
```bash
python carla_project/collect/collect_large.py
```

输出：`carla_project/data/raw/`

#### Step 2: 训练VQ-VAE
```bash
bash carla_project/script/start_training.sh train_vqvae vqvae 0
```

输出：`carla_project/checkpoints/vqvae_v2/best.pth`

#### Step 3: 导出Tokens
```bash
python carla_project/utils/export_tokens_v2.py \
    --checkpoint carla_project/checkpoints/vqvae_v2/best.pth \
    --data-dir carla_project/data/raw \
    --output carla_project/data/tokens_v2/tokens_actions.npz
```

#### Step 4: 训练World Model
```bash
# Teacher Forcing
bash carla_project/script/start_training.sh train_wm world_model 1

# Scheduled Sampling (可选)
bash carla_project/script/start_training.sh train_ss ss 1
```

输出：`carla_project/checkpoints/world_model_v2/best.pth`

#### Step 5: 生成视频
```bash
python carla_project/visualize/dream.py \
    --vqvae-checkpoint carla_project/checkpoints/vqvae_v2/best.pth \
    --world-model-checkpoint carla_project/checkpoints/world_model_v2/best.pth \
    --token-file carla_project/data/tokens_v2/tokens_actions.npz \
    --num-frames 300 \
    --output carla_project/outputs/dream.mp4
```

#### Step 6: 评估模型
```bash
python carla_project/evaluate/evaluate_world_model.py \
    --vqvae-checkpoint carla_project/checkpoints/vqvae_v2/best.pth \
    --world-model-checkpoint carla_project/checkpoints/world_model_v2/best.pth \
    --token-file carla_project/data/tokens_v2/tokens_actions.npz \
    --output evaluation_results.json
```

## WASD控制生成

### 1. 创建动作文件
```bash
cat > my_actions.txt << 'EOF'
W
W
W
A
A
D
D
N
EOF
```

### 2. 生成视频
```bash
python carla_project/visualize/dream.py \
    --vqvae-checkpoint carla_project/checkpoints/vqvae_v2/best.pth \
    --world-model-checkpoint carla_project/checkpoints/world_model_v2/best.pth \
    --token-file carla_project/data/tokens_v2/tokens_actions.npz \
    --action-txt my_actions.txt \
    --output my_video.mp4
```

详见 [WASD.md](WASD.md)

## 监控训练

### 查看日志
```bash
tail -f carla_project/logs/train_wm_v2.log
```

### 查看GPU
```bash
watch -n 1 nvidia-smi
```

### 查看checkpoint
```bash
python -c "
import torch
ckpt = torch.load('carla_project/checkpoints/world_model_v2/best.pth', map_location='cpu', weights_only=False)
print(f'Epoch: {ckpt[\"epoch\"]}, Loss: {ckpt[\"loss\"]:.6f}')
"
```

## 常见问题

### Q: CARLA连接失败
A: 确保CARLA服务器已启动，端口2000未被占用

### Q: 显存不足
A: 减小batch size或使用梯度累积

### Q: 训练速度慢
A: 确认使用了混合精度训练（bf16）

### Q: 生成视频质量差
A: 检查是否使用了Scheduled Sampling模型

## 下一步

- 查看 [SETUP.md](SETUP.md) 了解项目结构
- 查看 [WASD.md](WASD.md) 学习键盘控制
- 查看 [CHANGELOG.md](CHANGELOG.md) 了解开发历史
