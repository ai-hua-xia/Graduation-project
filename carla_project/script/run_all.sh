#!/bin/bash
# CARLA项目快速开始脚本

set -e  # 遇到错误立即退出

echo "========================================="
echo "  CARLA World Model - Quick Start"
echo "========================================="
echo ""

# 检查CARLA连接
echo "Step 0: Checking CARLA connection..."
python -c "import carla; client = carla.Client('localhost', 2000); client.set_timeout(5.0); print('CARLA version:', client.get_server_version())" || {
    echo "Error: Cannot connect to CARLA server!"
    echo "Please start CARLA server first:"
    echo "  docker run -p 2000-2002:2000-2002 --runtime=nvidia --gpus all carlasim/carla:0.9.15"
    exit 1
}

echo "✓ CARLA connected"
echo ""

# 步骤1：数据采集
echo "Step 1: Collecting data..."
cd collect
python carla_collector.py \
    --host localhost \
    --port 2000 \
    --episodes 10 \
    --output ../data/raw
cd ..

echo "✓ Data collection complete"
echo ""

# 步骤2：训练VQ-VAE
echo "Step 2: Training VQ-VAE..."
cd train
python train_vqvae.py \
    --data-path ../data/raw \
    --save-dir ../checkpoints/vqvae \
    --epochs 50 \
    --batch-size 32
cd ..

echo "✓ VQ-VAE training complete"
echo ""

# 步骤3：导出Tokens
echo "Step 3: Exporting tokens..."
cd utils
python export_tokens.py \
    --data-path ../data/raw \
    --vqvae-checkpoint ../checkpoints/vqvae/best.pth \
    --output ../data/tokens/tokens_actions.npz
cd ..

echo "✓ Token export complete"
echo ""

# 步骤4：训练World Model
echo "Step 4: Training World Model..."
cd train
python train_world_model.py \
    --token-path ../data/tokens/tokens_actions.npz \
    --save-dir ../checkpoints/world_model \
    --epochs 100 \
    --batch-size 32
cd ..

echo "✓ World Model training complete"
echo ""

# 步骤5：生成视频
echo "Step 5: Generating video..."
cd visualize
python dream.py \
    --vqvae-checkpoint ../checkpoints/vqvae/best.pth \
    --world-model-checkpoint ../checkpoints/world_model/best.pth \
    --token-file ../data/tokens/tokens_actions.npz \
    --output ../outputs/dream_result.mp4 \
    --num-frames 200
cd ..

echo "✓ Video generation complete"
echo ""

echo "========================================="
echo "  All steps complete!"
echo "  Video saved to: outputs/dream_result.mp4"
echo "========================================="
