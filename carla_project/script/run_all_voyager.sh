#!/bin/bash
# CARLA项目完整运行脚本（使用voyager环境）

set -e  # 遇到错误立即退出

# 激活voyager环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate voyager

echo "========================================="
echo "  CARLA World Model - 完整流程"
echo "  环境: voyager"
echo "========================================="
echo ""

# 检查环境
echo "Step 0: 检查环境..."
python -c "import torch; import carla; print('✓ 环境检查通过')" || {
    echo "环境检查失败，请先运行: ./script/setup_env.sh"
    exit 1
}

echo ""
read -p "环境检查通过，是否继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# 检查CARLA服务器
echo ""
echo "Step 0.5: 检查CARLA服务器..."
python -c "
import carla
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    client.get_world()
    print('✓ CARLA服务器已运行')
except Exception as e:
    print('✗ CARLA服务器未运行')
    print('请先启动CARLA服务器: ./script/start_carla_server.sh')
    exit(1)
" || exit 1

echo ""

# 步骤1：数据采集
echo "========================================="
echo "Step 1: 数据采集"
echo "========================================="
read -p "采集多少个episodes? (建议: 10测试, 100正式): " num_episodes
num_episodes=${num_episodes:-10}

cd collect
python carla_collector.py \
    --host localhost \
    --port 2000 \
    --episodes $num_episodes \
    --output ../data/raw

cd ..

echo ""
echo "✓ 数据采集完成"
echo ""
read -p "是否继续训练VQ-VAE? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "流程中断，可稍后继续"
    exit 0
fi

# 步骤2：训练VQ-VAE
echo ""
echo "========================================="
echo "Step 2: 训练VQ-VAE"
echo "========================================="
read -p "训练多少轮? (建议: 10测试, 100正式): " vqvae_epochs
vqvae_epochs=${vqvae_epochs:-10}

cd train
python train_vqvae.py \
    --data-path ../data/raw \
    --save-dir ../checkpoints/vqvae \
    --epochs $vqvae_epochs \
    --batch-size 32

cd ..

echo ""
echo "✓ VQ-VAE训练完成"
echo ""
read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# 步骤3：导出Tokens
echo ""
echo "========================================="
echo "Step 3: 导出Tokens"
echo "========================================="

cd utils
python export_tokens.py \
    --data-path ../data/raw \
    --vqvae-checkpoint ../checkpoints/vqvae/best.pth \
    --output ../data/tokens/tokens_actions.npz

cd ..

echo ""
echo "✓ Token导出完成"
echo ""
read -p "是否继续训练World Model? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# 步骤4：训练World Model
echo ""
echo "========================================="
echo "Step 4: 训练World Model"
echo "========================================="
read -p "训练多少轮? (建议: 20测试, 200正式): " wm_epochs
wm_epochs=${wm_epochs:-20}

cd train
python train_world_model.py \
    --token-path ../data/tokens/tokens_actions.npz \
    --save-dir ../checkpoints/world_model \
    --epochs $wm_epochs \
    --batch-size 32

cd ..

echo ""
echo "✓ World Model训练完成"
echo ""
read -p "是否生成测试视频? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# 步骤5：生成视频
echo ""
echo "========================================="
echo "Step 5: 生成测试视频"
echo "========================================="
read -p "生成多少帧? (建议: 100测试, 300正式): " num_frames
num_frames=${num_frames:-100}

cd visualize
python dream.py \
    --vqvae-checkpoint ../checkpoints/vqvae/best.pth \
    --world-model-checkpoint ../checkpoints/world_model/best.pth \
    --token-file ../data/tokens/tokens_actions.npz \
    --output ../outputs/dream_result.mp4 \
    --num-frames $num_frames

cd ..

echo ""
echo "========================================="
echo "  ✓ 完整流程完成！"
echo "========================================="
echo ""
echo "生成的视频: outputs/dream_result.mp4"
echo ""
echo "下一步建议："
echo "  1. 查看生成视频，评估效果"
echo "  2. 如果效果好，增加数据采集规模"
echo "  3. 如果效果不理想，调整训练参数"
echo ""
