#!/bin/bash
# 测试WASD动作生成功能

echo "======================================"
echo "  测试WASD动作生成功能"
echo "======================================"
echo

# 检查必要文件
echo "1. 检查必要文件..."
if [ ! -f "carla_project/checkpoints/vqvae_v2/best.pth" ]; then
    echo "❌ VQ-VAE checkpoint不存在"
    exit 1
fi

if [ ! -f "carla_project/checkpoints/world_model_v2/best.pth" ]; then
    echo "❌ World Model checkpoint不存在"
    exit 1
fi

if [ ! -f "carla_project/data/tokens_v2/tokens_actions.npz" ]; then
    echo "❌ Token文件不存在"
    exit 1
fi

echo "✅ 所有必要文件存在"
echo

# 创建测试动作序列
echo "2. 创建测试动作序列..."
cat > /tmp/test_actions.txt << 'EOF'
# 简单测试序列
W
W
W
A
A
D
D
N
N
S
EOF

echo "✅ 测试动作序列已创建"
echo

# 测试生成（只生成10帧）
echo "3. 测试生成视频（10帧）..."
python3 carla_project/visualize/dream.py \
    --vqvae-checkpoint carla_project/checkpoints/vqvae_v2/best.pth \
    --world-model-checkpoint carla_project/checkpoints/world_model_v2/best.pth \
    --token-file carla_project/data/tokens_v2/tokens_actions.npz \
    --action-txt /tmp/test_actions.txt \
    --output carla_project/outputs/test_wasd.mp4 \
    --device cuda

if [ $? -eq 0 ]; then
    echo
    echo "✅ 测试成功！"
    echo "生成的视频: carla_project/outputs/test_wasd.mp4"
else
    echo
    echo "❌ 测试失败"
    exit 1
fi
