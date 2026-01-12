#!/bin/bash
# CARLA World Model 完整训练流程
# 包含 VQ-VAE v2 + World Model (Teacher Forcing + Scheduled Sampling)

set -e  # 遇到错误立即退出

echo "========================================="
echo "  CARLA World Model - Complete Pipeline"
echo "========================================="
echo ""

# 配置
DATA_ROOT="carla_project/data/raw"
VQVAE_CHECKPOINT="carla_project/checkpoints/vqvae_v2/best.pth"
TOKENS_FILE="carla_project/data/tokens_v2/tokens_actions.npz"
WM_CHECKPOINT="carla_project/checkpoints/world_model_v2/best.pth"
WM_SS_CHECKPOINT="carla_project/checkpoints/world_model_ss/best.pth"

# 步骤1：数据采集（如果需要）
if [ ! -d "$DATA_ROOT" ]; then
    echo "Step 1: Collecting data..."
    echo "Please start CARLA server first:"
    echo "  docker run -p 2000-2002:2000-2002 --runtime=nvidia --gpus all carlasim/carla:0.9.15"
    echo ""
    read -p "Press Enter when CARLA is ready..."

    python carla_project/collect/carla_collector.py \
        --host localhost \
        --port 2000 \
        --episodes 50 \
        --output "$DATA_ROOT"

    echo "✓ Data collection complete"
    echo ""
else
    echo "Step 1: Data already exists, skipping collection"
    echo ""
fi

# 步骤2：训练VQ-VAE v2
if [ ! -f "$VQVAE_CHECKPOINT" ]; then
    echo "Step 2: Training VQ-VAE v2..."
    python carla_project/train/train_vqvae_v2.py \
        --data-path "$DATA_ROOT" \
        --save-dir carla_project/checkpoints/vqvae_v2 \
        --epochs 100 \
        --batch-size 32

    echo "✓ VQ-VAE v2 training complete"
    echo ""
else
    echo "Step 2: VQ-VAE v2 checkpoint exists, skipping training"
    echo ""
fi

# 步骤3：导出Tokens
if [ ! -f "$TOKENS_FILE" ]; then
    echo "Step 3: Exporting tokens with VQ-VAE v2..."
    python carla_project/utils/export_tokens.py \
        --data-path "$DATA_ROOT" \
        --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
        --output "$TOKENS_FILE"

    echo "✓ Token export complete"
    echo ""
else
    echo "Step 3: Tokens already exist, skipping export"
    echo ""
fi

# 步骤4：训练World Model (Teacher Forcing)
if [ ! -f "$WM_CHECKPOINT" ]; then
    echo "Step 4: Training World Model (Teacher Forcing)..."
    python carla_project/train/train_world_model.py \
        --token-path "$TOKENS_FILE" \
        --save-dir carla_project/checkpoints/world_model_v2 \
        --epochs 150 \
        --batch-size 32

    echo "✓ World Model (TF) training complete"
    echo ""
else
    echo "Step 4: World Model (TF) checkpoint exists, skipping training"
    echo ""
fi

# 步骤5：训练World Model (Scheduled Sampling)
if [ ! -f "$WM_SS_CHECKPOINT" ]; then
    echo "Step 5: Training World Model (Scheduled Sampling)..."
    python carla_project/train/train_world_model_ss.py \
        --token-path "$TOKENS_FILE" \
        --pretrained "$WM_CHECKPOINT" \
        --save-dir carla_project/checkpoints/world_model_ss \
        --epochs 100 \
        --seq-len 12 \
        --ss-schedule linear \
        --ss-k 0.5 \
        --batch-size 8

    echo "✓ World Model (SS) training complete"
    echo ""
else
    echo "Step 5: World Model (SS) checkpoint exists, skipping training"
    echo ""
fi

# 步骤6：生成对比视频
echo "Step 6: Generating comparison videos..."

# Teacher Forcing 对比视频
python carla_project/visualize/compare_video.py \
    --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
    --world-model-checkpoint "$WM_CHECKPOINT" \
    --token-file "$TOKENS_FILE" \
    --output carla_project/outputs/comparison_tf.mp4 \
    --num-frames 200

# Scheduled Sampling 对比视频
python carla_project/visualize/compare_video.py \
    --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
    --world-model-checkpoint "$WM_SS_CHECKPOINT" \
    --token-file "$TOKENS_FILE" \
    --output carla_project/outputs/comparison_ss.mp4 \
    --num-frames 200

# Dream 视频 (TF)
python carla_project/visualize/dream.py \
    --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
    --world-model-checkpoint "$WM_CHECKPOINT" \
    --token-file "$TOKENS_FILE" \
    --output carla_project/outputs/dream_tf.mp4 \
    --num-frames 200

# Dream 视频 (SS)
python carla_project/visualize/dream.py \
    --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
    --world-model-checkpoint "$WM_SS_CHECKPOINT" \
    --token-file "$TOKENS_FILE" \
    --output carla_project/outputs/dream_ss.mp4 \
    --num-frames 200

echo "✓ Video generation complete"
echo ""

# 步骤7：评估
echo "Step 7: Evaluating models..."

# 评估 Teacher Forcing
python carla_project/evaluate/evaluate_world_model.py \
    --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
    --world-model-checkpoint "$WM_CHECKPOINT" \
    --token-file "$TOKENS_FILE" \
    --output carla_project/outputs/eval_tf.json \
    --num-samples 100 \
    --num-sequences 10 \
    --sequence-length 50

# 评估 Scheduled Sampling
python carla_project/evaluate/evaluate_world_model.py \
    --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
    --world-model-checkpoint "$WM_SS_CHECKPOINT" \
    --token-file "$TOKENS_FILE" \
    --output carla_project/outputs/eval_ss.json \
    --num-samples 100 \
    --num-sequences 10 \
    --sequence-length 50

# 可视化评估结果
python carla_project/evaluate/visualize_results.py \
    --results carla_project/outputs/eval_tf.json \
    --output-dir carla_project/figures/tf

python carla_project/evaluate/visualize_results.py \
    --results carla_project/outputs/eval_ss.json \
    --output-dir carla_project/figures/ss

echo "✓ Evaluation complete"
echo ""

echo "========================================="
echo "  Pipeline Complete!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  Videos:"
echo "    - carla_project/outputs/comparison_tf.mp4"
echo "    - carla_project/outputs/comparison_ss.mp4"
echo "    - carla_project/outputs/dream_tf.mp4"
echo "    - carla_project/outputs/dream_ss.mp4"
echo ""
echo "  Evaluation:"
echo "    - carla_project/outputs/eval_tf.json"
echo "    - carla_project/outputs/eval_ss.json"
echo "    - carla_project/figures/tf/"
echo "    - carla_project/figures/ss/"
echo ""
echo "========================================="
