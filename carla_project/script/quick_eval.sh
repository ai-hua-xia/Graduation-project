#!/bin/bash
# 快速评估已有模型

set -e

echo "========================================="
echo "  Quick Evaluation"
echo "========================================="
echo ""

VQVAE_CHECKPOINT="carla_project/checkpoints/vqvae_v2/best.pth"
TOKENS_FILE="carla_project/data/tokens_v2/tokens_actions.npz"

# 检查参数
if [ $# -eq 0 ]; then
    echo "Usage: $0 <world_model_checkpoint> [output_name]"
    echo ""
    echo "Example:"
    echo "  $0 carla_project/checkpoints/world_model_v2/best.pth tf"
    echo "  $0 carla_project/checkpoints/world_model_ss/best.pth ss"
    exit 1
fi

WM_CHECKPOINT=$1
OUTPUT_NAME=${2:-"result"}

echo "Evaluating: $WM_CHECKPOINT"
echo "Output name: $OUTPUT_NAME"
echo ""

# 生成对比视频
echo "Generating comparison video..."
python carla_project/visualize/compare_video.py \
    --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
    --world-model-checkpoint "$WM_CHECKPOINT" \
    --token-file "$TOKENS_FILE" \
    --output "carla_project/outputs/comparison_${OUTPUT_NAME}.mp4" \
    --num-frames 200

# 生成dream视频
echo "Generating dream video..."
python carla_project/visualize/dream.py \
    --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
    --world-model-checkpoint "$WM_CHECKPOINT" \
    --token-file "$TOKENS_FILE" \
    --output "carla_project/outputs/dream_${OUTPUT_NAME}.mp4" \
    --num-frames 200

# 评估指标
echo "Computing metrics..."
python carla_project/evaluate/evaluate_world_model.py \
    --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
    --world-model-checkpoint "$WM_CHECKPOINT" \
    --token-file "$TOKENS_FILE" \
    --output "carla_project/outputs/eval_${OUTPUT_NAME}.json" \
    --num-samples 100 \
    --num-sequences 10 \
    --sequence-length 50

# 可视化
echo "Visualizing results..."
python carla_project/evaluate/visualize_results.py \
    --results "carla_project/outputs/eval_${OUTPUT_NAME}.json" \
    --output-dir "carla_project/figures/${OUTPUT_NAME}"

echo ""
echo "========================================="
echo "  Evaluation Complete!"
echo "========================================="
echo ""
echo "Generated files:"
echo "  - carla_project/outputs/comparison_${OUTPUT_NAME}.mp4"
echo "  - carla_project/outputs/dream_${OUTPUT_NAME}.mp4"
echo "  - carla_project/outputs/eval_${OUTPUT_NAME}.json"
echo "  - carla_project/figures/${OUTPUT_NAME}/"
echo ""
