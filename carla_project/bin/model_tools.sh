#!/bin/bash
# 统一的模型评估和视频生成工具

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

show_help() {
    cat << EOF
========================================
  CARLA World Model Tools
========================================

Usage: $0 <command> [options]

Commands:
  status              - 查看训练进度
  eval                - 快速评估模型
  diagnose            - 诊断模型问题
  video <frames>      - 生成预测视频
  analyze             - 分析视频质量
  figures             - 生成论文图表

Examples:
  $0 status           # 查看训练进度
  $0 eval             # 评估当前模型
  $0 video 30         # 生成30帧视频
  $0 video 150        # 生成150帧视频
  $0 diagnose         # 诊断模型问题
  $0 analyze          # 分析视频质量衰减
  $0 figures          # 生成所有图表

========================================
EOF
}

cmd_status() {
    echo "=========================================="
    echo "  Training Status"
    echo "=========================================="
    echo ""

    # 检查SS训练
    if [ -f "logs/train_ss.log" ]; then
        latest_epoch=$(grep -oP "^Epoch \d+" logs/train_ss.log | tail -1)
        echo "📊 Scheduled Sampling: $latest_epoch"

        # 最近3个epoch
        grep -A 3 "^Epoch [0-9]\+:$" logs/train_ss.log | tail -12 | grep -E "(Epoch|Loss|Sampling)" | tail -9
    fi

    echo ""

    # GPU状态
    if command -v nvidia-smi &> /dev/null; then
        echo "🖥️  GPU Status:"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %s%% util, %s/%s MB\n", $1, $2, $3, $4}'
    fi

    echo ""
    echo "=========================================="
}

cmd_eval() {
    echo "=========================================="
    echo "  Quick Model Evaluation"
    echo "=========================================="
    echo ""

    # 检查文件是否存在
    VQVAE_CHECKPOINT="checkpoints/vqvae_v2/best.pth"
    WM_CHECKPOINT="checkpoints/world_model_ss/best.pth"
    TOKEN_FILE="data/tokens_v2/tokens_actions.npz"

    if [ ! -f "$VQVAE_CHECKPOINT" ]; then
        echo "❌ VQ-VAE checkpoint not found: $VQVAE_CHECKPOINT"
        exit 1
    fi

    if [ ! -f "$WM_CHECKPOINT" ]; then
        echo "❌ World Model checkpoint not found: $WM_CHECKPOINT"
        exit 1
    fi

    if [ ! -f "$TOKEN_FILE" ]; then
        echo "❌ Token file not found: $TOKEN_FILE"
        exit 1
    fi

    echo "📊 Evaluating models:"
    echo "   VQ-VAE: $VQVAE_CHECKPOINT"
    echo "   World Model: $WM_CHECKPOINT"
    echo "   Data: $TOKEN_FILE"
    echo ""

    mkdir -p outputs/evaluations

    python evaluate/evaluate_world_model.py \
        --vqvae-checkpoint "$VQVAE_CHECKPOINT" \
        --world-model-checkpoint "$WM_CHECKPOINT" \
        --token-file "$TOKEN_FILE" \
        --output outputs/evaluations/quick_eval.json \
        --num-samples 50 \
        --num-sequences 5 \
        --sequence-length 16 \
        --device cuda

    echo ""
    echo "=========================================="
    echo "✅ Evaluation complete!"
    echo "Results saved to: outputs/evaluations/quick_eval.json"
    echo "=========================================="
}

cmd_diagnose() {
    echo "=========================================="
    echo "  Model Diagnostic"
    echo "=========================================="
    echo ""

    python utils/diagnose_model.py
}

cmd_video() {
    local frames=${1:-30}
    local output_name="demo_${frames}frames"

    echo "=========================================="
    echo "  Generating Video"
    echo "=========================================="
    echo ""
    echo "Frames: $frames"
    echo "Duration: ~$((frames / 10))s"
    echo ""

    mkdir -p outputs/videos

    python utils/generate_videos.py \
        --vqvae-checkpoint checkpoints/vqvae_v2/best.pth \
        --world-model-checkpoint checkpoints/world_model_ss/best.pth \
        --token-file data/tokens_v2/tokens_actions.npz \
        --output-dir outputs/videos \
        --num-videos 1 \
        --num-frames "$frames" \
        --fps 10 \
        --temperature 1.0

    if [ -f "outputs/videos/prediction_01.mp4" ]; then
        mv outputs/videos/prediction_01.mp4 "outputs/videos/${output_name}.mp4"
        echo ""
        echo "✅ Video saved to: outputs/videos/${output_name}.mp4"
    fi
}

cmd_analyze() {
    echo "=========================================="
    echo "  Video Quality Analysis"
    echo "=========================================="
    echo ""

    mkdir -p outputs/analysis

    python tools/analyze_video_quality.py

    echo ""
    echo "Analysis plot: outputs/analysis/video_quality_analysis.png"
}

cmd_figures() {
    echo "=========================================="
    echo "  Generating Figures"
    echo "=========================================="
    echo ""

    mkdir -p outputs/figures

    python utils/generate_figures.py

    echo ""
    echo "Figures saved to: outputs/figures/"
}

# Main
cd "$PROJECT_DIR" || exit 1

case "${1:-}" in
    status)
        cmd_status
        ;;
    eval)
        cmd_eval
        ;;
    diagnose)
        cmd_diagnose
        ;;
    video)
        cmd_video "${2:-30}"
        ;;
    analyze)
        cmd_analyze
        ;;
    figures)
        cmd_figures
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
