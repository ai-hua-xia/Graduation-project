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
  video <frames> [start_idx] [--pred-only] - 生成预测视频
  dream <action_file> [--show-controls] - 使用WASD动作文件生成视频
  analyze             - 分析视频质量
  figures             - 生成论文图表

Examples:
  $0 status           # 查看训练进度
  $0 eval             # 评估当前模型
  $0 video 30         # 生成30帧对比视频（随机场景）
  $0 video 100 1990   # 生成100帧对比视频（最连续场景）
  $0 video 100 1990 --pred-only  # 生成100帧纯预测视频
  $0 dream actions.txt  # 使用WASD动作文件生成视频（推荐）
  $0 dream actions.txt --show-controls  # 显示按键指示器
  $0 diagnose         # 诊断模型问题
  $0 analyze          # 分析视频质量衰减
  $0 figures          # 生成所有图表

Note:
  --pred-only: 只显示预测帧，不显示Ground Truth对比
  dream命令: 使用WASD动作文件，完全自回归生成，场景最连续
  推荐起始位置: 1990 (数据集中最连续的100帧片段)

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
    local start_idx=${2:-""}
    local pred_only_flag=""
    local output_name="demo_${frames}frames"

    # 检查第三个参数是否是 --pred-only
    if [ "$3" = "--pred-only" ]; then
        pred_only_flag="--prediction-only"
        output_name="demo_${frames}frames_pred_only"
    fi

    echo "=========================================="
    echo "  Generating Video"
    echo "=========================================="
    echo ""
    echo "Frames: $frames"
    echo "Duration: ~$((frames / 10))s"
    if [ -n "$start_idx" ]; then
        echo "Start index: $start_idx (fixed scene)"
        if [ -n "$pred_only_flag" ]; then
            echo "Mode: Prediction only (no GT comparison)"
            output_name="demo_${frames}frames_idx${start_idx}_pred_only"
        else
            echo "Mode: Comparison (prediction vs ground truth)"
            output_name="demo_${frames}frames_idx${start_idx}"
        fi
    else
        echo "Start index: random"
        if [ -n "$pred_only_flag" ]; then
            echo "Mode: Prediction only (no GT comparison)"
        else
            echo "Mode: Comparison (prediction vs ground truth)"
        fi
    fi
    echo ""

    mkdir -p outputs/videos

    if [ -n "$start_idx" ]; then
        python utils/generate_videos.py \
            --vqvae-checkpoint checkpoints/vqvae_v2/best.pth \
            --world-model-checkpoint checkpoints/world_model_ss/best.pth \
            --token-file data/tokens_v2/tokens_actions.npz \
            --output-dir outputs/videos \
            --num-videos 1 \
            --num-frames "$frames" \
            --fps 10 \
            --temperature 1.0 \
            --start-idx "$start_idx" \
            $pred_only_flag
    else
        python utils/generate_videos.py \
            --vqvae-checkpoint checkpoints/vqvae_v2/best.pth \
            --world-model-checkpoint checkpoints/world_model_ss/best.pth \
            --token-file data/tokens_v2/tokens_actions.npz \
            --output-dir outputs/videos \
            --num-videos 1 \
            --num-frames "$frames" \
            --fps 10 \
            --temperature 1.0 \
            $pred_only_flag
    fi

    if [ -f "outputs/videos/prediction_01.mp4" ]; then
        mv outputs/videos/prediction_01.mp4 "outputs/videos/${output_name}.mp4"
        echo ""
        echo "✅ Video saved to: outputs/videos/${output_name}.mp4"
    fi
}

cmd_dream() {
    local action_file=${1:-""}
    local show_controls=""

    # 检查第二个参数是否是 --show-controls
    if [ "$2" = "--show-controls" ]; then
        show_controls="--show-controls"
    fi

    echo "=========================================="
    echo "  Dream: WASD Action-Controlled Generation"
    echo "=========================================="
    echo ""

    if [ -z "$action_file" ]; then
        echo "❌ Error: Action file required"
        echo ""
        echo "Usage: $0 dream <action_file> [--show-controls]"
        echo ""
        echo "Example:"
        echo "  $0 dream actions.txt"
        echo "  $0 dream actions.txt --show-controls"
        echo ""
        echo "Action file format (WASD):"
        echo "  W  # 加速直行"
        echo "  W  # 加速直行"
        echo "  A  # 左转"
        echo "  D  # 右转"
        echo "  N  # 保持"
        echo ""
        exit 1
    fi

    if [ ! -f "$action_file" ]; then
        echo "❌ Error: Action file not found: $action_file"
        exit 1
    fi

    local num_actions=$(grep -v "^#" "$action_file" | grep -v "^$" | wc -l)
    local duration=$((num_actions / 10))

    echo "Action file: $action_file"
    echo "Actions: $num_actions"
    echo "Duration: ~${duration}s"
    if [ -n "$show_controls" ]; then
        echo "Controls overlay: Enabled"
    fi
    echo ""

    mkdir -p outputs/videos

    local output_name="dream_wasd_${num_actions}frames"
    if [ -n "$show_controls" ]; then
        output_name="${output_name}_controls"
    fi
    output_name="${output_name}.mp4"

    python visualize/dream.py \
        --vqvae-checkpoint checkpoints/vqvae_v2/best.pth \
        --world-model-checkpoint checkpoints/world_model_ss/best.pth \
        --token-file data/tokens_v2/tokens_actions.npz \
        --action-txt "$action_file" \
        --output "outputs/videos/${output_name}" \
        --fps 10 \
        $show_controls \
        --device cuda

    echo ""
    echo "✅ Video saved to: outputs/videos/${output_name}"
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
        cmd_video "${2:-30}" "${3:-}" "${4:-}"
        ;;
    dream)
        cmd_dream "${2:-}" "${3:-}"
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
