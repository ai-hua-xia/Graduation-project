#!/bin/bash
# 在tmux中启动训练任务

set -e

echo "========================================="
echo "  Start Training in tmux"
echo "========================================="
echo ""

# 检查参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 <task_name> <training_type> [gpu_id]"
    echo ""
    echo "Training types:"
    echo "  vqvae       - Train VQ-VAE v2"
    echo "  world_model - Train World Model (Teacher Forcing)"
    echo "  ss          - Train World Model (Scheduled Sampling)"
    echo ""
    echo "Example:"
    echo "  $0 train_vqvae vqvae 0"
    echo "  $0 train_wm world_model 1"
    echo "  $0 train_ss ss 1"
    exit 1
fi

TASK_NAME=$1
TRAINING_TYPE=$2
GPU_ID=${3:-0}

# 创建日志目录
mkdir -p carla_project/logs

case $TRAINING_TYPE in
    vqvae)
        echo "Starting VQ-VAE v2 training on GPU $GPU_ID..."
        tmux new-session -d -s "$TASK_NAME" \
            "CUDA_VISIBLE_DEVICES=$GPU_ID python carla_project/train/train_vqvae_v2.py \
                --data-path carla_project/data/raw \
                --save-dir carla_project/checkpoints/vqvae_v2 \
                --epochs 100 \
                --batch-size 32 \
                2>&1 | tee carla_project/logs/${TASK_NAME}.log"
        ;;

    world_model)
        echo "Starting World Model (TF) training on GPU $GPU_ID..."
        tmux new-session -d -s "$TASK_NAME" \
            "CUDA_VISIBLE_DEVICES=$GPU_ID python carla_project/train/train_world_model.py \
                --token-path carla_project/data/tokens_v2/tokens_actions.npz \
                --save-dir carla_project/checkpoints/world_model_v2 \
                --epochs 150 \
                --batch-size 32 \
                2>&1 | tee carla_project/logs/${TASK_NAME}.log"
        ;;

    ss)
        echo "Starting World Model (SS) training on GPU $GPU_ID..."
        if [ ! -f "carla_project/checkpoints/world_model_v2/best.pth" ]; then
            echo "Error: World Model checkpoint not found!"
            echo "Please train World Model first."
            exit 1
        fi

        tmux new-session -d -s "$TASK_NAME" \
            "CUDA_VISIBLE_DEVICES=$GPU_ID python carla_project/train/train_world_model_ss.py \
                --token-path carla_project/data/tokens_v2/tokens_actions.npz \
                --pretrained carla_project/checkpoints/world_model_v2/best.pth \
                --save-dir carla_project/checkpoints/world_model_ss \
                --epochs 100 \
                --seq-len 12 \
                --ss-schedule linear \
                --ss-k 0.5 \
                --batch-size 8 \
                2>&1 | tee carla_project/logs/${TASK_NAME}.log"
        ;;

    *)
        echo "Error: Unknown training type: $TRAINING_TYPE"
        exit 1
        ;;
esac

echo ""
echo "✓ Training started in tmux session: $TASK_NAME"
echo ""
echo "Commands:"
echo "  View session:  tmux attach -t $TASK_NAME"
echo "  View log:      tail -f carla_project/logs/${TASK_NAME}.log"
echo "  List sessions: tmux list-sessions"
echo "  Kill session:  tmux kill-session -t $TASK_NAME"
echo ""
