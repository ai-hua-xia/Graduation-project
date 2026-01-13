#!/bin/bash
# WASD动作生成 - 快速开始指南

echo "======================================"
echo "  WASD动作生成 - 快速开始"
echo "======================================"
echo
echo "功能：支持从文本文件读取WASD动作序列生成视频"
echo
echo "使用方法："
echo
echo "1. 创建动作文件 (my_actions.txt):"
echo "   W  # 加速"
echo "   W"
echo "   A  # 左转"
echo "   D  # 右转"
echo "   N  # 直行"
echo
echo "2. 运行生成命令:"
echo "   python carla_project/visualize/dream.py \\"
echo "       --vqvae-checkpoint carla_project/checkpoints/vqvae_v2/best.pth \\"
echo "       --world-model-checkpoint carla_project/checkpoints/world_model_v2/best.pth \\"
echo "       --token-file carla_project/data/tokens_v2/tokens_actions.npz \\"
echo "       --action-txt my_actions.txt \\"
echo "       --output output.mp4"
echo
echo "支持的按键："
echo "  W - 加速      (steering=0.0,  throttle=0.65)"
echo "  S - 减速      (steering=0.0,  throttle=0.42)"
echo "  A - 左转      (steering=-0.4, throttle=0.55)"
echo "  D - 右转      (steering=0.4,  throttle=0.55)"
echo "  Q - 左转+加速 (steering=-0.4, throttle=0.65)"
echo "  E - 右转+加速 (steering=0.4,  throttle=0.65)"
echo "  N - 直行      (steering=0.0,  throttle=0.55)"
echo
echo "示例文件："
echo "  - carla_project/visualize/example_actions.txt (WASD格式)"
echo "  - carla_project/visualize/example_actions_numeric.txt (数值格式)"
echo
echo "完整文档："
echo "  - carla_project/docs/WASD_README.md"
echo
echo "测试："
echo "  bash carla_project/script/test_wasd.sh"
echo
