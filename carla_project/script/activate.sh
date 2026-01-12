#!/bin/bash
# 项目激活脚本 - 快速进入工作状态

# 激活voyager环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate voyager

# 进入项目目录
cd ~/HunyuanWorld-Voyager/bishe/carla_project

# 显示状态
clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║        🚗 CARLA World Model Project                       ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "环境: voyager (Python 3.11.9)"
echo "位置: $(pwd)"
echo "GPU: 2× NVIDIA H200 NVL (150GB)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查CARLA服务器
if pgrep -f "CarlaUE4" > /dev/null; then
    echo "✓ CARLA服务器正在运行"
else
    echo "⚠ CARLA服务器未运行"
    echo "  启动命令: ./script/start_carla_server.sh"
fi
echo ""

# 快速命令提示
echo "快速命令："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  检查环境:     python -c 'import torch; import carla; print(\"OK\")'"
echo "  启动CARLA:    ./script/start_carla_server.sh"
echo "  采集数据:     cd collect && python carla_collector.py --episodes 5"
echo "  完整流程:     ./script/run_all_voyager.sh"
echo "  查看文档:     cat docs/QUICKSTART.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 启动bash
exec bash
