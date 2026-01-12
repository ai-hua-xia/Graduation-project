#!/bin/bash
# CARLA快速启动脚本

CARLA_DIR=~/CARLA_0.9.16
LOG_FILE=~/carla_server.log

echo "========================================="
echo "  CARLA服务器启动"
echo "========================================="
echo ""

# 检查CARLA是否已安装
if [ ! -d "$CARLA_DIR" ]; then
    echo "✗ CARLA服务器未安装！"
    echo ""
    echo "请先安装CARLA："
    echo "  1. cd ~/"
    echo "  2. wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz"
    echo "  3. mkdir CARLA_0.9.16 && tar -xzf CARLA_0.9.16.tar.gz -C CARLA_0.9.16"
    echo ""
    echo "详细安装指南: cat docs/INSTALL_SERVER.md"
    exit 1
fi

# 检查是否已经在运行
if pgrep -f "CarlaUE4" > /dev/null; then
    echo "⚠ CARLA服务器已在运行！"
    echo ""
    read -p "是否重启? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "停止现有服务器..."
        pkill -f CarlaUE4
        sleep 2
    else
        echo "保持现有服务器运行"
        exit 0
    fi
fi

echo "启动配置："
echo "  目录: $CARLA_DIR"
echo "  日志: $LOG_FILE"
echo "  端口: 2000-2002"
echo "  模式: 无渲染"
echo "  GPU: 自动选择"
echo ""

# 进入CARLA目录
cd $CARLA_DIR

# 后台启动
echo "正在启动CARLA服务器..."
nohup ./CarlaUE4.sh -RenderOffScreen > $LOG_FILE 2>&1 &

# 等待启动
sleep 5

# 检查是否成功启动
if pgrep -f "CarlaUE4" > /dev/null; then
    echo "✓ CARLA服务器已启动"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  查看日志: tail -f $LOG_FILE"
    echo "  停止服务: pkill -f CarlaUE4"
    echo "  监控GPU: nvidia-smi -l 1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # 测试连接
    echo "测试连接..."
    conda activate voyager 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh && conda activate voyager

    sleep 3

    python -c "
import carla
import sys

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print('✓ 连接成功！')
    print(f'  当前地图: {world.get_map().name}')
except Exception as e:
    print(f'⚠ 连接失败: {e}')
    print('  服务器可能还在启动中，请等待10-20秒后再试')
    sys.exit(1)
" && echo "" && echo "✓ CARLA服务器就绪，可以开始数据采集！"

else
    echo "✗ CARLA服务器启动失败"
    echo ""
    echo "请查看日志："
    echo "  tail -f $LOG_FILE"
    exit 1
fi
