#!/bin/bash
# 环境检查和配置脚本（使用voyager环境）

echo "========================================="
echo "  CARLA项目环境检查与配置"
echo "========================================="
echo ""

# 激活voyager环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate voyager

echo "✓ 激活voyager环境"
echo "  Python版本: $(python --version)"
echo ""

# 检查已有依赖
echo "检查已安装的依赖..."
echo ""

check_package() {
    pkg_name=$1
    required_version=$2

    installed=$(pip show $pkg_name 2>/dev/null | grep Version | cut -d' ' -f2)

    if [ -z "$installed" ]; then
        echo "  ✗ $pkg_name - 未安装"
        return 1
    else
        echo "  ✓ $pkg_name - $installed"
        return 0
    fi
}

# 核心依赖检查
check_package "torch" "2.0.0"
torch_ok=$?

check_package "torchvision" "0.15.0"
torchvision_ok=$?

check_package "opencv-python" "4.8.0"
opencv_ok=$?

check_package "numpy" "1.24.0"
numpy_ok=$?

check_package "pillow" "10.0.0"
pillow_ok=$?

check_package "imageio" "2.31.0"
imageio_ok=$?

check_package "imageio-ffmpeg" "0.4.9"
ffmpeg_ok=$?

check_package "matplotlib" "3.7.0"
matplotlib_ok=$?

check_package "scipy" "1.11.0"
scipy_ok=$?

check_package "tqdm" "4.66.0"
tqdm_ok=$?

check_package "pyyaml" "6.0"
yaml_ok=$?

check_package "psutil" "5.9.0"
psutil_ok=$?

check_package "seaborn" "0.12.0"
seaborn_ok=$?

echo ""

# 检查CARLA
echo "检查CARLA Python API..."
check_package "carla" "0.9.15"
carla_ok=$?

# 如果没有CARLA，安装
if [ $carla_ok -ne 0 ]; then
    echo ""
    echo "CARLA未安装，准备安装..."
    echo ""
    echo "选项1：使用pip安装（推荐）"
    echo "  pip install carla==0.9.15"
    echo ""
    echo "选项2：从.whl文件安装"
    echo "  下载：https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz"
    echo "  解压后在PythonAPI/carla/dist/目录找到对应Python版本的.whl文件"
    echo ""

    read -p "是否现在通过pip安装CARLA? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install carla==0.9.15

        # 验证安装
        if python -c "import carla; print('CARLA version:', carla.__version__)" 2>/dev/null; then
            echo "✓ CARLA安装成功"
        else
            echo "✗ CARLA安装失败，请手动安装"
            exit 1
        fi
    fi
fi

echo ""

# 检查可选依赖
echo "检查可选依赖..."
check_package "h5py" "3.9.0"
h5py_ok=$?

if [ $h5py_ok -ne 0 ]; then
    echo ""
    read -p "是否安装h5py（用于数据存储）? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install h5py>=3.9.0
    fi
fi

echo ""
echo "========================================="
echo "  环境检查完成"
echo "========================================="
echo ""

# 生成环境报告
echo "环境报告："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python --version
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip list | grep -E "torch|opencv|numpy|carla|pillow|imageio|matplotlib|scipy|tqdm|yaml|psutil|seaborn|h5py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 测试CARLA连接（如果安装了）
if python -c "import carla" 2>/dev/null; then
    echo "测试CARLA Python API..."
    python -c "import carla; print('CARLA API version:', carla.__version__)"
    echo ""
    echo "注意：CARLA Python API已安装，但仍需要CARLA服务器！"
    echo ""
    echo "启动CARLA服务器的方法："
    echo "  1. 下载CARLA服务器: https://github.com/carla-simulator/carla/releases/tag/0.9.15"
    echo "  2. 解压: tar -xzf CARLA_0.9.15.tar.gz"
    echo "  3. 运行: cd CARLA_0.9.15 && ./CarlaUE4.sh"
    echo "     或无渲染模式: ./CarlaUE4.sh -RenderOffScreen"
fi

echo ""
echo "✓ 环境配置完成，可以开始使用CARLA项目！"
