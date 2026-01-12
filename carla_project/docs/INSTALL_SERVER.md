# CARLA服务器安装指南（非Docker）

## ✅ 环境已配置完成

你的voyager环境已经包含所有依赖：
- ✅ Python 3.11.9
- ✅ PyTorch 2.4.0 (CUDA 12.4)
- ✅ CARLA Python API 0.9.16
- ✅ 所有其他依赖
- ✅ 2× NVIDIA H200 NVL (150GB 显存)

---

## 📥 安装CARLA服务器

### 方法1：下载预编译版本（推荐）

```bash
# 1. 进入工作目录
cd ~/

# 2. 下载CARLA 0.9.16
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz

# 3. 解压
mkdir CARLA_0.9.16
tar -xzf CARLA_0.9.16.tar.gz -C CARLA_0.9.16

# 4. 验证
ls CARLA_0.9.16/
# 应该看到: CarlaUE4.sh, Import/, PythonAPI/, Unreal/, 等文件
```

### 方法2：从GitHub下载（备选）

如果wget失败，可以从GitHub Release下载：
```bash
# 访问：https://github.com/carla-simulator/carla/releases/tag/0.9.16
# 下载：CARLA_0.9.16.tar.gz
# 上传到服务器后解压
```

---

## 🚀 启动CARLA服务器

### 基础启动（有渲染窗口）

```bash
cd ~/CARLA_0.9.16
./CarlaUE4.sh
```

**注意**：这需要显示器。如果是远程服务器，使用下面的无渲染模式。

### 无渲染模式（推荐用于服务器）

```bash
cd ~/CARLA_0.9.16

# 方式1：完全无渲染（最快）
./CarlaUE4.sh -RenderOffScreen

# 方式2：低质量渲染（节省资源）
./CarlaUE4.sh -quality-level=Low

# 方式3：指定端口（如果默认2000被占用）
./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000
```

### 后台运行

```bash
cd ~/CARLA_0.9.16
nohup ./CarlaUE4.sh -RenderOffScreen > carla_server.log 2>&1 &

# 查看日志
tail -f carla_server.log

# 查看进程
ps aux | grep Carla

# 停止服务器
pkill -f CarlaUE4
```

---

## 🔍 验证服务器运行

### 检查端口

```bash
# 查看CARLA端口（默认2000-2002）
netstat -tuln | grep 2000
# 或
ss -tuln | grep 2000
```

### 测试连接

```bash
# 激活环境
conda activate voyager

# 测试连接
python -c "
import carla
import time

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # 获取世界信息
    world = client.get_world()
    print(f'✓ 连接成功！')
    print(f'  当前地图: {world.get_map().name}')
    print(f'  天气: {world.get_weather()}')

except Exception as e:
    print(f'✗ 连接失败: {e}')
    print('  请确保CARLA服务器正在运行')
"
```

---

## ⚙️ 常用配置

### 性能优化

```bash
# 如果GPU显存够用，可以启用高质量
./CarlaUE4.sh -quality-level=Epic -RenderOffScreen

# 如果要节省资源
./CarlaUE4.sh -quality-level=Low -RenderOffScreen

# 固定时间步长（用于数据采集）
./CarlaUE4.sh -RenderOffScreen -benchmark -fps=20
```

### 多GPU配置

你有2个H200 NVL，可以指定使用哪个GPU：

```bash
# 使用GPU 0
CUDA_VISIBLE_DEVICES=0 ./CarlaUE4.sh -RenderOffScreen

# 使用GPU 1（让GPU 0用于训练）
CUDA_VISIBLE_DEVICES=1 ./CarlaUE4.sh -RenderOffScreen
```

---

## 🛠️ 故障排除

### 问题1：端口被占用

```bash
# 查找占用2000端口的进程
lsof -i :2000

# 杀死进程
kill -9 <PID>

# 或使用其他端口
./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=3000
```

### 问题2：服务器崩溃

```bash
# 查看日志
cat carla_server.log

# 常见原因：
# - GPU驱动问题：更新驱动
# - 内存不足：降低质量或关闭其他程序
# - 依赖缺失：安装依赖

# 安装依赖（Ubuntu）
sudo apt-get update
sudo apt-get install libvulkan1 vulkan-utils
```

### 问题3：连接超时

```bash
# 增加超时时间
python -c "
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(30.0)  # 增加到30秒
world = client.get_world()
print('连接成功')
"
```

---

## 📝 推荐启动脚本

创建一个启动脚本 `~/start_carla.sh`：

```bash
#!/bin/bash

# CARLA启动脚本

CARLA_DIR=~/CARLA_0.9.16
LOG_FILE=~/carla_server.log

echo "启动CARLA服务器..."
echo "  目录: $CARLA_DIR"
echo "  日志: $LOG_FILE"
echo "  端口: 2000-2002"
echo "  模式: 无渲染"

cd $CARLA_DIR

# 检查是否已经在运行
if pgrep -f "CarlaUE4" > /dev/null; then
    echo "✗ CARLA已在运行！"
    echo "  如需重启，先执行: pkill -f CarlaUE4"
    exit 1
fi

# 后台启动
nohup ./CarlaUE4.sh -RenderOffScreen > $LOG_FILE 2>&1 &

echo "✓ CARLA服务器已启动"
echo ""
echo "查看日志: tail -f $LOG_FILE"
echo "停止服务: pkill -f CarlaUE4"
echo "测试连接: python -c 'import carla; carla.Client(\"localhost\", 2000).get_world()'"
```

使用：
```bash
chmod +x ~/start_carla.sh
~/start_carla.sh
```

---

## 🎯 下一步

1. **启动CARLA服务器**
   ```bash
   cd ~/CARLA_0.9.16
   ./CarlaUE4.sh -RenderOffScreen
   ```

2. **测试连接**
   ```bash
   conda activate voyager
   cd ~/HunyuanWorld-Voyager/bishe/carla_project
   python -c "import carla; carla.Client('localhost', 2000).get_world()"
   ```

3. **开始数据采集**
   ```bash
   cd collect
   python carla_collector.py --episodes 10
   ```

---

## 💡 提示

- **GPU资源分配**：建议用GPU 1运行CARLA，GPU 0用于训练
- **长时间运行**：使用tmux或screen保持会话
- **监控资源**：`nvidia-smi -l 1` 监控GPU使用
- **日志管理**：定期清理日志文件

---

**服务器配置完成后，就可以开始数据采集了！**
