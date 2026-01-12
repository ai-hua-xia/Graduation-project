# 🚀 快速开始指南（Voyager环境）

**环境**：voyager conda环境
**Python**：3.11.9
**GPU**：2× NVIDIA H200 NVL (150GB显存)
**状态**：✅ 所有依赖已安装

---

## 第一步：验证环境

```bash
cd ~/HunyuanWorld-Voyager/bishe/carla_project

# 激活环境
conda activate voyager

# 检查环境
python -c "import torch; import carla; print('✓ 环境检查通过')"
```

**预期输出**：所有依赖显示 ✓

---

## 第二步：安装CARLA服务器

### 下载和解压

```bash
# 进入home目录
cd ~/

# 下载CARLA 0.9.16
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz

# 解压
mkdir CARLA_0.9.16
tar -xzf CARLA_0.9.16.tar.gz -C CARLA_0.9.16

# 验证
ls CARLA_0.9.16/CarlaUE4.sh
```

**如果wget失败**，从浏览器下载：
- 访问：https://github.com/carla-simulator/carla/releases/tag/0.9.16
- 下载：CARLA_0.9.16.tar.gz
- 上传到服务器后解压

---

## 第三步：启动CARLA服务器

### 方式1：使用启动脚本（推荐）

```bash
cd ~/HunyuanWorld-Voyager/bishe/carla_project
./script/start_carla_server.sh
```

### 方式2：手动启动

```bash
cd ~/CARLA_0.9.16
nohup ./CarlaUE4.sh -RenderOffScreen > ~/carla_server.log 2>&1 &

# 查看日志
tail -f ~/carla_server.log

# 等待10-20秒启动
```

### 验证服务器运行

```bash
conda activate voyager

python -c "
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
print('✓ CARLA连接成功')
print(f'当前地图: {world.get_map().name}')
"
```

---

## 第四步：开始数据采集

### 快速测试（10 episodes）

```bash
cd ~/HunyuanWorld-Voyager/bishe/carla_project
conda activate voyager

cd collect
python carla_collector.py \
    --host localhost \
    --port 2000 \
    --episodes 10 \
    --output ../data/raw
```

**预期输出**：
- 采集进度条
- 保存到 `data/raw/episode_0000/` 等目录
- 每个episode约200帧

### 验证数据质量

```bash
# 查看采集的数据
ls data/raw/episode_0000/images/ | wc -l  # 应该有约200张图片

# 查看第一帧
# 如果有图形界面：
eog data/raw/episode_0000/images/frame_000000.png

# 或复制到本地查看
```

**关键检查**：人眼查看连续帧，看能否明显看到转向效果！

---

## 第五步：完整流程（一键运行）

如果测试数据质量OK，运行完整流程：

```bash
cd ~/HunyuanWorld-Voyager/bishe/carla_project
conda activate voyager

# 一键运行所有步骤（交互式）
./script/run_all_voyager.sh
```

这个脚本会依次：
1. ✅ 检查环境
2. 📥 采集数据（可自定义episodes数）
3. 🎨 训练VQ-VAE（可自定义epochs）
4. 🔢 导出Tokens
5. 🧠 训练World Model
6. 🎬 生成测试视频

---

## 第六步：分步运行（推荐）

### 1. 训练VQ-VAE

```bash
cd train
python train_vqvae.py \
    --data-path ../data/raw \
    --save-dir ../checkpoints/vqvae \
    --epochs 50 \
    --batch-size 32
```

**训练时间**：约1-2小时（50 epochs）

### 2. 导出Tokens

```bash
cd utils
python export_tokens.py \
    --data-path ../data/raw \
    --vqvae-checkpoint ../checkpoints/vqvae/best.pth \
    --output ../data/tokens/tokens_actions.npz
```

### 3. 训练World Model

```bash
cd train
python train_world_model.py \
    --token-path ../data/tokens/tokens_actions.npz \
    --save-dir ../checkpoints/world_model \
    --epochs 100 \
    --batch-size 32
```

**训练时间**：约2-4小时（100 epochs）

### 4. 生成视频

```bash
cd visualize
python dream.py \
    --vqvae-checkpoint ../checkpoints/vqvae/best.pth \
    --world-model-checkpoint ../checkpoints/world_model/best.pth \
    --token-file ../data/tokens/tokens_actions.npz \
    --output ../outputs/dream_result.mp4 \
    --num-frames 200
```

---

## 🛠️ 常用命令

### 监控训练

```bash
# 监控GPU使用
nvidia-smi -l 1

# 查看训练进度（在train目录）
tail -f train.log  # 如果有日志文件
```

### 管理CARLA服务器

```bash
# 查看CARLA进程
ps aux | grep Carla

# 停止CARLA
pkill -f CarlaUE4

# 查看CARLA日志
tail -f ~/carla_server.log

# 重启CARLA
pkill -f CarlaUE4
sleep 2
./script/start_carla_server.sh
```

### 清理磁盘空间

```bash
# 删除中间数据（谨慎！）
rm -rf data/raw/*
rm -rf checkpoints/vqvae/*
rm -rf checkpoints/world_model/*
```

---

## ⚡ 性能优化建议

### GPU分配

你有2个H200 NVL，建议：

```bash
# GPU 1运行CARLA服务器
CUDA_VISIBLE_DEVICES=1 ~/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen &

# GPU 0用于训练
CUDA_VISIBLE_DEVICES=0 python train_vqvae.py ...
```

### Batch Size调整

你的显存超大（150GB），可以增大batch size：

```bash
# VQ-VAE训练
python train_vqvae.py --batch-size 128  # 默认64

# World Model训练
python train_world_model.py --batch-size 128  # 默认64
```

---

## 📊 预期时间线

| 步骤 | 时间（测试） | 时间（正式） |
|------|------------|------------|
| 环境配置 | ✅ 完成 | ✅ 完成 |
| CARLA安装 | 30分钟 | 30分钟 |
| 数据采集 | 1小时（10 episodes） | 5-8小时（100 episodes） |
| VQ-VAE训练 | 30分钟（10 epochs） | 2小时（100 epochs） |
| World Model训练 | 1小时（20 epochs） | 4小时（200 epochs） |
| 视频生成 | 5分钟 | 5分钟 |
| **总计** | **约3小时** | **约12小时** |

---

## 🎯 关键检查点

### Checkpoint 1：数据采集后

```bash
# 1. 检查数据量
ls data/raw/ | wc -l  # episode数量

# 2. 检查单个episode
ls data/raw/episode_0000/images/ | wc -l  # 应该约200帧

# 3. 人眼验证转向可见性
# 打开连续5帧图片，看是否能看到明显转向
```

**决策**：如果转向明显 → 继续；如果不明显 → 调整采集参数

### Checkpoint 2：VQ-VAE训练后

```bash
# 检查重建质量
# 应该生成了一些重建样例图片
```

### Checkpoint 3：World Model训练后

```bash
# 检查损失收敛
# CE loss应该降到2.5左右
```

### Checkpoint 4：视频生成后

```bash
# 播放视频
vlc outputs/dream_result.mp4

# 或复制到本地查看
```

**评估**：
- 画面是否稳定？
- 转向是否响应？
- 是否有明显artifact？

---

## 🆘 故障排除

### 问题1：CARLA连接失败

```bash
# 检查服务器是否运行
ps aux | grep Carla

# 检查端口
netstat -tuln | grep 2000

# 重启服务器
pkill -f CarlaUE4 && ./script/start_carla_server.sh
```

### 问题2：训练OOM（内存不足）

虽然你有150GB显存，但如果仍然OOM：

```bash
# 减小batch size
python train_vqvae.py --batch-size 16

# 或使用混合精度（已默认启用）
```

### 问题3：数据采集很慢

```bash
# 检查是否在同步模式
# 在collect/config.py中调整

# 或降低采集帧率
FRAMES_PER_EPISODE = 100  # 从200降到100
```

---

## 📞 获取帮助

如果遇到问题：

1. **查看日志**
   - CARLA: `~/carla_server.log`
   - Python错误信息

2. **检查环境**
   ```bash
   python -c "import torch; import carla; print('✓ 环境OK')"
   ```

3. **验证CARLA版本**
   ```bash
   python -c "import carla; print(dir(carla))"
   ```

---

**准备好了吗？开始你的第一次数据采集！** 🚗💨

```bash
cd ~/HunyuanWorld-Voyager/bishe/carla_project
conda activate voyager
./script/start_carla_server.sh  # 先启动服务器
# 等待10-20秒
cd collect && python carla_collector.py --episodes 5  # 小规模测试
```
