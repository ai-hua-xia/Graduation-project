# CARLA项目 - 环境配置完成总结

## ✅ 当前状态

### 环境信息
- **Conda环境**：voyager
- **Python版本**：3.11.9
- **PyTorch版本**：2.4.0 (CUDA 12.4)
- **CARLA API**：0.9.16
- **GPU**：2× NVIDIA H200 NVL (150GB显存/个)

### 已安装依赖
```
✓ torch 2.4.0+cu124
✓ torchvision 0.19.0+cu124
✓ opencv-python 4.9.0.80
✓ numpy 1.26.4
✓ pillow 11.3.0
✓ imageio 2.34.0
✓ imageio-ffmpeg 0.5.1
✓ matplotlib 3.10.7
✓ scipy 1.11.4
✓ seaborn 0.13.2
✓ tqdm 4.66.2
✓ pyyaml 6.0.3
✓ psutil 7.1.3
✓ carla 0.9.16 ⭐
✓ h5py 3.15.1
```

---

## 📂 项目结构

```
carla_project/
├── 📖 文档 (docs/)
│   ├── README_CARLA.md          # 完整项目文档
│   ├── QUICKSTART.md            # 快速开始指南 ⭐⭐⭐
│   ├── INSTALL_SERVER.md        # CARLA服务器安装指南
│   ├── COMPARISON.md            # CARLA vs MetaDrive对比分析
│   ├── STATUS.md                # 项目状态总结
│   └── debug_history.md         # 调试记录
│
├── 🚀 启动脚本 (script/)
│   ├── start_carla_server.sh    # CARLA服务器启动脚本
│   ├── run_all_voyager.sh       # 完整流程运行脚本
│   ├── run_all.sh               # 完整流程（通用）
│   ├── setup_env.sh             # 环境配置脚本
│   └── activate.sh              # 快速激活脚本
│
├── 📥 数据采集 (collect/)
│   ├── carla_collector.py       # 主采集程序
│   ├── config.py                # 采集配置（转向优先）
│   └── utils.py                 # 工具函数
│
├── 🧠 模型 (models/)
│   ├── vqvae.py                 # VQ-VAE模型
│   ├── world_model.py           # Transformer世界模型
│   └── film.py                  # FiLM调制层
│
├── 🎓 训练 (train/)
│   ├── train_vqvae.py           # VQ-VAE训练
│   ├── train_world_model.py     # 世界模型训练（课程学习）
│   └── config.py                # 训练配置
│
├── 🛠️ 工具 (utils/)
│   ├── dataset.py               # 数据集类
│   └── export_tokens.py         # Token导出
│
├── 🎬 可视化 (visualize/)
│   └── dream.py                 # 视频生成
│
└── 📁 数据目录
    ├── data/raw/                # 原始CARLA数据
    ├── data/tokens/             # Token化数据
    ├── checkpoints/vqvae/       # VQ-VAE模型权重
    ├── checkpoints/world_model/ # 世界模型权重
    └── outputs/                 # 生成的视频
```

---

## 🎯 下一步行动

### ⏰ 现在立即做

1. **安装CARLA服务器**
   ```bash
   cd ~/
   wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz
   mkdir CARLA_0.9.16
   tar -xzf CARLA_0.9.16.tar.gz -C CARLA_0.9.16
   ```

2. **启动CARLA服务器**
   ```bash
   cd ~/HunyuanWorld-Voyager/bishe/carla_project
   conda activate voyager
   ./script/start_carla_server.sh
   ```

3. **测试连接**
   ```bash
   python -c "import carla; carla.Client('localhost', 2000).get_world()"
   ```

### 📅 今天完成

4. **采集测试数据**（5-10 episodes）
   ```bash
   cd collect
   python carla_collector.py --episodes 5
   ```

5. **验证数据质量**
   - 检查图像数量
   - **人眼查看连续帧，确认转向可见性**
   - 这是最关键的验证！

### 📆 本周完成

6. **如果数据质量OK**
   - 采集完整数据集（100 episodes）
   - 训练VQ-VAE（50-100 epochs）
   - 导出Tokens
   - 训练World Model（100-200 epochs）

7. **生成测试视频**
   - 评估效果
   - 与MetaDrive对比

---

## 📚 关键文档速查

### 第一次使用？
👉 **阅读：[QUICKSTART.md](QUICKSTART.md)**

### 安装CARLA服务器？
👉 **阅读：[INSTALL_SERVER.md](INSTALL_SERVER.md)**

### 想了解技术细节？
👉 **阅读：[README_CARLA.md](README_CARLA.md)**

### 想知道为什么换CARLA？
👉 **阅读：[COMPARISON.md](COMPARISON.md)**

---

## 🔧 常用命令速查表

### 环境管理
```bash
# 激活环境
conda activate voyager

# 检查环境
python -c "import torch; import carla; print('OK')"

# 查看已安装包
pip list | grep -E "torch|carla|opencv"
```

### CARLA服务器
```bash
# 启动
./script/start_carla_server.sh

# 停止
pkill -f CarlaUE4

# 查看日志
tail -f ~/carla_server.log

# 检查进程
ps aux | grep Carla
```

### 数据采集
```bash
cd collect

# 小规模测试
python carla_collector.py --episodes 5

# 正式采集
python carla_collector.py --episodes 100 --output ../data/raw
```

### 训练
```bash
cd train

# VQ-VAE
python train_vqvae.py --epochs 50 --batch-size 32

# World Model
python train_world_model.py --epochs 100 --batch-size 32
```

### 生成视频
```bash
cd visualize

python dream.py \
    --vqvae-checkpoint ../checkpoints/vqvae/best.pth \
    --world-model-checkpoint ../checkpoints/world_model/best.pth \
    --token-file ../data/tokens/tokens_actions.npz \
    --num-frames 200
```

### GPU监控
```bash
# 实时监控
nvidia-smi -l 1

# 查看显存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## ⚠️ 重要提示

### GPU分配建议
你有2个H200 NVL，建议：
- **GPU 1**：运行CARLA服务器
- **GPU 0**：训练模型

```bash
# CARLA使用GPU 1
CUDA_VISIBLE_DEVICES=1 ~/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen &

# 训练使用GPU 0
CUDA_VISIBLE_DEVICES=0 python train_vqvae.py ...
```

### 关键验证点

#### ✅ Checkpoint 1：数据质量
在开始训练前，**必须人眼验证**：
```bash
# 查看连续5帧
ls data/raw/episode_0000/images/frame_00000{0..5}.png
```

**问自己**：我能明显看到车在转向吗？
- ✅ 如果是 → 继续
- ❌ 如果否 → 调整采集参数或重新考虑

#### ✅ Checkpoint 2：VQ-VAE重建
训练后检查重建质量（应该生成样例图片）

#### ✅ Checkpoint 3：World Model收敛
CE loss应该降到2.5左右

#### ✅ Checkpoint 4：视频生成
最终视频应该：
- 画面稳定
- 转向响应明显
- 无严重artifact

---

## 🎓 技术亮点（毕设答辩）

### 1. 问题发现
> "通过定量分析MetaDrive数据，发现开放道路场景转向时光流幅度仅2-3像素，导致模型难以学习动作-视觉映射。"

### 2. 解决方案
> "提出使用CARLA城市场景 + 课程学习策略，从数据和训练两个维度增强动作响应性。"

### 3. 技术创新
- **数据层**：转向优先采集（70%转向样本）
- **模型层**：FiLM动作调制 + 3层MLP编码器
- **训练层**：课程学习（平滑权重0→0.02）

### 4. 对比实验
- MetaDrive vs CARLA数据质量对比
- 有/无课程学习的消融实验
- 不同平滑权重的效果曲线

---

## 📊 预期性能

### 数据质量
| 指标 | MetaDrive | CARLA（预期） |
|------|-----------|--------------|
| 转向光流幅度 | 2-3像素 | 10-20像素 |
| 相邻帧SSIM | >0.95 | 0.85-0.90 |
| 转向可见性 | 肉眼难辨 | 明显可见 |

### 模型性能
| 指标 | 目标值 |
|------|--------|
| VQ-VAE PSNR | >25 dB |
| Token利用率 | >85% |
| 生成帧率 | 15-20 FPS |
| World Model Loss | <2.5 |

---

## 🔗 资源链接

### 官方文档
- [CARLA Documentation](https://carla.readthedocs.io/)
- [CARLA Python API](https://carla.readthedocs.io/en/latest/python_api/)
- [CARLA GitHub](https://github.com/carla-simulator/carla)

### 论文参考
- [VQ-VAE (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937)
- [World Models (Ha & Schmidhuber, 2018)](https://arxiv.org/abs/1803.10122)
- [FiLM (Perez et al., 2018)](https://arxiv.org/abs/1709.07871)
- [GAIA-1](https://arxiv.org/abs/2309.17080)

---

## 💬 FAQ

### Q: 为什么用CARLA 0.9.16而不是0.9.15？
A: PyPI上没有0.9.15，0.9.16向后兼容且更稳定。

### Q: 训练需要多长时间？
A:
- VQ-VAE: 1-2小时（100 epochs）
- World Model: 2-4小时（200 epochs）
- 总计约6-8小时（使用H200 NVL）

### Q: 需要多少存储空间？
A:
- 原始数据：~20GB（100 episodes）
- Token文件：~2GB
- 模型权重：~500MB
- 总计：~25GB

### Q: 如果效果还是不好怎么办？
A: 参考COMPARISON.md中的Plan B（真实视频预训练）

---

## 🎉 准备就绪！

你的环境已经100%配置完成，现在可以：

1. **立即行动**：安装CARLA服务器
2. **今天完成**：采集测试数据，验证质量
3. **本周完成**：完整训练流程

**祝你实验顺利！有任何问题随时查阅文档或询问。**

---

**最后更新**：2026-01-11
**状态**：✅ 环境配置完成，等待CARLA服务器安装
