# CARLA World Model

基于CARLA模拟器的自动驾驶世界模型项目，使用VQ-VAE和Transformer学习从动作预测未来视觉帧。

## 🎯 项目简介

本项目实现了一个端到端的世界模型系统，能够根据驾驶动作预测未来的视觉场景。

**核心技术**：
- **VQ-VAE v2**: 将256×256图像压缩为16×16离散tokens
- **Transformer World Model**: 238M参数，基于历史帧和动作预测下一帧
- **Scheduled Sampling**: 缓解自回归生成时的误差累积
- **WASD控制**: 支持键盘输入生成自定义驾驶视频

## 📊 当前状态

| 模块 | 状态 | 详情 |
|------|------|------|
| 数据采集 | ✅ 完成 | 10,000帧，Town03地图 |
| VQ-VAE v2 | ✅ 训练完成 | Epoch 99, Loss 0.0018 |
| World Model (TF) | ✅ 训练完成 | Epoch 149, Loss 0.138 |
| World Model (SS) | ✅ 训练完成 | Epoch 2, Loss 0.050 |
| 评估系统 | ✅ 已实现 | PSNR/SSIM/稳定性指标 |
| WASD控制 | ✅ 已实现 | 支持文本文件输入 |

**所有模型已训练完成，可直接使用！**

## 🚀 快速开始

### 使用统一工具脚本

```bash
# 查看训练状态
./bin/model_tools.sh status

# 快速评估模型
./bin/model_tools.sh eval

# 生成30帧对比视频（随机场景）
./bin/model_tools.sh video 30

# 生成100帧纯预测视频（推荐，最连续场景）
./bin/model_tools.sh video 100 1000 --pred-only

# 生成100帧对比视频（最连续场景）
./bin/model_tools.sh video 100 1000

# 诊断模型问题
./bin/model_tools.sh diagnose

# 分析视频质量
./bin/model_tools.sh analyze

# 生成论文图表
./bin/model_tools.sh figures
```

### 直接使用Python脚本

#### 方式1: 生成预测视频
```bash
# 随机场景
python utils/generate_videos.py \
    --vqvae-checkpoint checkpoints/vqvae_v2/best.pth \
    --world-model-checkpoint checkpoints/world_model_ss/best.pth \
    --token-file data/tokens_v2/tokens_actions.npz \
    --output-dir outputs/videos \
    --num-videos 1 \
    --num-frames 30 \
    --fps 10

# 固定场景（推荐）
python utils/generate_videos.py \
    --vqvae-checkpoint checkpoints/vqvae_v2/best.pth \
    --world-model-checkpoint checkpoints/world_model_ss/best.pth \
    --token-file data/tokens_v2/tokens_actions.npz \
    --output-dir outputs/videos \
    --num-videos 1 \
    --num-frames 150 \
    --fps 10 \
    --start-idx 5000
```

#### 方式2: 评估模型
```bash
python evaluate/evaluate_world_model.py \
    --vqvae-checkpoint checkpoints/vqvae_v2/best.pth \
    --world-model-checkpoint checkpoints/world_model_ss/best.pth \
    --token-file data/tokens_v2/tokens_actions.npz \
    --output outputs/evaluations/eval.json \
    --num-samples 100 \
    --num-sequences 10 \
    --sequence-length 50
```

## 🎮 WASD键盘控制

支持7个按键控制驾驶：

| 按键 | 动作 | 说明 |
|------|------|------|
| **W** | 加速 | 直行+最大油门 |
| **S** | 减速 | 直行+最小油门 |
| **A** | 左转 | 左转+中等油门 |
| **D** | 右转 | 右转+中等油门 |
| **Q** | 左转+加速 | 组合动作 |
| **E** | 右转+加速 | 组合动作 |
| **N** | 直行 | 保持中等油门 |

**示例动作文件**：
```
# 加速起步
W
W
W
# 左转弯
Q
Q
# 直行
N
N
# 右转弯
E
E
# 减速
S
```

详见 [WASD.md](docs/WASD.md)

## 📁 项目结构

```
carla_project/
├── bin/                 # 🔧 可执行脚本
│   ├── model_tools.sh  # 统一工具入口（推荐）
│   ├── show_structure.sh
│   ├── setup_env.sh
│   ├── activate.sh
│   ├── start_carla_server.sh
│   └── test_wasd.sh
├── tools/              # 🐍 Python分析工具
│   ├── analyze_video_quality.py
│   ├── diagnose_model.py
│   └── extract_losses.py
├── utils/              # 🔧 核心Python库
│   ├── generate_videos.py
│   ├── generate_figures.py
│   └── export_tokens.py
├── outputs/            # 📊 所有输出文件
│   ├── evaluations/   # 评估结果 (.json)
│   ├── videos/        # 生成视频 (.mp4)
│   ├── analysis/      # 分析图表 (.png)
│   └── figures/       # 论文图表 (.png)
├── checkpoints/        # ✅ 已训练模型
│   ├── vqvae_v2/      # VQ-VAE (240MB)
│   ├── world_model_v2/# World Model TF (2.7GB)
│   └── world_model_ss/# World Model SS (2.7GB)
├── data/
│   └── tokens_v2/     # ✅ 10,000帧tokens (3.4MB)
├── models/            # 模型定义
├── train/             # 训练脚本
├── evaluate/          # 评估脚本
└── docs/              # 📚 文档
```

## 📈 模型性能

### VQ-VAE v2
- **Codebook**: 1024 embeddings × 256 dim
- **训练**: 100 epochs
- **Loss**: 0.0018
- **压缩**: 256×256 → 16×16 tokens

### World Model v2 (Teacher Forcing)
- **参数量**: 238M
- **架构**: 16层Transformer, 16个注意力头
- **训练**: 150 epochs
- **Loss**: 0.138
- **上下文**: 4帧历史

### World Model (Scheduled Sampling)
- **基于**: World Model v2预训练
- **训练**: 3 epochs
- **Loss**: 0.050
- **优势**: 更稳定的长期生成

## 🔧 常用命令

```bash
# 查看项目结构
./bin/show_structure.sh

# 查看训练状态
./bin/model_tools.sh status

# 快速评估
./bin/model_tools.sh eval

# 生成视频
./bin/model_tools.sh video 30

# 测试WASD功能
./bin/test_wasd.sh

# 查看GPU状态
nvidia-smi

# 查看训练日志
tail -f logs/train_ss.log
```

## 📚 文档

- **[快速开始](docs/QUICKSTART.md)** - 详细使用指南
- **[环境配置](docs/SETUP.md)** - 安装和项目结构
- **[WASD控制](docs/WASD.md)** - 键盘动作控制详解
- **[变更日志](docs/CHANGELOG.md)** - 开发历史

## 🛠️ 技术栈

- **深度学习**: PyTorch 2.5.1, Mixed Precision (bf16)
- **模拟器**: CARLA 0.9.15
- **评估指标**: PSNR, SSIM, LPIPS
- **可视化**: OpenCV, Matplotlib, FFmpeg

## 💡 使用建议

### 选择模型
- **快速测试**: 使用World Model v2 (TF)
- **长期生成**: 使用World Model (SS)，更稳定
- **对比实验**: 同时测试两个模型

### 动作设计
- 保持在训练范围内：steering [-0.6, 0.6], throttle [0.4, 0.7]
- 避免频繁切换动作
- 使用平滑的动作序列

### 生成质量
- 使用Scheduled Sampling模型
- 控制生成长度（建议<300帧）
- 调整temperature和top_k参数

## 🎓 研究价值

本项目展示了：
1. **VQ-VAE在视觉压缩中的应用**
2. **Transformer在序列预测中的能力**
3. **Scheduled Sampling缓解误差累积**
4. **离散token空间的世界建模**

## 📝 引用

如果使用本项目，请引用：
```
[待补充]
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可

[待补充]

---

**项目状态**: ✅ 所有模型已训练完成，可直接使用

**最后更新**: 2026-01-13
