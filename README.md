# 本科毕设：基于世界模型的自动驾驶场景预测

> 使用VQ-VAE和Transformer构建世界模型，在驾驶模拟器中实现基于动作的未来场景预测

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.16-green.svg)](https://carla.org/)
[![License](https://img.shields.io/badge/License-待定-yellow.svg)]()

## 🎯 项目概述

本项目实现了两个完整的世界模型系统，用于自动驾驶场景的视觉预测：

### 1️⃣ **CARLA项目**（主要项目，⭐ 用于毕设）
- 🚗 **大规模模型**：238M参数的Transformer世界模型
- 🎮 **WASD键盘控制**：支持交互式驾驶动作序列输入
- 📊 **完整评估体系**：PSNR/SSIM/LPIPS + 长期稳定性分析
- 🔧 **Scheduled Sampling**：缓解误差累积问题
- ✅ **训练完成**：所有模型已训练就绪，可直接使用

### 2️⃣ **MetaDrive项目**（辅助项目）
- 🌃 **风格迁移**：支持夜晚/雾天/雪天等多种场景
- 🌐 **Web演示**：提供在线交互式演示界面
- 🧪 **实验探索**：适合风格迁移和场景变换研究

**核心技术流程**：
`VQ-VAE离散化压缩` → `Transformer预测未来token` → `解码器生成图像`

## 🔥 项目亮点

- ✨ **动作条件预测**：根据WASD控制指令预测未来驾驶场景
- 🎯 **记忆增强**：引入记忆模块提升长期依赖建模能力
- 📈 **数据策略优化**：使用动作相关性采集增强数据质量
- 🎬 **可视化完善**：支持对比视频、纯预测视频、WASD指示器
- 📊 **评估系统完整**：包含单步预测、自回归生成、稳定性分析

## 📁 项目结构

```
bishe/
├── MetaDrive/              # MetaDrive世界模型（风格迁移）
│   ├── collect_data.py    # 数据采集
│   ├── train/             # 训练脚本
│   ├── utils/             # 可视化工具
│   └── web/               # Web演示
│
├── carla_project/          # CARLA世界模型（大规模+WASD控制）⭐ 主要项目
│   ├── models/            # 模型定义
│   ├── train/             # 训练脚本
│   ├── evaluate/          # 评估系统
│   ├── visualize/         # 可视化工具
│   ├── checkpoints/       # ✅ 已训练模型
│   ├── data/              # ✅ 数据集
│   ├── docs/              # 📚 完整文档
│   └── script/            # 便捷脚本
│
└── README.md              # 本文档
```

## 🚀 快速开始

### 前置要求
- Ubuntu 20.04/22.04
- NVIDIA GPU（推荐 16GB+ 显存）
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+

### CARLA项目（推荐）⭐

**所有模型已训练完成，可直接使用！无需重新训练。**

#### 📦 安装依赖

```bash
cd carla_project
pip install -r requirements_carla.txt
```

#### 🎮 1. 生成WASD控制视频（最推荐）

创建动作文件 `my_drive.txt`：
```bash
cat > my_drive.txt << 'EOF'
W
W
W
A
D
N
EOF
```

生成视频：
```bash
python visualize/dream.py \
    --vqvae-checkpoint checkpoints/vqvae_v2/best.pth \
    --world-model-checkpoint checkpoints/world_model_v2/best.pth \
    --token-file data/tokens_v2/tokens_actions.npz \
    --action-txt my_drive.txt \
    --output my_video.mp4
```

或使用便捷脚本：
```bash
./bin/model_tools.sh dream my_drive.txt --show-controls
```

#### 📊 2. 评估模型性能

```bash
python evaluate/evaluate_world_model.py \
    --vqvae-checkpoint checkpoints/vqvae_v2/best.pth \
    --world-model-checkpoint checkpoints/world_model_v2/best.pth \
    --token-file data/tokens_v2/tokens_actions.npz \
    --output evaluation_results.json
```

或使用便捷脚本：
```bash
./bin/model_tools.sh eval
```

#### 🎬 3. 生成对比视频

```bash
# 生成30帧对比视频（真实 vs 预测）
./bin/model_tools.sh video 30

# 固定起点生成100帧视频
./bin/model_tools.sh video 100 1990

# 生成纯预测视频（不显示真实画面）
./bin/model_tools.sh video 100 1990 --pred-only
```

#### 📚 4. 查看详细文档

```bash
# 项目主文档
cat carla_project/docs/README.md

# 快速开始指南
cat carla_project/docs/QUICKSTART.md

# WASD控制详解
cat carla_project/docs/WASD.md
```

### MetaDrive项目

#### 1. 数据采集
```bash
cd MetaDrive
python collect_data.py
```

#### 2. 训练VQ-VAE
```bash
python train/train_vqvae_256.py
```

#### 3. 训练世界模型
```bash
python export_vqvae_tokens.py --checkpoint checkpoints_vqvae_256/vqvae_256_epXX.pth
python train/train_world_model.py
```

#### 4. 生成视频
```bash
python utils/visualize_dream.py
```

#### 5. Web演示
```bash
cd web
python -m http.server 8000
# 访问 http://localhost:8000
```

## 📊 项目对比

| 特性 | MetaDrive项目 | CARLA项目 |
|------|--------------|-----------|
| **模拟器** | MetaDrive | CARLA 0.9.16 |
| **数据规模** | 中等 | 10,000帧 |
| **模型规模** | 小型 | 238M参数 |
| **特色功能** | 风格迁移（夜晚/雾/雪） | WASD键盘控制 |
| **训练状态** | 部分完成 | ✅ 全部完成 |
| **评估系统** | 基础 | 完整（PSNR/SSIM/稳定性） |
| **文档** | 基础 | 完整 |
| **推荐用途** | 风格迁移实验 | 主要研究和论文 |

## 🎓 CARLA项目亮点（推荐用于毕设）

### 1. 完整的训练流程
- ✅ VQ-VAE v2: Epoch 99, Loss 0.0018
- ✅ World Model (TF): Epoch 149, Loss 0.138
- ✅ World Model (SS): Epoch 2, Loss 0.050

### 2. 创新功能
- **WASD键盘控制**：支持7个按键（W/A/S/D/Q/E/N）
- **Scheduled Sampling**：缓解误差累积
- **完整评估系统**：单步预测、自回归生成、稳定性指标

### 3. 完善的文档
- 项目主文档（README.md）
- 快速开始指南（QUICKSTART.md）
- 环境配置（SETUP.md）
- WASD控制（WASD.md）
- 变更日志（CHANGELOG.md）

### 4. 便捷的脚本
- 一键训练流程
- 快速评估
- WASD测试
- 对比视频生成

## 💡 毕设建议

### 主要使用CARLA项目
1. **模型已训练完成**，可直接进行实验
2. **评估系统完整**，便于生成论文数据
3. **文档完善**，便于撰写论文
4. **创新点明确**：WASD控制、Scheduled Sampling

### 可以做的工作

#### 1. 实验与分析
- 对比Teacher Forcing vs Scheduled Sampling
- 分析长期生成的稳定性（崩溃点、半衰期）
- 不同动作序列的生成质量对比
- WASD控制的实用性验证

#### 2. 可视化与演示
- 生成多种驾驶场景视频
- 制作对比视频（真实 vs 预测）
- 展示WASD交互式控制
- 绘制评估指标图表

#### 3. 论文撰写
- 使用完整的评估数据
- 引用CARLA项目的技术细节
- 展示创新点（WASD控制、SS训练）
- 使用生成的可视化结果

#### 4. 可选：MetaDrive风格迁移
- 作为补充实验
- 展示模型的扩展性
- 对比不同模拟器的效果

## 📚 详细文档

### CARLA项目文档
- **主文档**: `carla_project/docs/README.md`
- **快速开始**: `carla_project/docs/QUICKSTART.md`
- **环境配置**: `carla_project/docs/SETUP.md`
- **WASD控制**: `carla_project/docs/WASD.md`
- **变更日志**: `carla_project/docs/CHANGELOG.md`

### MetaDrive项目
- 参考各脚本中的注释
- Web演示：`MetaDrive/web/`

## 🔧 环境要求

### CARLA项目
- Ubuntu 20.04/22.04
- NVIDIA GPU (16GB+ 显存)
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- CARLA 0.9.16

### MetaDrive项目
- Python 3.8+
- PyTorch
- MetaDrive模拟器

## 📈 项目状态

### CARLA项目 ✅
- [x] 数据采集（10,000帧）
- [x] VQ-VAE v2训练
- [x] World Model训练（TF + SS）
- [x] 评估系统实现
- [x] WASD控制实现
- [x] 文档完善
- [x] 脚本工具

### MetaDrive项目 🔄
- [x] 基础流程
- [x] 风格迁移
- [x] Web演示
- [ ] 完整评估
- [ ] 文档完善

## 🎯 推荐使用流程

1. **主要使用CARLA项目**进行研究和论文撰写
2. **参考MetaDrive项目**的风格迁移作为补充
3. 重点展示CARLA项目的创新点和完整性
4. 使用CARLA项目的评估数据和可视化结果

## 📝 引用

```
[待补充]
```

## 📄 许可

[待补充]

---

**推荐**: 优先使用 `carla_project/`，模型已训练完成，文档完善，适合毕设！

**最后更新**: 2026-01-16
