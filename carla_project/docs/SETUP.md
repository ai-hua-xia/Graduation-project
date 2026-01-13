# 环境配置和项目结构

## 环境配置

### 已配置环境（Voyager服务器）
- **系统**: Ubuntu 22.04
- **GPU**: 2× NVIDIA H200 NVL (143GB显存)
- **CUDA**: 12.4
- **Python**: 3.10
- **PyTorch**: 2.5.1+cu124

### CARLA服务器
- **版本**: 0.9.15
- **运行方式**: Docker容器
- **端口**: 2000
- **启动命令**: `bash carla_project/script/start_carla_server.sh`

## 项目结构

```
carla_project/
├── checkpoints/         # 模型权重
│   ├── vqvae_v2/       # VQ-VAE模型
│   ├── world_model_v2/ # World Model (TF)
│   └── world_model_ss/ # World Model (SS)
│
├── collect/            # 数据采集
│   ├── collect_large.py
│   └── utils.py
│
├── data/               # 数据集
│   ├── raw/           # 原始图像+动作
│   └── tokens_v2/     # VQ-VAE编码后的tokens
│
├── docs/               # 文档
│   ├── README.md      # 项目主文档
│   ├── QUICKSTART.md  # 快速开始
│   ├── SETUP.md       # 本文档
│   ├── WASD.md        # WASD控制
│   └── CHANGELOG.md   # 变更日志
│
├── evaluate/           # 评估
│   ├── evaluate_world_model.py
│   └── metrics.py
│
├── models/             # 模型定义
│   ├── vqvae_v2.py
│   ├── world_model.py
│   └── film.py
│
├── script/             # 便捷脚本
│   ├── run_pipeline.sh
│   ├── start_training.sh
│   └── quick_eval.sh
│
├── train/              # 训练脚本
│   ├── train_vqvae_v2.py
│   ├── train_world_model.py
│   └── train_world_model_ss.py
│
├── utils/              # 工具
│   ├── export_tokens_v2.py
│   └── dataset.py
│
└── visualize/          # 可视化
    ├── dream.py
    ├── compare_video.py
    └── example_actions.txt
```

## 核心文件说明

### 模型文件
- `models/vqvae_v2.py` - VQ-VAE v2模型（1024 codebook）
- `models/world_model.py` - 238M参数Transformer
- `models/film.py` - FiLM条件机制

### 训练脚本
- `train/train_vqvae_v2.py` - VQ-VAE训练
- `train/train_world_model.py` - World Model训练（Teacher Forcing）
- `train/train_world_model_ss.py` - Scheduled Sampling训练
- `train/config.py` - 训练配置

### 数据处理
- `collect/collect_large.py` - 大规模数据采集
- `utils/export_tokens_v2.py` - 导出VQ-VAE tokens
- `utils/dataset.py` - 数据集类

### 评估和可视化
- `evaluate/evaluate_world_model.py` - 完整评估系统
- `evaluate/metrics.py` - 评估指标（PSNR/SSIM/LPIPS）
- `visualize/dream.py` - 视频生成（支持WASD）
- `visualize/compare_video.py` - 对比视频生成

## 数据流

```
CARLA采集 → 原始数据 → VQ-VAE编码 → Tokens → World Model训练 → 视频生成
   ↓           ↓            ↓           ↓            ↓              ↓
collect/    data/raw/   vqvae_v2/  data/tokens_v2/  checkpoints/  outputs/
```

## 配置文件

### VQ-VAE配置 (`train/config.py`)
```python
VQVAE_CONFIG = {
    'num_embeddings': 1024,
    'embed_dim': 256,
    'lr': 2e-4,
    'batch_size': 64,
    'use_amp': True,
    'amp_dtype': 'bf16',
}
```

### World Model配置
```python
WM_CONFIG = {
    'num_embeddings': 1024,
    'embed_dim': 512,
    'hidden_dim': 1024,
    'num_heads': 16,
    'num_layers': 16,
    'context_frames': 4,
    'action_dim': 2,
    'lr': 5e-5,
    'batch_size': 32,
}
```

## 常用路径

### Checkpoints
- VQ-VAE: `carla_project/checkpoints/vqvae_v2/best.pth`
- World Model (TF): `carla_project/checkpoints/world_model_v2/best.pth`
- World Model (SS): `carla_project/checkpoints/world_model_ss/best.pth`

### 数据
- 原始数据: `carla_project/data/raw/`
- Tokens: `carla_project/data/tokens_v2/tokens_actions.npz`

### 日志
- VQ-VAE: `carla_project/logs/train_vqvae_v2.log`
- World Model: `carla_project/logs/train_wm_v2.log`
- SS训练: `carla_project/logs/train_ss.log`

## 磁盘使用

| 目录 | 大小 | 说明 |
|------|------|------|
| checkpoints/ | ~60GB | 模型权重 |
| data/raw/ | ~2GB | 原始图像 |
| data/tokens_v2/ | ~100MB | Token数据 |
| logs/ | ~200MB | 训练日志 |

**优化建议**: 删除中间checkpoint可节省约48GB空间

## 环境变量

```bash
# CUDA设备
export CUDA_VISIBLE_DEVICES=0,1

# Python不缓冲输出
export PYTHONUNBUFFERED=1

# CARLA服务器
export CARLA_HOST=localhost
export CARLA_PORT=2000
```

## 下一步

- 查看 [QUICKSTART.md](QUICKSTART.md) 开始训练
- 查看 [WASD.md](WASD.md) 学习键盘控制
- 查看 [README.md](README.md) 了解项目概况
