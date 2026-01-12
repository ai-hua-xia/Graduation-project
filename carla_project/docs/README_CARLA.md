# CARLA World Model Project

基于CARLA仿真器的世界模型视频生成系统

---

## 📁 项目结构

```
carla_project/
├── collect/                    # 数据采集
│   ├── carla_collector.py     # CARLA数据采集主程序
│   ├── config.py              # 采集配置
│   └── utils.py               # 采集工具函数
├── data/                       # 数据目录
│   ├── raw/                   # 原始CARLA数据
│   ├── processed/             # 处理后的数据
│   └── tokens/                # Token化数据
├── models/                     # 模型定义
│   ├── vqvae.py              # VQ-VAE模型
│   ├── world_model.py        # Transformer世界模型
│   └── film.py               # FiLM调制层
├── train/                      # 训练脚本
│   ├── train_vqvae.py        # VQ-VAE训练
│   ├── train_world_model.py  # 世界模型训练
│   └── config.py             # 训练配置
├── utils/                      # 工具函数
│   ├── dataset.py            # 数据集类
│   ├── export_tokens.py      # Token导出
│   └── video_utils.py        # 视频处理
├── visualize/                  # 可视化生成
│   ├── dream.py              # 视频生成
│   └── config.py             # 生成配置
├── checkpoints/                # 模型权重
│   ├── vqvae/
│   └── world_model/
└── outputs/                    # 生成结果
```

---

## 🚀 快速开始

### 1. 环境配置

#### 安装CARLA（推荐：Docker方式）

```bash
# 方式1：Docker（推荐）
docker pull carlasim/carla:0.9.15
docker run -p 2000-2002:2000-2002 --runtime=nvidia --gpus all carlasim/carla:0.9.15

# 方式2：从源码编译（高级）
# 参考：https://carla.readthedocs.io/en/latest/build_linux/
```

#### 安装Python依赖

```bash
cd carla_project
pip install -r requirements_carla.txt
```

---

### 2. 数据采集

```bash
# 启动CARLA服务器（另一个终端）
./CarlaUE4.sh

# 运行采集脚本
cd collect
python carla_collector.py \
    --host localhost \
    --port 2000 \
    --episodes 100 \
    --frames-per-episode 200 \
    --output ../data/raw
```

**采集配置**：
- 分辨率：256x256
- 采集场景：Town01-Town07（城市+高速）
- 天气：晴天、雨天、雾天
- 动作记录：转向、油门、刹车

**预期输出**：
- `data/raw/episode_XXXX/images/*.png`：RGB图像
- `data/raw/episode_XXXX/actions.npy`：动作序列
- `data/raw/episode_XXXX/metadata.json`：元数据

---

### 3. 数据预处理

```bash
cd utils
python process_data.py --input ../data/raw --output ../data/processed
```

---

### 4. 训练VQ-VAE

```bash
cd train
python train_vqvae.py \
    --data-path ../data/processed \
    --save-dir ../checkpoints/vqvae \
    --epochs 100 \
    --batch-size 64
```

**关键参数**：
- 图像尺寸：256x256
- Token网格：16x16
- 词表大小：1024
- 嵌入维度：256

---

### 5. 导出Token

```bash
cd utils
python export_tokens.py \
    --data-path ../data/processed \
    --vqvae-checkpoint ../checkpoints/vqvae/best.pth \
    --output ../data/tokens/tokens_actions.npz
```

---

### 6. 训练世界模型

```bash
cd train
python train_world_model.py \
    --token-path ../data/tokens/tokens_actions.npz \
    --save-dir ../checkpoints/world_model \
    --epochs 200 \
    --batch-size 64
```

**关键技术**：
- **FiLM动作调制**：动作通过FiLM层影响Transformer
- **课程学习**：逐步引入时间平滑正则化
- **动作自适应**：大动作允许更多变化

---

### 7. 生成视频

```bash
cd visualize
python dream.py \
    --vqvae-checkpoint ../checkpoints/vqvae/best.pth \
    --world-model-checkpoint ../checkpoints/world_model/best.pth \
    --output ../outputs/dream_result.mp4 \
    --num-frames 300
```

**控制方式**：
- 键盘实时控制（W/A/S/D）
- 从文件读取动作序列
- 回放数据集动作

---

## 🔧 核心技术

### VQ-VAE架构
```
Encoder: 256×256 → 128×128 → 64×64 → 32×32 → 16×16
         (64ch)    (128ch)   (256ch)   (256ch)   (256ch)
                              ↓
                        Vector Quantizer
                        (1024 embeddings)
                              ↓
Decoder: 16×16 → 32×32 → 64×64 → 128×128 → 256×256
```

### Transformer世界模型
```
Input: [4帧历史token (256个token/帧)] + [动作向量]
       ↓
Action Embedding → FiLM Modulation
       ↓
Transformer (8层, 8头, 512维)
       ↓
Output: 下一帧token分布 (256个token, 每个1024类)
```

### FiLM调制机制
```python
# 在每个Transformer层中
hidden = attention(hidden)
gamma, beta = film_layer(action_embedding)
hidden = gamma * hidden + beta  # 动作调制
hidden = feedforward(hidden)
```

---

## 📊 预期性能

| 指标 | 目标值 |
|------|--------|
| VQ-VAE重建PSNR | > 25 dB |
| Token利用率 | > 85% |
| 生成帧率 | 15-20 FPS (RTX 3090) |
| 时间连续性SSIM | > 0.92 |
| 动作响应延迟 | < 2帧 |

---

## 🎯 与MetaDrive项目的对比

| 特性 | MetaDrive | CARLA |
|------|-----------|-------|
| 场景复杂度 | 简单道路 | 城市+建筑 |
| 参考物密度 | 稀疏 | 丰富 |
| 转向视觉变化 | 微弱 | 明显 |
| 动作响应性 | 困难 | 自然 |
| 数据真实感 | 中 | 高 |

**关键优势**：CARLA城市场景中，转向时建筑物产生明显视差，模型更容易学到动作-视觉映射。

---

## 🐛 常见问题

### Q1: CARLA连接失败
```bash
# 检查CARLA服务器是否运行
ps aux | grep Carla

# 测试连接
python -c "import carla; client = carla.Client('localhost', 2000); print(client.get_server_version())"
```

### Q2: GPU内存不足
- 减小batch_size（64 → 32）
- 使用混合精度训练（默认已启用）
- 降低图像分辨率（256 → 128，需修改配置）

### Q3: 采集速度慢
- 关闭渲染窗口（`--no-rendering`）
- 使用同步模式（`--sync-mode`）
- 降低采集帧率

---

## 📝 配置说明

### 采集配置 (collect/config.py)
```python
# 图像尺寸
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

# 采集参数
EPISODES = 100
FRAMES_PER_EPISODE = 200
FPS = 20

# CARLA场景
TOWNS = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']
WEATHERS = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'ClearSunset']
```

### 训练配置 (train/config.py)
```python
# VQ-VAE
VQVAE_CONFIG = {
    'embed_dim': 256,
    'num_embeddings': 1024,
    'commitment_cost': 0.25,
    'lr': 2e-4,
    'epochs': 100,
}

# World Model
WM_CONFIG = {
    'hidden_dim': 512,
    'num_heads': 8,
    'num_layers': 8,
    'context_frames': 4,
    'lr': 3e-4,
    'epochs': 200,
    'temporal_smooth_weight': 0.0,  # 课程学习，逐步增加到0.02
}
```

---

## 📈 实验建议

### 对比实验
1. **消融实验**：
   - 有/无FiLM调制
   - 有/无时间平滑
   - 不同context_frames (1/2/4/8)

2. **数据对比**：
   - CARLA vs MetaDrive（转向响应性）
   - 不同天气条件

3. **模型对比**：
   - VQ-VAE vs VAE
   - Transformer vs LSTM

---

## 🔗 参考资源

- [CARLA Documentation](https://carla.readthedocs.io/)
- [CARLA Python API](https://carla.readthedocs.io/en/latest/python_api/)
- [VQ-VAE Paper](https://arxiv.org/abs/1711.00937)
- [World Models Paper](https://arxiv.org/abs/1803.10122)
- [GAIA-1](https://arxiv.org/abs/2309.17080)

---

## 📧 开发记录

**创建日期**：2026-01-11
**目的**：解决MetaDrive数据动作响应性不足的问题
**核心改进**：使用CARLA城市场景，增强动作-视觉映射的可学习性
