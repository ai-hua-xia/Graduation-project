# CARLA World Model

基于 CARLA 模拟器的动作条件世界模型项目，使用 VQ-VAE v2 和 Transformer 学习从动作预测未来视觉帧。

## 🎯 项目简介

本项目实现了一个端到端的世界模型系统，能够根据驾驶动作预测未来的视觉场景，并支持生成对比视频与 WASD 动作序列的“梦境”视频。

**核心技术**：
- **VQ-VAE v2/v3**: 256×256 图像压缩为 16×16 离散 tokens（f=16，默认）；可选 f=8（32×32）
- **Transformer World Model**: 基于历史帧与动作预测下一帧
- **Scheduled Sampling**: 缓解自回归生成误差累积
- **WASD 控制**: 支持文本文件输入动作序列

## 📊 当前状态

| 模块 | 现状 | 说明 |
|------|------|------|
| 数据集 | ✅ 已就绪 | `data/raw`、`data/raw_action_corr_v2`、`data/raw_action_corr_v3` |
| Tokens | ✅ 已就绪 | `data/tokens_action_corr_v2/tokens_actions.npz`、`data/tokens_raw/tokens_actions.npz`（可选：`data/tokens_action_corr_f8/tokens_actions.npz`） |
| VQ-VAE | ✅ v3 checkpoint | `checkpoints/vqvae/vqvae_action_corr_v2/best.pth`（兼容 `vqvae_v2`；可选：`checkpoints/vqvae/vqvae_action_corr_f8/best.pth`） |
| World Model | ✅ v5 系列 | `checkpoints/wm/world_model_v5`、`checkpoints/wm_ss/world_model_v5_ss`、`checkpoints/wm_ss/world_model_v5_ss_fast` |
| Scheduled Sampling | ✅ 有可用权重 | `checkpoints/wm_ss/world_model_v5_ss`、`checkpoints/wm_ss/world_model_v5_ss_fast`、`checkpoints/wm_ss/world_model_v4_ss_e029` |
| 工具脚本 | ✅ 已统一 | `bin/model_tools.sh` + `bin/run_collect_10.sh`（10 端口并行采集） |

**自动选择规则（model_tools.sh）**：
- Token 文件：优先 `data/tokens_action_corr_v2/tokens_actions.npz`，否则使用 `data/tokens_raw/tokens_actions.npz`
- VQ-VAE：优先 `checkpoints/vqvae/vqvae_action_corr_v2/best.pth`，否则回退到 `checkpoints/vqvae/vqvae_v2/best.pth`
- World Model checkpoint：`wm_ss/world_model_v5_ss_fast` → `wm_ss/world_model_v5_ss` → `wm/world_model_v5` → `wm_ss/world_model_v4_ss_e029` → `wm/world_model_v4` → `wm/world_model_v3` → `wm_ss/world_model_v2_ss`

> 注：f=8（32×32 tokens）需要手动指定 `vqvae_action_corr_f8` 与 `tokens_action_corr_f8`，不会自动选择。

## 🚀 快速开始

### 使用统一工具脚本

```bash
# 查看训练进度摘要
./bin/model_tools.sh status

# 快速评估模型
./bin/model_tools.sh eval

# 生成30帧对比视频（随机场景）
./bin/model_tools.sh video 30

# 固定起点生成视频（可按数据集实际情况调整 start_idx）
./bin/model_tools.sh video 100 1990

# 生成纯预测视频（不显示 GT）
./bin/model_tools.sh video 100 1990 --pred-only

# 使用 WASD 动作文件生成梦境视频
./bin/model_tools.sh dream actions.txt --show-controls
```

### 直接使用 Python 脚本

#### 方式1: 生成预测视频
```bash
python utils/generate_videos.py \
    --vqvae-checkpoint checkpoints/vqvae/vqvae_action_corr_v2/best.pth \
    --world-model-checkpoint checkpoints/wm_ss/world_model_v5_ss_fast/best.pth \
    --token-file data/tokens_action_corr_v2/tokens_actions.npz \
    --output-dir outputs/videos \
    --num-videos 1 \
    --num-frames 150 \
    --fps 10 \
    --temperature 1.0 \
    --prediction-only
```

#### 方式2: 评估模型
```bash
python evaluate/evaluate_world_model.py \
    --vqvae-checkpoint checkpoints/vqvae/vqvae_action_corr_v2/best.pth \
    --world-model-checkpoint checkpoints/wm_ss/world_model_v5_ss_fast/best.pth \
    --token-file data/tokens_action_corr_v2/tokens_actions.npz \
    --output outputs/evaluations/eval.json \
    --num-samples 100 \
    --num-sequences 10 \
    --sequence-length 50 \
    --device cuda
```

#### 可选：训练 f=8 VQ-VAE（更高分辨率 token）
```bash
python train/train_vqvae_v3.py \
    --data-path data/raw_action_corr_v3 \
    --save-dir checkpoints/vqvae/vqvae_action_corr_f8 \
    --downsample-factor 8 \
    --batch-size 32

python utils/export_tokens_v2.py \
    --data-path data/raw_action_corr_v3 \
    --vqvae-checkpoint checkpoints/vqvae/vqvae_action_corr_f8/best.pth \
    --output data/tokens_action_corr_f8/tokens_actions.npz
```

## 🎮 WASD 键盘控制

支持 7 个按键控制驾驶（WASD/QE/N），映射逻辑在 `visualize/dream.py` 中定义。

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

**生成命令**：
```bash
./bin/model_tools.sh dream actions.txt --show-controls
```

## 📁 项目结构

```
carla_project/
├── bin/                   # 🔧 脚本入口
│   ├── model_tools.sh
│   ├── setup_env.sh
│   ├── activate.sh
│   ├── start_carla_server.sh
│   └── test_wasd.sh
├── collect/               # 数据采集
├── train/                 # 训练脚本
├── evaluate/              # 评估脚本
├── visualize/             # 可视化/梦境生成
├── tools/                 # 分析工具
├── utils/                 # 核心库
├── outputs/               # 输出结果
│   ├── evaluations/
│   ├── videos/
│   ├── analysis/
│   └── figures/
├── checkpoints/           # 已训练模型
│   ├── vqvae/
│   │   ├── vqvae_v2/
│   │   ├── vqvae_action_corr_v2/
│   │   └── vqvae_action_corr_f8/   # 可选 f=8
│   ├── wm/
│   │   ├── world_model/
│   │   ├── world_model_v5/
│   │   └── world_model_v4/
│   └── wm_ss/
│       ├── world_model_v5_ss/
│       ├── world_model_v5_ss_fast/
│       └── world_model_v4_ss_e029/
├── data/                  # 数据与 tokens
│   ├── raw/
│   ├── raw_action_corr_v2/
│   ├── raw_action_corr_v3/
│   ├── tokens_raw/
│   ├── tokens_action_corr_v2/
│   └── tokens_action_corr_f8/  # 可选 f=8
└── docs/                  # 📚 文档
```

## 🗃️ Legacy 脚本

历史脚本已归档到 `legacy/`（保留实验记录，不再作为主流程使用）。

## 🧠 模型与数据配置

- **VQ-VAE v2**: codebook 1024 × 256，256×256 → 16×16 tokens（可选 f=8 → 32×32；见 `train/train_vqvae_v2.py` / `train/train_vqvae_v3.py`）
- **World Model**: A-XL 规模（32 层、18 heads、context=4，详见 `train/config.py`）
- **数据集**: `data/raw` 为基础采集，`data/raw_action_corr_v3` 为动作相关性采集版本

## 🔧 常用命令

```bash
# 启动 CARLA 服务器
./bin/start_carla_server.sh

# 查看训练状态
./bin/model_tools.sh status

# 快速评估
./bin/model_tools.sh eval

# 生成视频
./bin/model_tools.sh video 30

# WASD 梦境生成
./bin/model_tools.sh dream actions.txt
```

## 📚 文档

- **[快速开始](QUICKSTART.md)**
- **[项目结构](PROJECT_STRUCTURE.md)**
- **[CARLA 服务器安装](INSTALL_SERVER.md)**
- **[变更日志](CHANGELOG.md)**
- **[开题报告](开题报告.md)**

## 🛠️ 技术栈

- **深度学习**: PyTorch 2.x（详见 `requirements_carla.txt`）
- **模拟器**: CARLA 0.9.16 服务器（Python API 版本需与服务器一致）
- **评估指标**: PSNR, SSIM, LPIPS
- **可视化**: OpenCV, Matplotlib, FFmpeg

## 💡 使用建议

- 使用与训练数据分布一致的动作范围，WASD 映射默认约为 steering ±0.4、throttle 0.42-0.65
- 长序列生成更容易累计误差，优先尝试 Scheduled Sampling 权重
- 通过 `--temperature` 控制采样多样性（0 为贪心）

## 📝 引用

如果使用本项目，请引用：
```
[待补充]
```

## 📄 许可

[待补充]

---

**最后更新**: 2026-01-16
