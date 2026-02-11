# 项目结构说明

## 目录组织（当前）

### `bin/` - 执行脚本

| 脚本 | 功能 | 备注 |
|------|------|------|
| `model_tools.sh` | 统一工具入口 | 兼容旧模型路径自动选择 |
| `run_collect_10.sh` | 10 端口并行采集 | Phase A/B 采集 |
| `start_carla_server.sh` | 启动 CARLA 服务 | 服务器入口 |
| `setup_env.sh` | 环境检查 | conda/cuda 检查 |
| `activate.sh` | 快速激活环境 | 本地路径可按需改 |

### `collect/` - 数据采集

| 脚本 | 功能 |
|------|------|
| `collect_data_action_correlated.py` | 动作相关性采集主脚本 |
| `verify_data_action_focused.py` | 采集后质量验证 |
| `utils.py` | 采集辅助函数 |

### `train/` - 训练脚本

| 脚本 | 功能 |
|------|------|
| `train_vqvae_v2.py` | VQ-VAE v2 训练 |
| `train_vqvae_v3.py` | VQ-VAE v3 / f=8 主训练脚本 |
| `train_world_model.py` | World Model 训练（当前含 AdaLN-Zero + ActionAux + rollout） |
| `train_world_model_ss.py` | Scheduled Sampling 训练 |
| `config.py` | 全局训练配置 |

### `models/` - 模型定义

| 文件 | 功能 |
|------|------|
| `vqvae.py` | VQ-VAE 主体 |
| `world_model.py` | World Model 主体（条件注入与辅助头） |
| `film.py` | FiLM 与 AdaLN-Zero 层 |

### `evaluate/` - 评估

| 脚本 | 功能 |
|------|------|
| `evaluate_world_model.py` | 主评估脚本（PSNR/SSIM/LPIPS/FID/FVD + collapse 指标） |
| `metrics.py` | 指标实现 |
| `visualize_results.py` | 评估结果可视化 |

### `tools/` - 分析工具（当前）

| 工具 | 功能 |
|------|------|
| `analyze_ss_training.py` | SS 训练分析 |
| `analyze_video_quality.py` | 视频质量衰减分析 |

### `utils/` - 核心工具

| 工具 | 功能 |
|------|------|
| `dataset.py` | 数据加载/采样（含 episode 过滤、A/B 分层、rollout future targets） |
| `export_tokens_v2.py` | tokens 导出 |
| `generate_videos.py` | 预测视频生成 |
| `diagnose_model.py` | 模型诊断 |
| `generate_figures.py` | 图表生成 |
| `extract_loss_from_logs.py` | 日志提取 |
| `extract_vqvae_loss.py` | VQ-VAE 日志提取 |

## 核心数据路径

```text
data/
├── raw/                      # 基础采集
├── raw_action_corr_v2/       # 旧版动作相关采集
├── raw_action_corr_f8/       # 当前主线采集
├── tokens_raw/
├── tokens_action_corr_v2/
└── tokens_action_corr_f8/    # 当前主线 token
```

## 核心权重路径

```text
checkpoints/
├── vqvae/
│   ├── vqvae_v2/
│   ├── vqvae_action_corr/
│   ├── vqvae_action_corr_v2/
│   └── vqvae_action_corr_f8/          # 当前主线 VQ-VAE
├── wm/
│   ├── world_model/
│   ├── world_model_v2/
│   ├── world_model_v4/
│   ├── world_model_v5/
│   ├── world_model_f8/                # 旧版 f8 基线
│   └── world_model_f8_adaln_aux/      # 当前主线 WM
└── wm_ss/
```

## 日志目录（已重构）

```text
logs/
├── data_collect/
├── train_vqvae/
├── train_wm/
└── train_ss/
```

## 输出目录

```text
outputs/
├── evaluations/
├── videos/
├── analysis/
├── debug/
└── figures/
```

## 设计原则

- 主线优先：`f8 + AdaLN-Zero + ActionAux + rollout`
- 旧模型可回退：保留 `best.pth` 即可
- 日志分层：按训练类型分类归档

## 常用命令

```bash
# 查看 wm 主线训练趋势
rg -n "Epoch [0-9]+:|  Loss:|  CE:|  Contrast:|  ActionAux:|  Rollout:|Rollout Weight:" logs/train_wm/train_world_model_f8_adaln_aux.log | tail -n 160

# 快速评估
python evaluate/evaluate_world_model.py --help
```

---

最后更新：2026-02-11
