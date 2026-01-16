# 项目结构说明

## 📂 目录组织

### `bin/` - 可执行脚本（5个）

| 脚本 | 功能 | 备注 |
|------|------|------|
| **model_tools.sh** | 🌟 统一工具入口 | 推荐主入口，提供 status/eval/video/dream/diagnose/analyze/figures |
| start_carla_server.sh | 启动 CARLA 服务器 | 默认使用 `~/CARLA_0.9.16`，参考 `INSTALL_SERVER.md` |
| setup_env.sh | 环境检查与依赖提示 | 依赖本机 conda 环境 `voyager` |
| activate.sh | 快速进入工作环境 | 含硬编码路径，可按需修改 |
| test_wasd.sh | WASD 测试脚本 | 旧路径写法，建议使用 `model_tools.sh dream` |

### `collect/` - 数据采集

| 脚本 | 功能 |
|------|------|
| collect_data.py | 基础采集（Town03，固定参数） |
| collect_data_action_correlated.py | 动作相关性采集（可配置） |
| verify_data_action_focused.py | 采集质量验证 |
| utils.py | 采集辅助函数 |

### `train/` - 训练脚本

| 脚本 | 功能 |
|------|------|
| train_vqvae_v2.py | 训练 VQ-VAE v2 |
| train_world_model.py | 训练 World Model（Teacher Forcing） |
| train_world_model_ss.py | 训练 Scheduled Sampling 版本 |
| train_vqvae.py | 旧版本 VQ-VAE（保留） |

### `evaluate/` - 评估

| 脚本 | 功能 |
|------|------|
| evaluate_world_model.py | 评估主脚本 |
| metrics.py | 指标实现 |
| visualize_results.py | 评估结果可视化 |

### `visualize/` - 可视化/梦境生成

| 脚本 | 功能 |
|------|------|
| dream.py | WASD 动作序列生成视频 |
| compare_video.py | 视频对比工具 |

### `tools/` - 分析工具（4个）

| 工具 | 功能 |
|------|------|
| analyze_action_data.py | 动作分布分析 |
| analyze_ss_training.py | SS 训练分析 |
| analyze_video_quality.py | 视频质量衰减分析 |
| training_roadmap.py | 训练路线图/记录 |

### `utils/` - 核心库

| 工具 | 功能 |
|------|------|
| dataset.py | 数据加载与采样 |
| diagnose_model.py | 模型诊断 |
| export_tokens_v2.py | 导出 VQ-VAE tokens |
| generate_videos.py | 生成预测视频 |
| generate_figures.py | 生成论文图表 |
| extract_loss_from_logs.py | 训练日志解析 |
| extract_vqvae_loss.py | VQ-VAE 损失提取 |

### `data/` - 数据

```
data/
├── raw/                 # 基础采集数据
├── raw_action_corr_v1/  # 动作相关性 v1
├── raw_action_corr_v2/  # 动作相关性 v2
├── tokens_v2/           # tokens_actions.npz
└── tokens_v3/           # tokens_actions.npz
```

### `checkpoints/` - 模型权重

```
checkpoints/
├── vqvae_v2/
├── world_model_v2/
├── world_model_v2_ss/
├── world_model_v3/
├── world_model_v4/
└── world_model_v4_ss_e029/
```

### `outputs/` - 输出目录

```
outputs/
├── evaluations/    # 评估结果 (.json)
├── videos/         # 生成视频 (.mp4)
├── analysis/       # 分析图表 (.png)
└── figures/        # 论文图表 (.png)
```

### `logs/` - 训练日志

- 训练输出集中在 `logs/`（例如 `train_wm_v4.log`、`train_wm_v4_ss_e029.log`）

## 🎯 设计原则

- **统一入口**：日常使用优先 `./bin/model_tools.sh`
- **层次清晰**：bin（入口）→ tools/utils（工具库）→ train/evaluate/visualize（业务脚本）
- **输出集中**：所有可视化与评估产物都写入 `outputs/`

## 📋 常用命令速查

```bash
# 启动 CARLA
./bin/start_carla_server.sh

# 查看训练状态
./bin/model_tools.sh status

# 快速评估
./bin/model_tools.sh eval

# 生成视频
./bin/model_tools.sh video 30

# WASD 梦境
./bin/model_tools.sh dream actions.txt
```

## 📚 文档

- **[README.md](README.md)** - 项目主文档
- **[QUICKSTART.md](QUICKSTART.md)** - 快速开始
- **[INSTALL_SERVER.md](INSTALL_SERVER.md)** - CARLA 服务器安装
- **[CHANGELOG.md](CHANGELOG.md)** - 变更日志

---

**最后更新**: 2026-01-16
