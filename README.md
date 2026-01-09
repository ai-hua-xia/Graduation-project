# Graduation-project
本科毕设：世界模型（MetaDrive + VQ‑VAE + Transformer + Adapter）

## 项目概览
目标：用动作条件的世界模型在 MetaDrive 场景里实时“想象”未来画面。
核心思路：VQ‑VAE 压缩成离散 token → Transformer 预测未来 token → 解码成图像；
风格由 Adapter/Decoder 控制（夜晚/雾/雪）。

## 流程（从数据到视频）
1) 数据采集
- 脚本：`collect_data.py`
- 输出：`dataset_v2_complex/images/*.png` + `dataset_v2_complex/actions.npy`

2) 训练 VQ‑VAE（视觉压缩）
- 脚本：`train/train_vqvae_256.py`
- 输出：`checkpoints_vqvae_256/vqvae_256_epXX.pth`

3) 导出 token + action（对齐）
- 脚本：`export_vqvae_tokens.py`
- 输出：`dataset_v2_complex/tokens_actions_vqvae_16x16.npz`
- 说明：包含 `tokens/actions/indices`，用 `indices` 保证与动作序列对齐

4) 训练世界模型（动作条件）
- 脚本：`train/train_world_model.py`
- 输出：`checkpoints_world_model/world_model_epXX.pth`

5) 训练风格 Adapter/Decoder（可选）
- 夜晚/雾：`train/train_adapter.py`
- 雪：`train/train_snow_decoder_noise.py` / `train/train_snow_decoder_noise_hr.py`
- 输出：`checkpoints_adapter/...`

6) 生成 Dream 视频
- 标准：`utils/visualize_dream.py`
- 雪：`utils/visualize_dream_snow_noise.py` / `utils/visualize_dream_snow_noise_hr.py`
- 输出：`dream_result.mp4`（H.264）

## 快速开始（示例命令）
```bash
# 1) 采集数据
python collect_data.py

# 2) 训练 VQ-VAE
python train/train_vqvae_256.py

# 3) 导出 token
python export_vqvae_tokens.py --checkpoint checkpoints_vqvae_256/vqvae_256_epXX.pth

# 4) 训练世界模型
python train/train_world_model.py

# 5) 生成 Dream 视频
python utils/visualize_dream.py
```

## 动作控制方式（生成视频时）
优先级：动作文件 > 键盘 > 数据集动作
- 数据集动作：默认读取 `dataset_v2_complex/actions.npy`
- 键盘输入：`USE_KEYBOARD = True`，`KEYBOARD_BACKEND = "terminal"`，W/A/S/D + Space + Q
- 动作文件：在根目录写 `action.txt`，每行一个动作（`w/a/s/d/space`）
  - 配置：`USE_ACTION_FILE = True`，`ACTIONS_FILE_PATH = "action.txt"`

## Web 展示
```bash
cd web
python -m http.server 8000
```
打开浏览器访问 `http://localhost:8000`，先上传 reality + night/fog/snow 三个视频，然后用按钮切换风格。

## 目录结构
```
bishe/
  collect_data.py
  export_vqvae_tokens.py
  train/
    train_vqvae_256.py
    train_world_model.py
    train_adapter.py
    train_snow_decoder_noise.py
    train_snow_decoder_noise_hr.py
  utils/
    visualize_dream.py
    visualize_dream_snow_noise.py
    visualize_dream_snow_noise_hr.py
  dataset_v2_complex/
    images/
    actions.npy
    tokens_actions_vqvae_16x16.npz
    tokens_actions_vqvae_16x16.json
  dataset_style/
  checkpoints_vqvae_256/
  checkpoints_world_model/
  checkpoints_adapter/
  web/
```

## 不重训也能做的工作（对毕设有帮助）
1) 实验与图表（只用现有结果）
- 对比图：VAE vs VQ‑VAE 重建、无风格 vs 有风格、不同风格效果
- 时间稳定性：计算相邻帧 SSIM/LPIPS 曲线，展示抖动程度
- 统计表：FPS、分辨率、参数量、生成时延

2) 可视化与演示包装
- 网页 Demo（已完成）：上传三种风格视频，一键切换
- 录制对比视频：左现实、右想象，中间风格切换
- 固化实验配置表（epoch/temperature/top‑k 等）

3) 代码与工程化完善
- 增加命令行参数（统一配置入口）
- 输出日志与自动截图（便于论文写作）
- 增加异常提示（比如动作文件为空/视频编码不兼容）

## 说明
- 生成视频动作默认来自真实采集，不是随机动作。
- `tokens_actions_vqvae_16x16.npz` 通过 `indices` 对齐，避免缺帧导致动作错位。
