# Graduation-project
本科毕设_世界模型（MetaDrive + VQ-VAE + Transformer + Adapter）

## 项目概览
目标：用动作条件的世界模型，在 MetaDrive 场景里进行实时“想象”。  
核心思路：把画面压缩成离散 token → 用 Transformer 预测未来 token → 解码成图像。

## 整体流程（从数据到视频）
1) 数据采集  
`collect_data.py` 采集图像与动作  
输出：`dataset_v2_complex/images/*.png` + `dataset_v2_complex/actions.npy`

2) 训练 VQ-VAE（视觉压缩）  
`train_vqvae_256.py`  
输出：`checkpoints_vqvae_256/vqvae_256_epXX.pth`

3) 导出 token 数据集（对齐动作）  
`export_vqvae_tokens.py`  
输出：`dataset_v2_complex/tokens_actions_vqvae_16x16.npz`  
包含：`tokens / actions / indices`，按文件名索引对齐，避免缺帧错位。

4) 训练世界模型（动作条件）  
`train_world_model.py`  
输入：tokens + actions  
输出：`checkpoints_world_model/world_model_epXX.pth`

5) 生成想象视频（Dream）  
`visualize_dream.py`  
输入：起始帧 token + 未来动作序列  
输出：`dream_result.avi`

## 快速开始（示例命令）
```bash
# 1) 采集数据
python collect_data.py

# 2) 训练 VQ-VAE
python train_vqvae_256.py

# 3) 导出 token
python export_vqvae_tokens.py --checkpoint checkpoints_vqvae_256/vqvae_256_epXX.pth

# 4) 训练世界模型
python train_world_model.py

# 5) 生成 Dream 视频
python visualize_dream.py
```

## 关键配置
- 图像分辨率：256x256
- VQ-VAE token 网格：16x16（每帧 256 token）
- 动作维度：2（转向、油门/刹车）

## 目录结构
```
bishe/
  collect_data.py
  train_vqvae_256.py
  export_vqvae_tokens.py
  train_world_model.py
  visualize_dream.py
  dataset_v2_complex/
    images/
    actions.npy
    tokens_actions_vqvae_16x16.npz
    tokens_actions_vqvae_16x16.json
  checkpoints_vqvae_256/
  checkpoints_world_model/
```

## 说明
- 生成视频的动作序列来自真实采集，不是随机动作。
- `tokens_actions_vqvae_16x16.npz` 通过 `indices` 做对齐，避免缺帧导致动作错位。

1) 增加实验与指标（最快体现贡献）

  - 加 2–3 个对比：
      - 无动作条件 vs 有动作条件
      - VAE vs VQ‑VAE
      - 无 Adapter vs 有 Adapter
  - 增加定量指标：
      - 重建：PSNR/SSIM/LPIPS
      - 预测：token CE、1‑step vs 5‑step 误差
      - 语义 Adapter（如果做）：mIoU
  - 这部分能写进论文“实验/消融”，非常有分量。

  2) 多步 rollout 训练（让世界模型更“像世界模型”）

  - 现在是 1‑step teacher forcing。
  - 可以加K‑step rollout loss 或 scheduled sampling，强化长时稳定性。
  - 结果会明显减少“场景跳变”。

  3) Adapter 强化（让你更像“有创新点”）

  - 做两个 Adapter：
      - 语义 Adapter（MetaDrive 语义相机监督）
      - 风格 Adapter（夜晚/雾）
  - 答辩展示时：切换按钮 + 视觉对比，效果很“学术”。

  4) 训练/推理可视化与系统化

  - 自动保存 rollout 对比图、loss 曲线、FPS 统计
  - 输出日志表格（简化论文写作）
  - 这能体现“工程和实验完整性”

  5) 模型结构小升级（代码量更多）

  - 换成 分层 Transformer（先时间后空间）
  - 或加入 Action‑FiLM / Action‑CrossAttention
  - 这能写成“结构改进贡献点”