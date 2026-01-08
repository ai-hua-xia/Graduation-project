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
