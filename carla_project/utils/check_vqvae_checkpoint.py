"""
从VQ-VAE checkpoint中提取loss历史
"""
import torch
from pathlib import Path
import numpy as np

checkpoint_dir = Path('/home/llb/HunyuanWorld-Voyager/bishe/carla_project/checkpoints/vqvae_v2')

# 读取best.pth看看里面有什么
best_ckpt = torch.load(checkpoint_dir / 'best.pth', map_location='cpu')

print("Keys in checkpoint:")
for key in best_ckpt.keys():
    print(f"  {key}: {type(best_ckpt[key])}")

# 检查是否有loss历史
if 'loss_history' in best_ckpt:
    print("\nFound loss_history!")
    print(f"  Type: {type(best_ckpt['loss_history'])}")
    print(f"  Length: {len(best_ckpt['loss_history'])}")
elif 'losses' in best_ckpt:
    print("\nFound losses!")
    print(f"  Type: {type(best_ckpt['losses'])}")
