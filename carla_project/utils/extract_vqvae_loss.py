"""
从所有VQ-VAE checkpoint中提取loss值
"""
import torch
from pathlib import Path
import numpy as np
import re

checkpoint_dir = Path('/home/llb/HunyuanWorld-Voyager/bishe/carla_project/checkpoints/vqvae_v2')

# 收集所有epoch checkpoint
checkpoints = sorted(checkpoint_dir.glob('vqvae_v2_epoch_*.pth'))

epochs = []
losses = []
perplexities = []

print(f"Found {len(checkpoints)} checkpoints")

for ckpt_path in checkpoints:
    # 从文件名提取epoch
    match = re.search(r'epoch_(\d+)', ckpt_path.name)
    if match:
        epoch = int(match.group(1))

        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            loss = ckpt.get('loss', None)
            perplexity = ckpt.get('perplexity', None)

            if loss is not None:
                epochs.append(epoch)
                losses.append(loss)
                if perplexity is not None:
                    perplexities.append(perplexity)
                else:
                    perplexities.append(0)

        except Exception as e:
            print(f"Error loading {ckpt_path.name}: {e}")

# 排序
sorted_indices = np.argsort(epochs)
epochs = np.array(epochs)[sorted_indices]
losses = np.array(losses)[sorted_indices]
perplexities = np.array(perplexities)[sorted_indices]

print(f"\nExtracted {len(epochs)} epochs")
print(f"Epoch range: {epochs[0]} - {epochs[-1]}")
print(f"Loss range: {losses.min():.6f} - {losses.max():.6f}")
print(f"Perplexity range: {perplexities.min():.2f} - {perplexities.max():.2f}")

# 保存
output_file = checkpoint_dir / 'vqvae_loss_history.npz'
np.savez(output_file,
         epochs=epochs,
         total_loss=losses,
         perplexity=perplexities)

print(f"\nSaved to: {output_file}")
