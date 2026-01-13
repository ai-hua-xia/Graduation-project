"""
从训练日志中提取loss数据
"""
import re
import numpy as np
from pathlib import Path

def extract_wm_loss(log_file):
    """从World Model日志中提取loss"""
    epochs = []
    ce_losses = []
    smooth_losses = []
    smooth_weights = []

    with open(log_file, 'r') as f:
        for line in f:
            # 匹配格式: Epoch 145 (smooth=0.0200): ... loss=0.1426, ce=0.0483, smooth=4.7146
            match = re.search(r'Epoch (\d+) \(smooth=([\d\.]+)\):.*?loss=([\d\.]+), ce=([\d\.]+), smooth=([\d\.]+)', line)
            if match:
                epoch = int(match.group(1))
                smooth_weight = float(match.group(2))
                ce_loss = float(match.group(4))
                smooth_loss = float(match.group(5))

                # 只保留每个epoch的最后一个值（最终的平均值）
                if not epochs or epochs[-1] != epoch:
                    epochs.append(epoch)
                    ce_losses.append(ce_loss)
                    smooth_losses.append(smooth_loss)
                    smooth_weights.append(smooth_weight)
                else:
                    # 更新当前epoch的值
                    ce_losses[-1] = ce_loss
                    smooth_losses[-1] = smooth_loss
                    smooth_weights[-1] = smooth_weight

    return {
        'epochs': np.array(epochs),
        'ce_loss': np.array(ce_losses),
        'smooth_loss': np.array(smooth_losses),
        'smooth_weight': np.array(smooth_weights)
    }


def extract_vqvae_loss(log_file):
    """从VQ-VAE日志中提取loss（如果有的话）"""
    # 先检查是否有VQ-VAE的训练日志
    if not Path(log_file).exists():
        print(f"VQ-VAE log not found: {log_file}")
        return None

    epochs = []
    recon_losses = []
    vq_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            # 尝试匹配VQ-VAE的loss格式
            # 格式可能是: Epoch X: recon_loss=Y, vq_loss=Z
            match = re.search(r'Epoch (\d+).*?recon.*?=([\d\.]+).*?vq.*?=([\d\.]+)', line, re.IGNORECASE)
            if match:
                epoch = int(match.group(1))
                recon = float(match.group(2))
                vq = float(match.group(3))

                if not epochs or epochs[-1] != epoch:
                    epochs.append(epoch)
                    recon_losses.append(recon)
                    vq_losses.append(vq)
                else:
                    recon_losses[-1] = recon
                    vq_losses[-1] = vq

    if not epochs:
        return None

    return {
        'epochs': np.array(epochs),
        'recon_loss': np.array(recon_losses),
        'vq_loss': np.array(vq_losses)
    }


if __name__ == '__main__':
    log_dir = Path('/home/llb/HunyuanWorld-Voyager/bishe/carla_project/logs')

    # 提取World Model loss
    print("Extracting World Model losses...")
    wm_log = log_dir / 'train_wm_v2.log'
    wm_data = extract_wm_loss(wm_log)
    print(f"  Found {len(wm_data['epochs'])} epochs")
    print(f"  Epoch range: {wm_data['epochs'][0]} - {wm_data['epochs'][-1]}")
    print(f"  CE loss range: {wm_data['ce_loss'].min():.4f} - {wm_data['ce_loss'].max():.4f}")
    print(f"  Smooth loss range: {wm_data['smooth_loss'].min():.4f} - {wm_data['smooth_loss'].max():.4f}")

    # 保存数据
    output_file = log_dir / 'extracted_losses.npz'
    np.savez(output_file, **wm_data)
    print(f"\nSaved to: {output_file}")

    # 尝试提取VQ-VAE loss
    print("\nLooking for VQ-VAE logs...")
    vqvae_logs = list(log_dir.glob('*vqvae*.log')) + list(log_dir.glob('*vq*.log'))
    if vqvae_logs:
        print(f"Found: {vqvae_logs}")
        for vq_log in vqvae_logs:
            vq_data = extract_vqvae_loss(vq_log)
            if vq_data:
                print(f"  Extracted {len(vq_data['epochs'])} epochs from {vq_log.name}")
    else:
        print("  No VQ-VAE logs found")
