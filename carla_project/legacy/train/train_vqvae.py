"""
VQ-VAE训练脚本
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import sys
from tqdm import tqdm

# 添加项目根目录到path（legacy脚本需要显式定位）
sys.path.append(str(Path(__file__).resolve().parents[2]))

from legacy.models.vqvae import VQVAE
from utils.dataset import get_vqvae_dataloader
from train.config import VQVAE_CONFIG, DATA_CONFIG


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, images in enumerate(pbar):
        images = images.to(device)

        # 归一化到[-1, 1]
        images = images * 2.0 - 1.0

        optimizer.zero_grad()

        # 混合精度训练
        if config['use_amp']:
            with autocast(dtype=torch.bfloat16 if config['amp_dtype'] == 'bf16' else torch.float16):
                recon, vq_loss, _ = model(images)
                recon_loss = nn.functional.mse_loss(recon, images)
                loss = recon_loss + vq_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recon, vq_loss, _ = model(images)
            recon_loss = nn.functional.mse_loss(recon, images)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

        # 统计
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()

        # 更新进度条
        if batch_idx % config['log_every'] == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'vq': f"{vq_loss.item():.4f}",
            })

    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon_loss / len(dataloader)
    avg_vq = total_vq_loss / len(dataloader)

    return avg_loss, avg_recon, avg_vq


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train VQ-VAE')
    parser.add_argument('--data-path', type=str, default='../data/raw',
                        help='Path to CARLA raw data')
    parser.add_argument('--save-dir', type=str, default='../checkpoints/vqvae',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # 配置
    config = VQVAE_CONFIG.copy()
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 数据加载
    print("\nLoading data...")
    dataloader = get_vqvae_dataloader(
        args.data_path,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    # 模型
    print("\nCreating model...")
    model = VQVAE(
        in_channels=config['in_channels'],
        base_channels=config['base_channels'],
        embed_dim=config['embed_dim'],
        num_embeddings=config['num_embeddings'],
        commitment_cost=config['commitment_cost'],
    ).to(device)

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 混合精度scaler
    scaler = GradScaler() if config['use_amp'] else None

    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # 训练
    print("\nStarting training...")
    best_loss = float('inf')

    for epoch in range(start_epoch, config['epochs']):
        avg_loss, avg_recon, avg_vq = train_epoch(
            model, dataloader, optimizer, scaler, device, epoch, config
        )

        print(f"\nEpoch {epoch}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Recon: {avg_recon:.4f}")
        print(f"  VQ: {avg_vq:.4f}")

        # 保存checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            save_path = save_dir / f"vqvae_epoch_{epoch:03d}.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, save_path)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = save_dir / "best.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, save_path)
            print(f"  New best model! Loss: {best_loss:.4f}")

    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
