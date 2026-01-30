"""
VQ-VAE V2训练脚本

改进点:
1. 使用EMA更新的VectorQuantizer
2. 定期重置死码
3. 监控codebook使用情况
4. 可选的感知损失
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import sys
from tqdm import tqdm

# 添加父目录到path
sys.path.append(str(Path(__file__).parent.parent))

from models.vqvae_v2 import VQVAE_V2
from utils.dataset import get_vqvae_dataloader

try:
    import lpips  # Optional perceptual loss
except ImportError:  # pragma: no cover - optional dependency
    lpips = None


# V2配置
VQVAE_V2_CONFIG = {
    # 模型参数
    'in_channels': 3,
    'base_channels': 128,  # 64 -> 128
    'embed_dim': 256,
    'num_embeddings': 1024,
    'commitment_cost': 0.25,
    'ema_decay': 0.99,

    # 训练参数
    'lr': 1e-4,  # 稍微降低学习率
    'epochs': 100,
    'batch_size': 32,  # 更大模型需要更小batch
    'num_workers': 8,

    # 死码重置
    'reset_dead_codes_every': 100,  # 每100个batch重置一次

    # 混合精度
    'use_amp': True,
    'amp_dtype': 'bf16',

    # 重建损失权重
    'recon_mse_weight': 1.0,
    'recon_l1_weight': 0.5,
    'recon_lpips_weight': 0.1,
    'lpips_net': 'alex',

    # 保存
    'save_every': 5,
    'log_every': 50,
}


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, config, lpips_model=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_l1_loss = 0
    total_lpips_loss = 0
    total_perplexity = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, images in enumerate(pbar):
        images = images.to(device)

        # 归一化到[-1, 1]
        images = images * 2.0 - 1.0

        optimizer.zero_grad()

        # 混合精度训练
        if config['use_amp']:
            with autocast(dtype=torch.bfloat16 if config['amp_dtype'] == 'bf16' else torch.float16):
                recon, vq_loss, indices, perplexity = model(images)
                mse_loss = nn.functional.mse_loss(recon, images)
                l1_loss = nn.functional.l1_loss(recon, images)
                lpips_loss = torch.tensor(0.0, device=device)
                if lpips_model is not None:
                    lpips_loss = lpips_model(recon, images).mean()
                recon_loss = (
                    config['recon_mse_weight'] * mse_loss
                    + config['recon_l1_weight'] * l1_loss
                    + config['recon_lpips_weight'] * lpips_loss
                )
                loss = recon_loss + vq_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recon, vq_loss, indices, perplexity = model(images)
            mse_loss = nn.functional.mse_loss(recon, images)
            l1_loss = nn.functional.l1_loss(recon, images)
            lpips_loss = torch.tensor(0.0, device=device)
            if lpips_model is not None:
                lpips_loss = lpips_model(recon, images).mean()
            recon_loss = (
                config['recon_mse_weight'] * mse_loss
                + config['recon_l1_weight'] * l1_loss
                + config['recon_lpips_weight'] * lpips_loss
            )
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

        # 定期重置死码
        if (batch_idx + 1) % config['reset_dead_codes_every'] == 0:
            with torch.no_grad():
                z = model.encoder(images)
                num_reset = model.quantizer.reset_dead_codes(z)
                if num_reset > 0:
                    tqdm.write(f"  Reset {num_reset} dead codes")

        # 统计
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_l1_loss += l1_loss.item()
        total_lpips_loss += lpips_loss.item()
        total_perplexity += perplexity.item()

        # 更新进度条
        if batch_idx % config['log_every'] == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'vq': f"{vq_loss.item():.4f}",
                'l1': f"{l1_loss.item():.4f}",
                'lpips': f"{lpips_loss.item():.4f}",
                'ppl': f"{perplexity.item():.1f}",
            })

    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_recon = total_recon_loss / n_batches
    avg_vq = total_vq_loss / n_batches
    avg_l1 = total_l1_loss / n_batches
    avg_lpips = total_lpips_loss / n_batches
    avg_perplexity = total_perplexity / n_batches

    return avg_loss, avg_recon, avg_vq, avg_l1, avg_lpips, avg_perplexity


def save_checkpoint(model, optimizer, epoch, loss, perplexity, codebook_usage, save_path):
    """保存checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'perplexity': perplexity,
        'codebook_usage': codebook_usage,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train VQ-VAE V2')
    parser.add_argument('--data-path', type=str, default='../data/raw',
                        help='Path to CARLA raw data')
    parser.add_argument('--save-dir', type=str, default='../checkpoints/vqvae_v2',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--recon-mse-weight', type=float, default=None,
                        help='Weight for MSE reconstruction loss')
    parser.add_argument('--recon-l1-weight', type=float, default=None,
                        help='Weight for L1 reconstruction loss')
    parser.add_argument('--recon-lpips-weight', type=float, default=None,
                        help='Weight for LPIPS perceptual loss')
    parser.add_argument('--lpips-net', type=str, default=None,
                        help='LPIPS backbone (alex/vgg/squeeze)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # 配置
    config = VQVAE_V2_CONFIG.copy()
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.recon_mse_weight is not None:
        config['recon_mse_weight'] = args.recon_mse_weight
    if args.recon_l1_weight is not None:
        config['recon_l1_weight'] = args.recon_l1_weight
    if args.recon_lpips_weight is not None:
        config['recon_lpips_weight'] = args.recon_lpips_weight
    if args.lpips_net is not None:
        config['lpips_net'] = args.lpips_net

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
    print(f"Dataset size: {len(dataloader.dataset)} images")
    print(f"Batches per epoch: {len(dataloader)}")

    # 模型
    print("\nCreating VQ-VAE V2 model...")
    model = VQVAE_V2(
        in_channels=config['in_channels'],
        base_channels=config['base_channels'],
        embed_dim=config['embed_dim'],
        num_embeddings=config['num_embeddings'],
        commitment_cost=config['commitment_cost'],
        ema_decay=config['ema_decay'],
    ).to(device)

    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )

    # 混合精度scaler
    scaler = GradScaler() if config['use_amp'] else None

    lpips_model = None
    if config['recon_lpips_weight'] > 0:
        if lpips is None:
            print("Warning: LPIPS not installed; recon_lpips_weight will be ignored.")
            config['recon_lpips_weight'] = 0.0
        else:
            lpips_model = lpips.LPIPS(net=config['lpips_net']).to(device)
            lpips_model.eval()
            for p in lpips_model.parameters():
                p.requires_grad_(False)

    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # 恢复scheduler状态
        for _ in range(start_epoch):
            scheduler.step()

    # 训练
    print("\n" + "="*60)
    print("Starting VQ-VAE V2 training")
    print("="*60)
    print(f"Config: base_channels={config['base_channels']}, "
          f"embed_dim={config['embed_dim']}, "
          f"num_embeddings={config['num_embeddings']}")
    print(f"EMA decay: {config['ema_decay']}")
    print(f"Recon weights: mse={config['recon_mse_weight']}, "
          f"l1={config['recon_l1_weight']}, "
          f"lpips={config['recon_lpips_weight']}")
    print(f"Dead code reset every {config['reset_dead_codes_every']} batches")
    print("="*60 + "\n")

    best_loss = float('inf')

    for epoch in range(start_epoch, config['epochs']):
        avg_loss, avg_recon, avg_vq, avg_l1, avg_lpips, avg_perplexity = train_epoch(
            model, dataloader, optimizer, scaler, device, epoch, config, lpips_model=lpips_model
        )

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 获取codebook使用统计
        codebook_usage = model.get_codebook_usage()

        print(f"\nEpoch {epoch}:")
        print(
            f"  Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}, "
            f"L1: {avg_l1:.4f}, LPIPS: {avg_lpips:.4f})"
        )
        print(f"  Perplexity: {avg_perplexity:.1f}")
        print(f"  Codebook: {codebook_usage['used_codes']}/{codebook_usage['total_codes']} "
              f"({codebook_usage['usage_ratio']*100:.1f}% used)")
        print(f"  LR: {current_lr:.2e}")

        # 保存checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            save_path = save_dir / f"vqvae_v2_epoch_{epoch:03d}.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, avg_perplexity,
                          codebook_usage, save_path)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = save_dir / "best.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, avg_perplexity,
                          codebook_usage, save_path)
            print(f"  New best model! Loss: {best_loss:.4f}")

        # 每个epoch结束时重置使用计数（用于下一个epoch的统计）
        model.quantizer.usage_count.zero_()

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
