"""
VQ-VAE V3 训练脚本（更大容量 + 细节增强损失）
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 添加父目录到path
sys.path.append(str(Path(__file__).parent.parent))

from models.vqvae_v2 import VQVAE_V2
from utils.dataset import get_vqvae_dataloader

try:
    import lpips  # Optional perceptual loss
except ImportError:  # pragma: no cover - optional dependency
    lpips = None


VQVAE_V3_CONFIG = {
    # 模型参数（更大容量）
    'in_channels': 3,
    'base_channels': 256,
    'embed_dim': 512,
    'num_embeddings': 4096,
    'commitment_cost': 0.25,
    'ema_decay': 0.99,
    'downsample_factor': 16,

    # 训练参数
    'lr': 1e-4,
    'epochs': 100,
    'batch_size': 32,
    'num_workers': 8,
    'reset_dead_codes_every': 20,

    # 混合精度
    'use_amp': True,
    'amp_dtype': 'bf16',

    # 重建损失权重
    'recon_mse_weight': 1.0,
    'recon_l1_weight': 0.5,
    'recon_lpips_weight': 0.1,
    'lpips_net': 'alex',
    'lpips_warmup_epochs': 10,

    # 细节增强
    'multiscale_l1_weight': 0.2,
    'multiscale_scales': (2, 4),
    'edge_weight': 0.1,

    # 保存
    'save_every': 5,
    'log_every': 50,
}


def _sobel_kernels(device, dtype, channels):
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=dtype)
    ky = torch.tensor([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], device=device, dtype=dtype)
    kx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return kx, ky


def edge_loss(recon, target):
    channels = recon.shape[1]
    kx, ky = _sobel_kernels(recon.device, recon.dtype, channels)
    grad_rx = F.conv2d(recon, kx, padding=1, groups=channels)
    grad_ry = F.conv2d(recon, ky, padding=1, groups=channels)
    grad_tx = F.conv2d(target, kx, padding=1, groups=channels)
    grad_ty = F.conv2d(target, ky, padding=1, groups=channels)
    grad_r = torch.sqrt(grad_rx ** 2 + grad_ry ** 2 + 1e-6)
    grad_t = torch.sqrt(grad_tx ** 2 + grad_ty ** 2 + 1e-6)
    return F.l1_loss(grad_r, grad_t)


def multiscale_l1(recon, target, scales):
    if not scales:
        return torch.tensor(0.0, device=recon.device)
    loss = 0.0
    for s in scales:
        r = F.avg_pool2d(recon, s)
        t = F.avg_pool2d(target, s)
        loss = loss + F.l1_loss(r, t)
    return loss / len(scales)


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, config, lpips_model=None):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_l1_loss = 0
    total_lpips_loss = 0
    total_ms_loss = 0
    total_edge_loss = 0
    total_perplexity = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    if config['lpips_warmup_epochs'] > 0:
        lpips_scale = min(1.0, (epoch + 1) / config['lpips_warmup_epochs'])
    else:
        lpips_scale = 1.0
    lpips_weight = config['recon_lpips_weight'] * lpips_scale

    for batch_idx, images in enumerate(pbar):
        images = images.to(device)
        images = images * 2.0 - 1.0

        optimizer.zero_grad()
        if config['use_amp']:
            with autocast(dtype=torch.bfloat16 if config['amp_dtype'] == 'bf16' else torch.float16):
                recon, vq_loss, _, perplexity = model(images)
                mse_loss = F.mse_loss(recon, images)
                l1_loss = F.l1_loss(recon, images)
                lpips_loss = torch.tensor(0.0, device=device)
                if lpips_model is not None and lpips_weight > 0:
                    lpips_loss = lpips_model(recon, images).mean()
                ms_loss = multiscale_l1(recon, images, config['multiscale_scales'])
                ed_loss = edge_loss(recon, images) if config['edge_weight'] > 0 else torch.tensor(0.0, device=device)

                recon_loss = (
                    config['recon_mse_weight'] * mse_loss
                    + config['recon_l1_weight'] * l1_loss
                    + lpips_weight * lpips_loss
                    + config['multiscale_l1_weight'] * ms_loss
                    + config['edge_weight'] * ed_loss
                )
                loss = recon_loss + vq_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recon, vq_loss, _, perplexity = model(images)
            mse_loss = F.mse_loss(recon, images)
            l1_loss = F.l1_loss(recon, images)
            lpips_loss = torch.tensor(0.0, device=device)
            if lpips_model is not None and lpips_weight > 0:
                lpips_loss = lpips_model(recon, images).mean()
            ms_loss = multiscale_l1(recon, images, config['multiscale_scales'])
            ed_loss = edge_loss(recon, images) if config['edge_weight'] > 0 else torch.tensor(0.0, device=device)

            recon_loss = (
                config['recon_mse_weight'] * mse_loss
                + config['recon_l1_weight'] * l1_loss
                + lpips_weight * lpips_loss
                + config['multiscale_l1_weight'] * ms_loss
                + config['edge_weight'] * ed_loss
            )
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

        if (batch_idx + 1) % config['reset_dead_codes_every'] == 0:
            with torch.no_grad():
                z = model.encoder(images)
                num_reset = model.quantizer.reset_dead_codes(z)
                if num_reset > 0:
                    tqdm.write(f"  Reset {num_reset} dead codes")

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_l1_loss += l1_loss.item()
        total_lpips_loss += lpips_loss.item()
        total_ms_loss += ms_loss.item()
        total_edge_loss += ed_loss.item()
        total_perplexity += perplexity.item()

        if batch_idx % config['log_every'] == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'vq': f"{vq_loss.item():.4f}",
                'l1': f"{l1_loss.item():.4f}",
                'lpips': f"{lpips_loss.item():.4f}",
                'ms': f"{ms_loss.item():.4f}",
                'edge': f"{ed_loss.item():.4f}",
                'ppl': f"{perplexity.item():.1f}",
            })

    n_batches = len(dataloader)
    return (
        total_loss / n_batches,
        total_recon_loss / n_batches,
        total_vq_loss / n_batches,
        total_l1_loss / n_batches,
        total_lpips_loss / n_batches,
        total_ms_loss / n_batches,
        total_edge_loss / n_batches,
        total_perplexity / n_batches,
    )


def save_checkpoint(model, optimizer, epoch, loss, perplexity, codebook_usage, save_path, downsample_factor):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'perplexity': perplexity,
        'codebook_usage': codebook_usage,
        'downsample_factor': downsample_factor,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train VQ-VAE V3')
    parser.add_argument('--data-path', type=str, default='../data/raw_action_corr',
                        help='Path to CARLA raw data')
    parser.add_argument('--save-dir', type=str, default='../checkpoints/vqvae_action_corr_v2',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader workers (default: from config)')
    parser.add_argument('--recon-mse-weight', type=float, default=None,
                        help='Weight for MSE reconstruction loss')
    parser.add_argument('--recon-l1-weight', type=float, default=None,
                        help='Weight for L1 reconstruction loss')
    parser.add_argument('--recon-lpips-weight', type=float, default=None,
                        help='Weight for LPIPS perceptual loss')
    parser.add_argument('--lpips-net', type=str, default=None,
                        help='LPIPS backbone (alex/vgg/squeeze)')
    parser.add_argument('--lpips-warmup-epochs', type=int, default=None,
                        help='Warmup epochs for LPIPS weight')
    parser.add_argument('--multiscale-l1-weight', type=float, default=None,
                        help='Weight for multiscale L1 loss')
    parser.add_argument('--edge-weight', type=float, default=None,
                        help='Weight for edge loss')
    parser.add_argument('--downsample-factor', type=int, default=None,
                        help='Downsample factor for VQ-VAE (8 or 16)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    config = VQVAE_V3_CONFIG.copy()
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    if args.recon_mse_weight is not None:
        config['recon_mse_weight'] = args.recon_mse_weight
    if args.recon_l1_weight is not None:
        config['recon_l1_weight'] = args.recon_l1_weight
    if args.recon_lpips_weight is not None:
        config['recon_lpips_weight'] = args.recon_lpips_weight
    if args.lpips_net is not None:
        config['lpips_net'] = args.lpips_net
    if args.lpips_warmup_epochs is not None:
        config['lpips_warmup_epochs'] = args.lpips_warmup_epochs
    if args.multiscale_l1_weight is not None:
        config['multiscale_l1_weight'] = args.multiscale_l1_weight
    if args.edge_weight is not None:
        config['edge_weight'] = args.edge_weight
    if args.downsample_factor is not None:
        config['downsample_factor'] = args.downsample_factor

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    dataloader = get_vqvae_dataloader(
        args.data_path,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    print(f"Dataset size: {len(dataloader.dataset)} images")
    print(f"Batches per epoch: {len(dataloader)}")

    print("\nCreating VQ-VAE model...")
    model = VQVAE_V2(
        in_channels=config['in_channels'],
        base_channels=config['base_channels'],
        embed_dim=config['embed_dim'],
        num_embeddings=config['num_embeddings'],
        commitment_cost=config['commitment_cost'],
        ema_decay=config['ema_decay'],
        downsample_factor=config['downsample_factor'],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )
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

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        ckpt_downsample = checkpoint.get('downsample_factor')
        if ckpt_downsample is not None and ckpt_downsample != config['downsample_factor']:
            raise ValueError(
                f"downsample_factor mismatch: checkpoint={ckpt_downsample} vs config={config['downsample_factor']}"
            )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}")

    print("\nTraining configuration:")
    print(f"  epochs={config['epochs']}, batch_size={config['batch_size']}, lr={config['lr']}")
    print(f"  multiscale_l1_weight={config['multiscale_l1_weight']}, edge_weight={config['edge_weight']}")
    print(f"  lpips_weight={config['recon_lpips_weight']} (warmup {config['lpips_warmup_epochs']} epochs)")
    print("=" * 60 + "\n")

    best_loss = float('inf')

    for epoch in range(start_epoch, config['epochs']):
        (
            avg_loss,
            avg_recon,
            avg_vq,
            avg_l1,
            avg_lpips,
            avg_ms,
            avg_edge,
            avg_perplexity,
        ) = train_epoch(model, dataloader, optimizer, scaler, device, epoch, config, lpips_model=lpips_model)

        scheduler.step()

        codebook_usage = model.get_codebook_usage()
        current_lr = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch}:")
        print(
            f"  Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}, "
            f"L1: {avg_l1:.4f}, LPIPS: {avg_lpips:.4f}, MS: {avg_ms:.4f}, Edge: {avg_edge:.4f})"
        )
        print(f"  Perplexity: {avg_perplexity:.1f}")
        print(f"  Codebook: {codebook_usage['used_codes']}/{codebook_usage['total_codes']} "
              f"({codebook_usage['usage_ratio']*100:.1f}% used)")
        print(f"  LR: {current_lr:.2e}")

        if (epoch + 1) % config['save_every'] == 0:
            save_path = save_dir / f"vqvae_v3_epoch_{epoch:03d}.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, avg_perplexity,
                            codebook_usage, save_path, config['downsample_factor'])

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = save_dir / "best.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, avg_perplexity,
                            codebook_usage, save_path, config['downsample_factor'])
            print(f"  New best model! Loss: {best_loss:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
