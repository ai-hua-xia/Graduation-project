"""
World Model训练脚本（带课程学习）
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.world_model import WorldModel, compute_temporal_smoothness_loss
from utils.dataset import get_world_model_dataloader
from train.config import WM_CONFIG


def get_smooth_weight(epoch, config):
    """课程学习：逐步增加平滑权重"""
    if epoch < config['smooth_warmup_epochs']:
        progress = epoch / config['smooth_warmup_epochs']
        return config['smooth_weight_start'] + progress * (
            config['smooth_weight_end'] - config['smooth_weight_start']
        )
    return config['smooth_weight_end']


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_smooth_loss = 0

    # 当前epoch的平滑权重
    smooth_weight = get_smooth_weight(epoch, config)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (smooth={smooth_weight:.4f})")

    for batch_idx, batch in enumerate(pbar):
        context_tokens = batch['context_tokens'].to(device)  # (B, T, H, W)
        context_actions = batch['context_actions'].to(device)  # (B, T, action_dim)
        target_token = batch['target_token'].to(device)  # (B, H, W)

        optimizer.zero_grad()

        # 混合精度训练
        if config['use_amp']:
            with autocast(dtype=torch.bfloat16 if config['amp_dtype'] == 'bf16' else torch.float16):
                logits = model(context_tokens, context_actions)  # (B, tokens_per_frame, vocab)

                # 交叉熵损失
                B, T, V = logits.shape
                target_flat = target_token.view(B, -1)  # (B, tokens_per_frame)
                ce_loss = nn.functional.cross_entropy(
                    logits.view(B * T, V),
                    target_flat.view(B * T)
                )

                # 时间平滑损失（计算相邻样本）
                if smooth_weight > 0 and B > 1:
                    # 计算动作幅度
                    action_magnitudes = torch.norm(
                        context_actions[:, -1, :], dim=-1
                    )  # (B,)

                    # 扩展为序列形式用于计算平滑损失
                    logits_seq = logits.unsqueeze(1)  # (B, 1, T, V)
                    action_mag_seq = action_magnitudes[:-1].unsqueeze(-1)  # (B-1, 1)

                    # 简化版：只计算batch内相邻样本的平滑度
                    smooth_loss = torch.tensor(0.0, device=device)
                    for i in range(B - 1):
                        p = nn.functional.softmax(logits[i], dim=-1)
                        q = nn.functional.softmax(logits[i+1], dim=-1)
                        kl = nn.functional.kl_div(q.log(), p, reduction='batchmean')

                        weight = torch.exp(-config['beta'] * action_magnitudes[i])
                        smooth_loss += kl * weight

                    smooth_loss = smooth_loss / (B - 1)
                else:
                    smooth_loss = torch.tensor(0.0, device=device)

                # 总损失
                loss = config['ce_weight'] * ce_loss + smooth_weight * smooth_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(context_tokens, context_actions)

            B, T, V = logits.shape
            target_flat = target_token.view(B, -1)
            ce_loss = nn.functional.cross_entropy(
                logits.view(B * T, V),
                target_flat.view(B * T)
            )

            smooth_loss = torch.tensor(0.0, device=device)
            if smooth_weight > 0 and B > 1:
                action_magnitudes = torch.norm(context_actions[:, -1, :], dim=-1)
                for i in range(B - 1):
                    p = nn.functional.softmax(logits[i], dim=-1)
                    q = nn.functional.softmax(logits[i+1], dim=-1)
                    kl = nn.functional.kl_div(q.log(), p, reduction='batchmean')
                    weight = torch.exp(-config['beta'] * action_magnitudes[i])
                    smooth_loss += kl * weight
                smooth_loss = smooth_loss / (B - 1)

            loss = config['ce_weight'] * ce_loss + smooth_weight * smooth_loss

            loss.backward()
            optimizer.step()

        # 统计
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_smooth_loss += smooth_loss.item()

        # 更新进度条
        if batch_idx % config['log_every'] == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{ce_loss.item():.4f}",
                'smooth': f"{smooth_loss.item():.4f}",
            })

    avg_loss = total_loss / len(dataloader)
    avg_ce = total_ce_loss / len(dataloader)
    avg_smooth = total_smooth_loss / len(dataloader)

    return avg_loss, avg_ce, avg_smooth


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
    parser = argparse.ArgumentParser(description='Train World Model')
    parser.add_argument('--token-path', type=str, required=True,
                        help='Path to tokens file')
    parser.add_argument('--save-dir', type=str, default='../checkpoints/world_model',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # 配置
    config = WM_CONFIG.copy()
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
    dataloader = get_world_model_dataloader(
        args.token_path,
        batch_size=config['batch_size'],
        context_frames=config['context_frames'],
        num_workers=config['num_workers'],
    )

    # 模型
    print("\nCreating model...")
    model = WorldModel(
        num_embeddings=config['num_embeddings'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        context_frames=config['context_frames'],
        action_dim=config['action_dim'],
        tokens_per_frame=config['tokens_per_frame'],
        dropout=config['dropout'],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 混合精度
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
    print(f"Curriculum learning: smooth weight {config['smooth_weight_start']} -> {config['smooth_weight_end']} over {config['smooth_warmup_epochs']} epochs")

    best_loss = float('inf')

    for epoch in range(start_epoch, config['epochs']):
        avg_loss, avg_ce, avg_smooth = train_epoch(
            model, dataloader, optimizer, scaler, device, epoch, config
        )

        print(f"\nEpoch {epoch}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  CE: {avg_ce:.4f}")
        print(f"  Smooth: {avg_smooth:.4f}")
        print(f"  Smooth Weight: {get_smooth_weight(epoch, config):.4f}")

        # 保存checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            save_path = save_dir / f"world_model_epoch_{epoch:03d}.pth"
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
