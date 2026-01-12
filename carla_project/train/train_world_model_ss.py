"""
World Model训练脚本（带Scheduled Sampling）

Scheduled Sampling: 训练时逐步用模型自己的预测替代真实token，
让模型学会处理自己的预测误差，减少训练-推理不匹配问题。
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.world_model import WorldModel
from utils.dataset import get_world_model_sequence_dataloader
from train.config import WM_CONFIG


def get_sampling_prob(epoch, total_epochs, schedule='linear', k=0.5):
    """
    计算使用预测token的概率（Scheduled Sampling）

    Args:
        epoch: 当前epoch
        total_epochs: 总epoch数
        schedule: 调度策略 ('linear', 'exponential', 'inverse_sigmoid')
        k: 调度参数

    Returns:
        使用预测token的概率 (0到1)
    """
    progress = epoch / total_epochs

    if schedule == 'linear':
        # 线性增长：从0到k
        return min(k, progress * k * 2)
    elif schedule == 'exponential':
        # 指数衰减：teacher forcing概率从1衰减
        return 1 - k ** epoch
    elif schedule == 'inverse_sigmoid':
        # 逆sigmoid：更平滑的过渡
        return k / (k + np.exp((1 - progress * 2) * 5))
    else:
        return progress * k


def train_epoch_with_ss(model, dataloader, optimizer, scaler, device, epoch, config, sampling_prob):
    """
    带Scheduled Sampling的训练

    数据格式: 每个batch包含连续的多帧序列

    优化：使用gradient accumulation减少显存占用
    """
    model.train()
    total_loss = 0
    total_ce_loss = 0
    num_batches = 0

    # Gradient accumulation steps
    accum_steps = 4

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (ss_prob={sampling_prob:.3f})")

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        # batch包含连续序列
        tokens_seq = batch['tokens'].to(device)  # (B, seq_len, H, W)
        actions_seq = batch['actions'].to(device)  # (B, seq_len, action_dim)

        B, seq_len, H, W = tokens_seq.shape
        context_frames = config['context_frames']

        if seq_len <= context_frames:
            continue

        # 初始化token buffer（使用真实的前context_frames帧）
        token_buffer = tokens_seq[:, :context_frames].clone()  # (B, context_frames, H, W)

        batch_loss = torch.tensor(0.0, device=device)
        batch_ce_loss = 0
        num_predictions = 0

        # 自回归预测序列中的每一帧
        for t in range(context_frames, seq_len):
            # 目标token
            target_token = tokens_seq[:, t]  # (B, H, W)

            # 动作窗口
            action_window = actions_seq[:, t-context_frames:t]  # (B, context_frames, action_dim)

            if config['use_amp']:
                with autocast(dtype=torch.bfloat16 if config['amp_dtype'] == 'bf16' else torch.float16):
                    # 前向传播
                    logits = model(token_buffer, action_window)  # (B, tokens_per_frame, vocab)

                    # 计算损失
                    B_curr, T, V = logits.shape
                    target_flat = target_token.reshape(B_curr, -1)
                    ce_loss = nn.functional.cross_entropy(
                        logits.reshape(B_curr * T, V),
                        target_flat.reshape(B_curr * T)
                    )

                    # 累加损失（除以序列长度和accumulation steps）
                    scaled_loss = ce_loss / (seq_len - context_frames) / accum_steps
                    batch_loss = batch_loss + scaled_loss
                    batch_ce_loss += ce_loss.item()
                    num_predictions += 1

            # Scheduled Sampling: 决定下一步用真实token还是预测token
            with torch.no_grad():
                if np.random.random() < sampling_prob:
                    # 使用模型预测的token（需要重新计算，因为上面的logits在计算图中）
                    with autocast(dtype=torch.bfloat16 if config['amp_dtype'] == 'bf16' else torch.float16):
                        logits_sample = model(token_buffer, action_window)
                    pred_probs = nn.functional.softmax(logits_sample, dim=-1)
                    pred_tokens = torch.argmax(pred_probs, dim=-1)  # (B, tokens_per_frame)
                    pred_tokens = pred_tokens.reshape(B_curr, H, W)
                    next_token = pred_tokens
                else:
                    # 使用真实token
                    next_token = target_token

            # 更新token buffer（detach以避免计算图过长）
            token_buffer = torch.cat([
                token_buffer[:, 1:],
                next_token.unsqueeze(1).detach()
            ], dim=1)

        # 反向传播（每accum_steps步或最后一步）
        if num_predictions > 0:
            if config['use_amp']:
                scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                if config['use_amp']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += batch_loss.item() * accum_steps
            total_ce_loss += batch_ce_loss / num_predictions
            num_batches += 1

        if batch_idx % config['log_every'] == 0:
            pbar.set_postfix({
                'loss': f"{batch_ce_loss / max(num_predictions, 1):.4f}",
            })

    # 处理最后不足accum_steps的梯度
    if (batch_idx + 1) % accum_steps != 0:
        if config['use_amp']:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(num_batches, 1)
    avg_ce = total_ce_loss / max(num_batches, 1)

    return avg_loss, avg_ce


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
    parser = argparse.ArgumentParser(description='Train World Model with Scheduled Sampling')
    parser.add_argument('--token-path', type=str, required=True,
                        help='Path to tokens file')
    parser.add_argument('--save-dir', type=str, default='../checkpoints/world_model_ss',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--seq-len', type=int, default=16,
                        help='Sequence length for training')
    parser.add_argument('--ss-schedule', type=str, default='linear',
                        choices=['linear', 'exponential', 'inverse_sigmoid'],
                        help='Scheduled sampling schedule')
    parser.add_argument('--ss-k', type=float, default=0.5,
                        help='Max sampling probability')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Load pretrained model (without optimizer state)')
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

    # 数据加载（需要序列数据）
    print("\nLoading data...")
    dataloader = get_world_model_sequence_dataloader(
        args.token_path,
        batch_size=config['batch_size'],
        seq_len=args.seq_len,
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
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)

    # 混合精度
    scaler = GradScaler() if config['use_amp'] else None

    # 加载预训练模型
    start_epoch = 0
    if args.pretrained:
        print(f"\nLoading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained model loaded (starting fresh training with SS)")

    # 恢复训练
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # 训练
    print("\nStarting training with Scheduled Sampling...")
    print(f"Schedule: {args.ss_schedule}, max_prob: {args.ss_k}")

    best_loss = float('inf')

    for epoch in range(start_epoch, config['epochs']):
        # 计算当前的sampling概率
        sampling_prob = get_sampling_prob(
            epoch, config['epochs'],
            schedule=args.ss_schedule,
            k=args.ss_k
        )

        avg_loss, avg_ce = train_epoch_with_ss(
            model, dataloader, optimizer, scaler, device, epoch, config, sampling_prob
        )

        print(f"\nEpoch {epoch}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  CE: {avg_ce:.4f}")
        print(f"  Sampling Prob: {sampling_prob:.4f}")

        # 保存checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            save_path = save_dir / f"world_model_ss_epoch_{epoch:03d}.pth"
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
