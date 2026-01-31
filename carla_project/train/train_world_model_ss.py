"""
World Model训练脚本（带Scheduled Sampling）

Scheduled Sampling: 训练时逐步用模型自己的预测替代真实token，
让模型学会处理自己的预测误差，减少训练-推理不匹配问题。
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
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


def should_apply_action_contrast(prob, device):
    if prob >= 1.0:
        return True
    if prob <= 0.0:
        return False
    flag = torch.rand(1, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(flag, 0)
    return flag.item() < prob


def perturb_actions(actions, steer_std, throttle_std):
    noise = torch.randn_like(actions)
    noise[..., 0] *= steer_std
    noise[..., 1] *= throttle_std
    perturbed = actions + noise
    perturbed[..., 0] = torch.clamp(perturbed[..., 0], -1.0, 1.0)
    perturbed[..., 1] = torch.clamp(perturbed[..., 1], 0.0, 1.0)
    return perturbed


def build_contrast_actions(context_actions, config):
    contrast_type = config.get('action_contrast_type', 'noise')
    if contrast_type == 'swap' and context_actions.size(0) > 1:
        batch_size = context_actions.size(0)
        shift = int(torch.randint(1, batch_size, (1,), device=context_actions.device).item())
        perm = (torch.arange(batch_size, device=context_actions.device) + shift) % batch_size
        return context_actions[perm]
    return perturb_actions(
        context_actions,
        config.get('action_noise_std_steer', 0.3),
        config.get('action_noise_std_throttle', 0.2),
    )


def compute_action_contrast_loss(logits, logits_pert, margin, mode):
    log_p = torch.log_softmax(logits, dim=-1)
    log_q = torch.log_softmax(logits_pert, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    kl_pq = nn.functional.kl_div(log_p, q, reduction='batchmean')
    kl_qp = nn.functional.kl_div(log_q, p, reduction='batchmean')
    divergence = 0.5 * (kl_pq + kl_qp)
    if mode == 'inverse':
        return 1.0 / (1.0 + divergence)
    return nn.functional.relu(margin - divergence)


def train_epoch_with_ss(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch,
    config,
    sampling_prob,
    is_main,
    distributed,
    tbptt_steps=1,
):
    """
    带Scheduled Sampling的训练

    数据格式: 每个batch包含连续的多帧序列

    优化：分段反向传播，减少显存占用
    """
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_contrast_loss = 0
    num_batches = 0

    # Gradient accumulation steps
    accum_steps = 4

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (ss_prob={sampling_prob:.3f})", disable=not is_main)

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
        memory_state = None

        batch_loss_value = 0.0
        batch_ce_loss = 0
        num_predictions = 0
        batch_contrast_loss = 0.0
        use_contrast = (
            config.get('action_contrast_weight', 0.0) > 0
            and should_apply_action_contrast(config.get('action_contrast_prob', 1.0), device)
        )
        use_memory = getattr(model, 'use_memory', False)

        sync_context = contextlib.nullcontext()
        if distributed and (batch_idx + 1) % accum_steps != 0:
            sync_context = model.no_sync()
        with sync_context:
            step_loss_accum = None
            step_count = 0

            # 自回归预测序列中的每一帧
            for t in range(context_frames, seq_len):
                # 目标token
                target_token = tokens_seq[:, t]  # (B, H, W)

                # 动作窗口
                action_window = actions_seq[:, t-context_frames:t]  # (B, context_frames, action_dim)

                memory_input = memory_state

                if config['use_amp']:
                    with autocast(dtype=torch.bfloat16 if config['amp_dtype'] == 'bf16' else torch.float16):
                        # 前向传播
                        if use_memory:
                            logits, memory_next = model(
                                token_buffer, action_window, memory=memory_input, return_memory=True
                            )
                        else:
                            logits = model(token_buffer, action_window)  # (B, tokens_per_frame, vocab)
                            memory_next = None

                        # 计算损失
                        B_curr, T, V = logits.shape
                        target_flat = target_token.reshape(B_curr, -1)
                        ce_loss = nn.functional.cross_entropy(
                            logits.reshape(B_curr * T, V),
                            target_flat.reshape(B_curr * T)
                        )
                else:
                    if use_memory:
                        logits, memory_next = model(
                            token_buffer, action_window, memory=memory_input, return_memory=True
                        )
                    else:
                        logits = model(token_buffer, action_window)
                        memory_next = None
                    B_curr, T, V = logits.shape
                    target_flat = target_token.reshape(B_curr, -1)
                    ce_loss = nn.functional.cross_entropy(
                        logits.reshape(B_curr * T, V),
                        target_flat.reshape(B_curr * T)
                    )

                step_loss = ce_loss / (seq_len - context_frames)
                batch_ce_loss += ce_loss.item()
                num_predictions += 1

                # 动作对比正则（仅在第一步做一次，降低开销）
                if use_contrast and t == context_frames:
                    perturbed_actions = build_contrast_actions(action_window, config)
                    if config['use_amp']:
                        with autocast(dtype=torch.bfloat16 if config['amp_dtype'] == 'bf16' else torch.float16):
                            if use_memory:
                                logits_pert = model(token_buffer, perturbed_actions, memory=memory_input)
                            else:
                                logits_pert = model(token_buffer, perturbed_actions)
                    else:
                        if use_memory:
                            logits_pert = model(token_buffer, perturbed_actions, memory=memory_input)
                        else:
                            logits_pert = model(token_buffer, perturbed_actions)

                    contrast_term = compute_action_contrast_loss(
                        logits,
                        logits_pert,
                        config.get('action_contrast_margin', 0.05),
                        config.get('action_contrast_mode', 'hinge'),
                    )
                    step_loss = step_loss + (
                        config.get('action_contrast_weight', 0.0) * contrast_term
                        / (seq_len - context_frames)
                    )
                    batch_contrast_loss += contrast_term.item()

                batch_loss_value += step_loss.item()
                if step_loss_accum is None:
                    step_loss_accum = step_loss
                else:
                    step_loss_accum = step_loss_accum + step_loss
                step_count += 1

                if step_count >= tbptt_steps or t == seq_len - 1:
                    chunk_loss = step_loss_accum / accum_steps
                    if config['use_amp']:
                        scaler.scale(chunk_loss).backward()
                    else:
                        chunk_loss.backward()
                    step_loss_accum = None
                    step_count = 0

                # Scheduled Sampling: 决定下一步用真实token还是预测token
                with torch.no_grad():
                    if np.random.random() < sampling_prob:
                        pred_tokens = torch.argmax(logits.detach(), dim=-1)  # (B, tokens_per_frame)
                        pred_tokens = pred_tokens.reshape(B_curr, H, W)
                        next_token = pred_tokens
                    else:
                        next_token = target_token

                # 更新token buffer（detach以避免计算图过长）
                token_buffer = torch.cat([
                    token_buffer[:, 1:],
                    next_token.unsqueeze(1).detach()
                ], dim=1)
                if use_memory:
                    memory_state = memory_next.detach()

        # 反向传播（每accum_steps步或最后一步）
        if num_predictions > 0:

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

            total_loss += batch_loss_value
            total_ce_loss += batch_ce_loss / num_predictions
            total_contrast_loss += batch_contrast_loss
            num_batches += 1

        if is_main and batch_idx % config['log_every'] == 0:
            pbar.set_postfix({
                'loss': f"{batch_ce_loss / max(num_predictions, 1):.4f}",
                'contrast': f"{batch_contrast_loss:.4f}",
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
    avg_contrast = total_contrast_loss / max(num_batches, 1)

    return avg_loss, avg_ce, avg_contrast


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        local_rank = args.local_rank
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        distributed = True
    else:
        local_rank = 0
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        distributed = False
    return distributed, local_rank, world_size, device


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(value, device):
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.item()


def main():
    parser = argparse.ArgumentParser(description='Train World Model with Scheduled Sampling')
    parser.add_argument('--token-path', type=str, required=True,
                        help='Path to tokens file')
    parser.add_argument('--save-dir', type=str, default='../checkpoints/wm_ss/world_model_v2_ss',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--seq-len', type=int, default=16,
                        help='Sequence length for training')
    parser.add_argument('--tbptt-steps', type=int, default=1,
                        help='Backward every N steps inside a sequence')
    parser.add_argument('--ss-schedule', type=str, default='linear',
                        choices=['linear', 'exponential', 'inverse_sigmoid'],
                        help='Scheduled sampling schedule')
    parser.add_argument('--ss-k', type=float, default=0.7,
                        help='Max sampling probability')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Load pretrained model (without optimizer state)')
    parser.add_argument('--allow-missing-keys', action='store_true',
                        help='Allow missing/unexpected keys when loading pretrained model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--local-rank', type=int, default=None, dest='local_rank',
                        help='Local rank for distributed training')
    parser.add_argument('--local_rank', type=int, default=None, dest='local_rank',
                        help='Local rank for distributed training')

    args = parser.parse_args()

    # 配置
    config = WM_CONFIG.copy()
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    # 设备/分布式
    distributed, local_rank, world_size, device = setup_distributed(args)
    rank = dist.get_rank() if distributed else 0
    if is_main_process():
        print(f"Using device: {device}")

    # 创建保存目录
    save_dir = Path(args.save_dir)
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    # 数据加载（需要序列数据）
    if is_main_process():
        print("\nLoading data...")
    if distributed:
        dataloader, sampler = get_world_model_sequence_dataloader(
            args.token_path,
            batch_size=config['batch_size'],
            seq_len=args.seq_len,
            num_workers=config['num_workers'],
            distributed=True,
            rank=rank,
            world_size=world_size,
            return_sampler=True,
        )
    else:
        dataloader = get_world_model_sequence_dataloader(
            args.token_path,
            batch_size=config['batch_size'],
            seq_len=args.seq_len,
            num_workers=config['num_workers'],
        )
        sampler = None

    # 自适应num_embeddings（与tokens文件一致）
    config['num_embeddings'] = int(dataloader.dataset.tokens.max()) + 1
    if is_main_process():
        print(f"Num embeddings: {config['num_embeddings']}")

    # 模型
    if is_main_process():
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
        use_memory=config.get('use_memory', False),
        memory_dim=config.get('memory_dim', 256),
        dropout=config['dropout'],
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main_process():
        num_params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
        print(f"Model parameters: {num_params / 1e6:.2f}M")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)

    # 混合精度
    scaler = GradScaler() if config['use_amp'] else None

    # 加载预训练模型
    start_epoch = 0
    if args.pretrained:
        if is_main_process():
            print(f"\nLoading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
        incompatible = unwrap_model(model).load_state_dict(
            checkpoint['model_state_dict'],
            strict=not args.allow_missing_keys,
        )
        if args.allow_missing_keys and is_main_process():
            print(f"Missing keys: {len(incompatible.missing_keys)}")
            print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
        if is_main_process():
            print("Pretrained model loaded (starting fresh training with SS)")

    # 恢复训练
    if args.resume:
        if is_main_process():
            print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # 训练
    if is_main_process():
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

        if sampler is not None:
            sampler.set_epoch(epoch)

        avg_loss, avg_ce, avg_contrast = train_epoch_with_ss(
            model,
            dataloader,
            optimizer,
            scaler,
            device,
            epoch,
            config,
            sampling_prob,
            is_main_process(),
            distributed,
            tbptt_steps=max(1, args.tbptt_steps),
        )

        avg_loss = reduce_mean(avg_loss, device)
        avg_ce = reduce_mean(avg_ce, device)
        avg_contrast = reduce_mean(avg_contrast, device)

        if is_main_process():
            print(f"\nEpoch {epoch}:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  CE: {avg_ce:.4f}")
            print(f"  Contrast: {avg_contrast:.4f}")
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

    if is_main_process():
        print("\nTraining complete!")
        print(f"Best loss: {best_loss:.4f}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
