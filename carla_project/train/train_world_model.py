"""
World Model训练脚本（带课程学习）
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
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


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, config, is_main):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_smooth_loss = 0
    total_contrast_loss = 0

    # 当前epoch的平滑权重
    smooth_weight = get_smooth_weight(epoch, config)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (smooth={smooth_weight:.4f})", disable=not is_main)

    for batch_idx, batch in enumerate(pbar):
        context_tokens = batch['context_tokens'].to(device)  # (B, T, H, W)
        context_actions = batch['context_actions'].to(device)  # (B, T, action_dim)
        target_token = batch['target_token'].to(device)  # (B, H, W)

        optimizer.zero_grad()

        use_contrast = (
            config.get('action_contrast_weight', 0.0) > 0
            and should_apply_action_contrast(config.get('action_contrast_prob', 1.0), device)
        )

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

                contrast_loss = torch.tensor(0.0, device=device)
                if use_contrast:
                    perturbed_actions = build_contrast_actions(context_actions, config)
                    logits_pert = model(context_tokens, perturbed_actions)
                    contrast_loss = compute_action_contrast_loss(
                        logits,
                        logits_pert,
                        config.get('action_contrast_margin', 0.05),
                        config.get('action_contrast_mode', 'hinge'),
                    )

                # 总损失
                loss = (
                    config['ce_weight'] * ce_loss
                    + smooth_weight * smooth_loss
                    + config.get('action_contrast_weight', 0.0) * contrast_loss
                )

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

            contrast_loss = torch.tensor(0.0, device=device)
            if use_contrast:
                perturbed_actions = build_contrast_actions(context_actions, config)
                logits_pert = model(context_tokens, perturbed_actions)
                contrast_loss = compute_action_contrast_loss(
                    logits,
                    logits_pert,
                    config.get('action_contrast_margin', 0.05),
                    config.get('action_contrast_mode', 'hinge'),
                )

            loss = (
                config['ce_weight'] * ce_loss
                + smooth_weight * smooth_loss
                + config.get('action_contrast_weight', 0.0) * contrast_loss
            )

            loss.backward()
            optimizer.step()

        # 统计
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_smooth_loss += smooth_loss.item()
        total_contrast_loss += contrast_loss.item()

        # 更新进度条
        if is_main and batch_idx % config['log_every'] == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{ce_loss.item():.4f}",
                'smooth': f"{smooth_loss.item():.4f}",
                'contrast': f"{contrast_loss.item():.4f}",
            })

    avg_loss = total_loss / len(dataloader)
    avg_ce = total_ce_loss / len(dataloader)
    avg_smooth = total_smooth_loss / len(dataloader)
    avg_contrast = total_contrast_loss / len(dataloader)

    return avg_loss, avg_ce, avg_smooth, avg_contrast


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

    # 数据加载
    if is_main_process():
        print("\nLoading data...")
    if distributed:
        dataloader, sampler = get_world_model_dataloader(
            args.token_path,
            batch_size=config['batch_size'],
            context_frames=config['context_frames'],
            num_workers=config['num_workers'],
            distributed=True,
            rank=rank,
            world_size=world_size,
            return_sampler=True,
        )
    else:
        dataloader = get_world_model_dataloader(
            args.token_path,
            batch_size=config['batch_size'],
            context_frames=config['context_frames'],
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

    # 加载预训练模型（仅权重）
    if args.pretrained:
        if is_main_process():
            print(f"\nLoading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        incompatible = unwrap_model(model).load_state_dict(
            checkpoint['model_state_dict'],
            strict=not args.allow_missing_keys,
        )
        if args.allow_missing_keys and is_main_process():
            print(f"Missing keys: {len(incompatible.missing_keys)}")
            print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 混合精度
    scaler = GradScaler() if config['use_amp'] else None

    # 恢复训练
    start_epoch = 0
    if args.resume:
        if is_main_process():
            print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # 训练
    if is_main_process():
        print("\nStarting training...")
        print(f"Curriculum learning: smooth weight {config['smooth_weight_start']} -> {config['smooth_weight_end']} over {config['smooth_warmup_epochs']} epochs")

    best_loss = float('inf')

    for epoch in range(start_epoch, config['epochs']):
        if sampler is not None:
            sampler.set_epoch(epoch)
        avg_loss, avg_ce, avg_smooth, avg_contrast = train_epoch(
            model, dataloader, optimizer, scaler, device, epoch, config, is_main_process()
        )

        avg_loss = reduce_mean(avg_loss, device)
        avg_ce = reduce_mean(avg_ce, device)
        avg_smooth = reduce_mean(avg_smooth, device)
        avg_contrast = reduce_mean(avg_contrast, device)

        if is_main_process():
            print(f"\nEpoch {epoch}:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  CE: {avg_ce:.4f}")
            print(f"  Smooth: {avg_smooth:.4f}")
            print(f"  Contrast: {avg_contrast:.4f}")
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

    if is_main_process():
        print("\nTraining complete!")
        print(f"Best loss: {best_loss:.4f}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
