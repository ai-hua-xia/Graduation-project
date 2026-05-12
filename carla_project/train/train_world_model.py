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


def get_action_inject_scale(epoch, config):
    start = config.get('action_inject_scale_start', 1.0)
    end = config.get('action_inject_scale_end', 1.0)
    start_epoch = config.get('action_inject_start_epoch', 0)
    warmup_epochs = config.get('action_inject_warmup_epochs', 0)
    if epoch < start_epoch:
        return start
    if warmup_epochs <= 0:
        return end
    progress = min(max(epoch - start_epoch, 0) / float(warmup_epochs), 1.0)
    return start + (end - start) * progress


def get_action_contrast_weight(epoch, config):
    if 'action_contrast_weight_start' not in config or 'action_contrast_weight_end' not in config:
        return config.get('action_contrast_weight', 0.0)
    start = config.get('action_contrast_weight_start', 0.0)
    end = config.get('action_contrast_weight_end', config.get('action_contrast_weight', 0.0))
    start_epoch = config.get('action_contrast_start_epoch', 0)
    warmup_epochs = config.get('action_contrast_warmup_epochs', 0)
    if epoch < start_epoch:
        return start
    if warmup_epochs <= 0:
        return end
    progress = min(max(epoch - start_epoch, 0) / float(warmup_epochs), 1.0)
    return start + (end - start) * progress


def get_action_aux_weight(epoch, config):
    start = config.get('action_aux_weight_start', 0.0)
    end = config.get('action_aux_weight_end', 0.0)
    warmup_epochs = config.get('action_aux_warmup_epochs', 0)
    if warmup_epochs <= 0:
        return end
    progress = min(max(epoch, 0) / float(warmup_epochs), 1.0)
    return start + (end - start) * progress


def get_rollout_weight(epoch, config):
    start = config.get('rollout_weight_start', 0.0)
    end = config.get('rollout_weight_end', 0.0)
    start_epoch = config.get('rollout_start_epoch', 0)
    warmup_epochs = config.get('rollout_warmup_epochs', 0)
    if epoch < start_epoch:
        return start
    if warmup_epochs <= 0:
        return end
    progress = min(max(epoch - start_epoch, 0) / float(warmup_epochs), 1.0)
    return start + (end - start) * progress


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


def compute_rollout_loss(
    model,
    first_logits,
    context_tokens,
    context_actions,
    future_tokens,
    future_actions,
    action_scale,
):
    """使用模型自回归预测的token计算短rollout监督损失。"""
    if future_tokens is None or future_actions is None:
        return torch.tensor(0.0, device=context_tokens.device)
    if future_tokens.ndim != 4 or future_actions.ndim != 3:
        return torch.tensor(0.0, device=context_tokens.device)

    rollout_steps = min(future_tokens.size(1), future_actions.size(1))
    if rollout_steps <= 0:
        return torch.tensor(0.0, device=context_tokens.device)

    B = context_tokens.size(0)
    h, w = context_tokens.size(-2), context_tokens.size(-1)

    # t+1使用第一步logits的argmax（detach）作为free-run输入
    pred_token = first_logits.argmax(dim=-1).detach().view(B, h, w)
    token_buffer = torch.cat([context_tokens[:, 1:], pred_token.unsqueeze(1)], dim=1)
    action_buffer = torch.cat([context_actions[:, 1:], future_actions[:, :1]], dim=1)

    rollout_loss = torch.tensor(0.0, device=context_tokens.device)
    for step in range(rollout_steps):
        logits_roll = model(token_buffer, action_buffer, action_scale=action_scale)
        _, tokens_per_frame, vocab_size = logits_roll.shape
        target_roll = future_tokens[:, step].view(B, -1)

        ce_roll = nn.functional.cross_entropy(
            logits_roll.reshape(B * tokens_per_frame, vocab_size),
            target_roll.reshape(B * tokens_per_frame),
        )
        rollout_loss = rollout_loss + ce_roll

        if step + 1 < rollout_steps:
            pred_roll = logits_roll.argmax(dim=-1).detach().view(B, h, w)
            token_buffer = torch.cat([token_buffer[:, 1:], pred_roll.unsqueeze(1)], dim=1)
            action_buffer = torch.cat([action_buffer[:, 1:], future_actions[:, step + 1:step + 2]], dim=1)

    return rollout_loss / float(rollout_steps)


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, config, is_main, global_step=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_smooth_loss = 0
    total_contrast_loss = 0
    total_action_aux_loss = 0
    total_rollout_loss = 0

    # 当前epoch的平滑权重
    smooth_weight = get_smooth_weight(epoch, config)
    action_scale = get_action_inject_scale(epoch, config)
    contrast_weight = get_action_contrast_weight(epoch, config)
    action_aux_weight = get_action_aux_weight(epoch, config)
    rollout_weight = get_rollout_weight(epoch, config)
    max_grad_norm = config.get('max_grad_norm', 0.0)

    max_steps = config.get('max_steps_per_epoch')
    total_steps = min(len(dataloader), max_steps) if max_steps is not None else len(dataloader)
    iterator = dataloader
    if max_steps is not None:
        import itertools
        iterator = itertools.islice(dataloader, max_steps)

    pbar = tqdm(
        iterator,
        desc=f"Epoch {epoch} (smooth={smooth_weight:.4f})",
        total=total_steps,
        disable=not is_main,
        mininterval=config.get('tqdm_mininterval', 30.0),
        miniters=config.get('tqdm_miniters', config.get('log_every', 100)),
    )

    warmup_steps = config.get('lr_warmup_steps', 0)
    base_lr = config['lr']
    for batch_idx, batch in enumerate(pbar):
        if max_steps is not None and batch_idx >= max_steps:
            break

        if warmup_steps and global_step < warmup_steps:
            lr_scale = float(global_step + 1) / float(warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * lr_scale
        context_tokens = batch['context_tokens'].to(device)  # (B, T, H, W)
        context_actions = batch['context_actions'].to(device)  # (B, T, action_dim)
        target_token = batch['target_token'].to(device)  # (B, H, W)
        target_action = batch.get('target_action')
        if target_action is not None:
            target_action = target_action.to(device)
        future_tokens = batch.get('future_tokens')
        if future_tokens is not None:
            future_tokens = future_tokens.to(device)
        future_actions = batch.get('future_actions')
        if future_actions is not None:
            future_actions = future_actions.to(device)

        optimizer.zero_grad()

        use_contrast = (
            contrast_weight > 0
            and should_apply_action_contrast(config.get('action_contrast_prob', 1.0), device)
        )

        # 混合精度训练
        if config['use_amp']:
            with autocast(dtype=torch.bfloat16 if config['amp_dtype'] == 'bf16' else torch.float16):
                if config.get('use_action_aux', False):
                    logits, action_pred = model(
                        context_tokens,
                        context_actions,
                        action_scale=action_scale,
                        return_action_pred=True,
                    )
                else:
                    logits = model(
                        context_tokens, context_actions, action_scale=action_scale
                    )  # (B, tokens_per_frame, vocab)
                    action_pred = None

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
                    logits_pert = model(
                        context_tokens, perturbed_actions, action_scale=action_scale
                    )
                    contrast_loss = compute_action_contrast_loss(
                        logits,
                        logits_pert,
                        config.get('action_contrast_margin', 0.05),
                        config.get('action_contrast_mode', 'hinge'),
                    )

                action_aux_loss = torch.tensor(0.0, device=device)
                if action_pred is not None and target_action is not None:
                    action_aux_loss = nn.functional.smooth_l1_loss(action_pred, target_action)

                rollout_loss = torch.tensor(0.0, device=device)
                if rollout_weight > 0 and future_tokens is not None and future_actions is not None:
                    rollout_loss = compute_rollout_loss(
                        model,
                        logits,
                        context_tokens,
                        context_actions,
                        future_tokens,
                        future_actions,
                        action_scale,
                    )

                # 总损失
                loss = (
                    config['ce_weight'] * ce_loss
                    + smooth_weight * smooth_loss
                    + contrast_weight * contrast_loss
                    + action_aux_weight * action_aux_loss
                    + rollout_weight * rollout_loss
                )

            scaler.scale(loss).backward()
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if config.get('use_action_aux', False):
                logits, action_pred = model(
                    context_tokens,
                    context_actions,
                    action_scale=action_scale,
                    return_action_pred=True,
                )
            else:
                logits = model(context_tokens, context_actions, action_scale=action_scale)
                action_pred = None

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
                logits_pert = model(
                    context_tokens, perturbed_actions, action_scale=action_scale
                )
                contrast_loss = compute_action_contrast_loss(
                    logits,
                    logits_pert,
                    config.get('action_contrast_margin', 0.05),
                    config.get('action_contrast_mode', 'hinge'),
                )

            action_aux_loss = torch.tensor(0.0, device=device)
            if action_pred is not None and target_action is not None:
                action_aux_loss = nn.functional.smooth_l1_loss(action_pred, target_action)

            rollout_loss = torch.tensor(0.0, device=device)
            if rollout_weight > 0 and future_tokens is not None and future_actions is not None:
                rollout_loss = compute_rollout_loss(
                    model,
                    logits,
                    context_tokens,
                    context_actions,
                    future_tokens,
                    future_actions,
                    action_scale,
                )

            loss = (
                config['ce_weight'] * ce_loss
                + smooth_weight * smooth_loss
                + contrast_weight * contrast_loss
                + action_aux_weight * action_aux_loss
                + rollout_weight * rollout_loss
            )

            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # 统计
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_smooth_loss += smooth_loss.item()
        total_contrast_loss += contrast_loss.item()
        total_action_aux_loss += action_aux_loss.item()
        total_rollout_loss += rollout_loss.item()

        # 更新进度条
        if is_main and batch_idx % config['log_every'] == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{ce_loss.item():.4f}",
                'smooth': f"{smooth_loss.item():.4f}",
                'contrast': f"{contrast_loss.item():.4f}",
                'a_aux': f"{action_aux_loss.item():.4f}",
                'roll': f"{rollout_loss.item():.4f}",
                'a_scale': f"{action_scale:.3f}",
                'c_w': f"{contrast_weight:.3f}",
                'aux_w': f"{action_aux_weight:.3f}",
                'r_w': f"{rollout_weight:.3f}",
            })

        global_step += 1

    num_steps = min(len(dataloader), max_steps) if max_steps is not None else len(dataloader)
    avg_loss = total_loss / num_steps
    avg_ce = total_ce_loss / num_steps
    avg_smooth = total_smooth_loss / num_steps
    avg_contrast = total_contrast_loss / num_steps
    avg_action_aux = total_action_aux_loss / num_steps
    avg_rollout = total_rollout_loss / num_steps

    return avg_loss, avg_ce, avg_smooth, avg_contrast, avg_action_aux, avg_rollout, global_step


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, optimizer, epoch, loss, save_path, config=None):
    """保存checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': dict(config) if config is not None else None,
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
    parser.add_argument('--save-dir', type=str, default='../checkpoints/wm/world_model',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--max-steps-per-epoch', type=int, default=None,
                        help='Override max training steps per epoch')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Override checkpoint save interval in epochs')
    parser.add_argument('--stratified-ab', dest='stratified_ab', action='store_true',
                        help='Enable A/B stratified sampling')
    parser.add_argument('--no-stratified-ab', dest='stratified_ab', action='store_false',
                        help='Disable A/B stratified sampling')
    parser.set_defaults(stratified_ab=None)
    parser.add_argument('--conditioning-type', choices=['film', 'adaln_zero'], default=None,
                        help='Override action conditioning layer type')
    parser.add_argument('--use-action-aux', dest='use_action_aux', action='store_true',
                        help='Enable auxiliary action prediction loss')
    parser.add_argument('--no-action-aux', dest='use_action_aux', action='store_false',
                        help='Disable auxiliary action prediction loss')
    parser.set_defaults(use_action_aux=None)
    parser.add_argument('--use-memory', dest='use_memory', action='store_true',
                        help='Enable recurrent memory token')
    parser.add_argument('--no-memory', dest='use_memory', action='store_false',
                        help='Disable recurrent memory token')
    parser.set_defaults(use_memory=None)
    parser.add_argument('--rollout-steps', type=int, default=None,
                        help='Override short rollout supervision steps')
    parser.add_argument('--rollout-weight-start', type=float, default=None,
                        help='Override initial rollout loss weight')
    parser.add_argument('--rollout-weight-end', type=float, default=None,
                        help='Override final rollout loss weight')
    parser.add_argument('--action-aux-weight-start', type=float, default=None,
                        help='Override initial auxiliary action loss weight')
    parser.add_argument('--action-aux-weight-end', type=float, default=None,
                        help='Override final auxiliary action loss weight')
    parser.add_argument('--action-contrast-weight-start', type=float, default=None,
                        help='Override initial action contrast loss weight')
    parser.add_argument('--action-contrast-weight-end', type=float, default=None,
                        help='Override final action contrast loss weight')
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
    if args.lr is not None:
        config['lr'] = args.lr
    if args.max_steps_per_epoch is not None:
        config['max_steps_per_epoch'] = args.max_steps_per_epoch
    if args.save_every is not None:
        config['save_every'] = args.save_every
    if args.stratified_ab is not None:
        config['stratified_ab'] = args.stratified_ab
    if args.conditioning_type is not None:
        config['conditioning_type'] = args.conditioning_type
    if args.use_action_aux is not None:
        config['use_action_aux'] = args.use_action_aux
    if args.use_memory is not None:
        config['use_memory'] = args.use_memory
    if args.rollout_steps is not None:
        config['rollout_steps'] = args.rollout_steps
    if args.rollout_weight_start is not None:
        config['rollout_weight_start'] = args.rollout_weight_start
    if args.rollout_weight_end is not None:
        config['rollout_weight_end'] = args.rollout_weight_end
    if args.action_aux_weight_start is not None:
        config['action_aux_weight_start'] = args.action_aux_weight_start
    if args.action_aux_weight_end is not None:
        config['action_aux_weight_end'] = args.action_aux_weight_end
    if args.action_contrast_weight_start is not None:
        config['action_contrast_weight_start'] = args.action_contrast_weight_start
    if args.action_contrast_weight_end is not None:
        config['action_contrast_weight_end'] = args.action_contrast_weight_end

    # 设备/分布式
    distributed, local_rank, world_size, device = setup_distributed(args)
    rank = dist.get_rank() if distributed else 0
    if is_main_process():
        print(f"Using device: {device}")

    # TF32加速（对视觉任务通常无明显质量损失）
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

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
            rollout_steps=config.get('rollout_steps', 0),
            pin_memory=config.get('pin_memory', True),
            prefetch_factor=config.get('prefetch_factor', 4),
            persistent_workers=config.get('persistent_workers', True),
            stratified_ab=False,
            ab_split=config.get('ab_split', 750),
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
            rollout_steps=config.get('rollout_steps', 0),
            pin_memory=config.get('pin_memory', True),
            prefetch_factor=config.get('prefetch_factor', 4),
            persistent_workers=config.get('persistent_workers', True),
            stratified_ab=config.get('stratified_ab', False),
            ab_split=config.get('ab_split', 750),
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
        conditioning_type=config.get('conditioning_type', 'adaln_zero'),
        use_action_aux=config.get('use_action_aux', False),
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
        # Ensure resumed runs honor the current config learning rate.
        for pg in optimizer.param_groups:
            pg['lr'] = config['lr']
        start_epoch = checkpoint['epoch'] + 1

    # 训练
    if is_main_process():
        print("\nStarting training...")
        print(f"Curriculum learning: smooth weight {config['smooth_weight_start']} -> {config['smooth_weight_end']} over {config['smooth_warmup_epochs']} epochs")

    best_loss = float('inf')

    steps_per_epoch = min(len(dataloader), config.get('max_steps_per_epoch')) if config.get('max_steps_per_epoch') is not None else len(dataloader)
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, config['epochs']):
        if sampler is not None:
            sampler.set_epoch(epoch)
        avg_loss, avg_ce, avg_smooth, avg_contrast, avg_action_aux, avg_rollout, global_step = train_epoch(
            model, dataloader, optimizer, scaler, device, epoch, config, is_main_process(),
            global_step=global_step,
        )

        avg_loss = reduce_mean(avg_loss, device)
        avg_ce = reduce_mean(avg_ce, device)
        avg_smooth = reduce_mean(avg_smooth, device)
        avg_contrast = reduce_mean(avg_contrast, device)
        avg_action_aux = reduce_mean(avg_action_aux, device)
        avg_rollout = reduce_mean(avg_rollout, device)

        if is_main_process():
            print(f"\nEpoch {epoch}:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  CE: {avg_ce:.4f}")
            print(f"  Smooth: {avg_smooth:.4f}")
            print(f"  Contrast: {avg_contrast:.4f}")
            print(f"  ActionAux: {avg_action_aux:.4f}")
            print(f"  Rollout: {avg_rollout:.4f}")
            print(f"  Smooth Weight: {get_smooth_weight(epoch, config):.4f}")
            print(f"  Action Scale: {get_action_inject_scale(epoch, config):.4f}")
            print(f"  Contrast Weight: {get_action_contrast_weight(epoch, config):.4f}")
            print(f"  ActionAux Weight: {get_action_aux_weight(epoch, config):.4f}")
            print(f"  Rollout Weight: {get_rollout_weight(epoch, config):.4f}")

            # 保存checkpoint
            if (epoch + 1) % config['save_every'] == 0:
                save_path = save_dir / f"world_model_epoch_{epoch:03d}.pth"
                save_checkpoint(model, optimizer, epoch, avg_loss, save_path, config=config)

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = save_dir / "best.pth"
                save_checkpoint(model, optimizer, epoch, avg_loss, save_path, config=config)
                print(f"  New best model! Loss: {best_loss:.4f}")

    if is_main_process():
        print("\nTraining complete!")
        print(f"Best loss: {best_loss:.4f}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
