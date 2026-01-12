"""
世界模型评估脚本

评估世界模型在不同条件下的生成质量：
1. 单步预测准确率
2. 多步自回归生成质量（误差累积分析）
3. 动作条件一致性
"""

import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
import sys
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.vqvae_v2 import VQVAE_V2
from models.world_model import WorldModel
from train.config import WM_CONFIG
from evaluate.metrics import VideoMetrics


def compute_stability_metrics(metrics_over_steps):
    """
    计算长期稳定性指标

    Args:
        metrics_over_steps: 包含每个时间步PSNR/SSIM的字典

    Returns:
        stability_metrics: 稳定性指标字典
    """
    psnr_values = np.array(metrics_over_steps['psnr'])
    ssim_values = np.array(metrics_over_steps['ssim'])
    steps = np.array(metrics_over_steps['step'])

    stability = {}

    # 1. 崩溃点检测（PSNR < 15 dB 认为完全崩溃）
    collapse_threshold = 15.0
    collapse_indices = np.where(psnr_values < collapse_threshold)[0]
    if len(collapse_indices) > 0:
        stability['collapse_frame'] = int(steps[collapse_indices[0]])
    else:
        stability['collapse_frame'] = -1  # 未崩溃

    # 2. PSNR半衰期（降到初始值50%的帧数）
    initial_psnr = psnr_values[0]
    half_psnr = initial_psnr * 0.5
    half_life_indices = np.where(psnr_values < half_psnr)[0]
    if len(half_life_indices) > 0:
        stability['psnr_half_life'] = int(steps[half_life_indices[0]])
    else:
        stability['psnr_half_life'] = -1  # 未达到

    # 3. SSIM半衰期
    initial_ssim = ssim_values[0]
    half_ssim = initial_ssim * 0.5
    ssim_half_life_indices = np.where(ssim_values < half_ssim)[0]
    if len(ssim_half_life_indices) > 0:
        stability['ssim_half_life'] = int(steps[ssim_half_life_indices[0]])
    else:
        stability['ssim_half_life'] = -1

    # 4. 平均衰减率（每帧PSNR下降多少）
    if len(psnr_values) > 1:
        psnr_decay_rate = (psnr_values[0] - psnr_values[-1]) / len(psnr_values)
        stability['psnr_decay_rate'] = float(psnr_decay_rate)
    else:
        stability['psnr_decay_rate'] = 0.0

    # 5. 稳定性分数（0-100，越高越稳定）
    # 基于PSNR方差和衰减率
    psnr_std = np.std(psnr_values)
    stability_score = max(0, 100 - psnr_std * 2 - abs(stability['psnr_decay_rate']) * 10)
    stability['stability_score'] = float(stability_score)

    return stability


def load_models(vqvae_path, world_model_path, device):
    """加载模型"""
    # VQ-VAE
    vqvae = VQVAE_V2().to(device)
    checkpoint = torch.load(vqvae_path, map_location=device, weights_only=False)
    vqvae.load_state_dict(checkpoint['model_state_dict'])
    vqvae.eval()

    # World Model
    config = WM_CONFIG
    world_model = WorldModel(
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

    checkpoint = torch.load(world_model_path, map_location=device, weights_only=False)
    world_model.load_state_dict(checkpoint['model_state_dict'])
    world_model.eval()

    return vqvae, world_model


def decode_tokens_to_frame(vqvae, tokens, device):
    """将tokens解码为图像帧"""
    with torch.no_grad():
        tokens_tensor = torch.from_numpy(tokens).unsqueeze(0).to(device)
        frame = vqvae.decode_tokens(tokens_tensor)
        frame = frame.squeeze(0).cpu().numpy()
        frame = (frame + 1.0) / 2.0
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        frame = np.transpose(frame, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        return frame


def evaluate_single_step(vqvae, world_model, tokens, actions, device, num_samples=100):
    """
    评估单步预测准确率

    Returns:
        token_accuracy: token预测准确率
        metrics: 图像质量指标
    """
    context_frames = world_model.context_frames
    metrics_calc = VideoMetrics(device=device, use_lpips=True)

    correct_tokens = 0
    total_tokens = 0
    pred_frames = []
    real_frames = []

    # 随机采样评估点
    max_idx = len(tokens) - context_frames - 1
    sample_indices = np.random.choice(max_idx, min(num_samples, max_idx), replace=False)

    print(f"Evaluating single-step prediction on {len(sample_indices)} samples...")

    with torch.no_grad():
        for idx in tqdm(sample_indices):
            # 准备输入
            context_tokens = torch.from_numpy(
                tokens[idx:idx + context_frames]
            ).unsqueeze(0).to(device)

            action_window = torch.from_numpy(
                actions[idx:idx + context_frames]
            ).float().unsqueeze(0).to(device)

            target_token = tokens[idx + context_frames]

            # 预测
            logits = world_model(context_tokens, action_window)
            pred_token = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            pred_token = pred_token.reshape(16, 16)

            # 计算token准确率
            correct_tokens += (pred_token == target_token).sum()
            total_tokens += target_token.size

            # 解码为图像
            pred_frame = decode_tokens_to_frame(vqvae, pred_token, device)
            real_frame = decode_tokens_to_frame(vqvae, target_token, device)

            pred_frames.append(pred_frame)
            real_frames.append(real_frame)

    pred_frames = np.array(pred_frames)
    real_frames = np.array(real_frames)

    # 计算指标
    token_accuracy = correct_tokens / total_tokens
    image_metrics = metrics_calc.compute_all_metrics(pred_frames, real_frames)

    return {
        'token_accuracy': float(token_accuracy),
        **image_metrics
    }


def evaluate_autoregressive(
    vqvae, world_model, tokens, actions, device,
    num_sequences=10, sequence_length=50, temperature=0.7, top_k=50
):
    """
    评估自回归生成质量（误差累积分析）

    Returns:
        metrics_over_steps: 每个时间步的指标
        overall_metrics: 整体指标
    """
    context_frames = world_model.context_frames
    metrics_calc = VideoMetrics(device=device, use_lpips=True)

    all_pred_frames = []
    all_real_frames = []

    # 随机选择起始点
    max_start = len(tokens) - context_frames - sequence_length
    start_indices = np.random.choice(max_start, min(num_sequences, max_start), replace=False)

    print(f"Evaluating autoregressive generation: {num_sequences} sequences x {sequence_length} steps...")

    for seq_idx, start_idx in enumerate(tqdm(start_indices)):
        # 初始化token buffer
        token_buffer = torch.from_numpy(
            tokens[start_idx:start_idx + context_frames]
        ).unsqueeze(0).to(device)

        seq_pred_frames = []
        seq_real_frames = []

        with torch.no_grad():
            for t in range(sequence_length):
                # 动作窗口
                action_idx = start_idx + context_frames + t
                action_window = torch.from_numpy(
                    actions[action_idx - context_frames + 1:action_idx + 1]
                ).float().unsqueeze(0).to(device)

                # 预测下一帧
                pred_tokens = world_model.predict_next_frame(
                    token_buffer, action_window,
                    temperature=temperature, top_k=top_k
                )

                # 解码
                pred_frame = decode_tokens_to_frame(
                    vqvae, pred_tokens.squeeze(0).cpu().numpy(), device
                )
                real_frame = decode_tokens_to_frame(
                    vqvae, tokens[start_idx + context_frames + t], device
                )

                seq_pred_frames.append(pred_frame)
                seq_real_frames.append(real_frame)

                # 更新buffer（使用预测的token，模拟真实推理）
                token_buffer = torch.cat([
                    token_buffer[:, 1:],
                    pred_tokens.unsqueeze(1)
                ], dim=1)

        all_pred_frames.append(seq_pred_frames)
        all_real_frames.append(seq_real_frames)

    # 转换为numpy数组
    all_pred_frames = np.array(all_pred_frames)  # (num_seq, seq_len, H, W, C)
    all_real_frames = np.array(all_real_frames)

    # 计算每个时间步的平均指标
    metrics_over_steps = {
        'step': [],
        'psnr': [],
        'ssim': [],
    }

    for t in range(sequence_length):
        pred_t = all_pred_frames[:, t]  # (num_seq, H, W, C)
        real_t = all_real_frames[:, t]

        psnr = metrics_calc.compute_psnr(pred_t, real_t)
        ssim = metrics_calc.compute_ssim(pred_t, real_t)

        metrics_over_steps['step'].append(t)
        metrics_over_steps['psnr'].append(float(psnr))
        metrics_over_steps['ssim'].append(float(ssim))

    # 计算整体指标
    all_pred_flat = all_pred_frames.reshape(-1, *all_pred_frames.shape[2:])
    all_real_flat = all_real_frames.reshape(-1, *all_real_frames.shape[2:])
    overall_metrics = metrics_calc.compute_all_metrics(all_pred_flat, all_real_flat)

    # 计算长期稳定性指标
    stability_metrics = compute_stability_metrics(metrics_over_steps)
    overall_metrics.update(stability_metrics)

    return metrics_over_steps, overall_metrics


def evaluate_action_consistency(
    vqvae, world_model, tokens, actions, device, num_samples=50
):
    """
    评估动作条件一致性：相同历史+不同动作应该产生不同结果

    Returns:
        action_sensitivity: 动作敏感度分数
    """
    context_frames = world_model.context_frames

    sensitivities = []

    max_idx = len(tokens) - context_frames - 1
    sample_indices = np.random.choice(max_idx, min(num_samples, max_idx), replace=False)

    print(f"Evaluating action consistency on {len(sample_indices)} samples...")

    with torch.no_grad():
        for idx in tqdm(sample_indices):
            # 准备输入
            context_tokens = torch.from_numpy(
                tokens[idx:idx + context_frames]
            ).unsqueeze(0).to(device)

            original_action = actions[idx:idx + context_frames]

            # 原始动作预测
            action_window = torch.from_numpy(original_action).float().unsqueeze(0).to(device)
            logits_original = world_model(context_tokens, action_window)
            probs_original = torch.softmax(logits_original, dim=-1)

            # 修改动作（增加转向）
            modified_action = original_action.copy()
            modified_action[:, 0] += 0.5  # 增加转向
            modified_action = np.clip(modified_action, -1, 1)

            action_window_mod = torch.from_numpy(modified_action).float().unsqueeze(0).to(device)
            logits_modified = world_model(context_tokens, action_window_mod)
            probs_modified = torch.softmax(logits_modified, dim=-1)

            # 计算预测分布的差异（KL散度）
            kl_div = torch.nn.functional.kl_div(
                probs_modified.log(), probs_original, reduction='batchmean'
            ).item()

            sensitivities.append(kl_div)

    return {
        'action_sensitivity_mean': float(np.mean(sensitivities)),
        'action_sensitivity_std': float(np.std(sensitivities)),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate World Model')
    parser.add_argument('--vqvae-checkpoint', type=str, required=True)
    parser.add_argument('--world-model-checkpoint', type=str, required=True)
    parser.add_argument('--token-file', type=str, required=True)
    parser.add_argument('--output', type=str, default='evaluation_results.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples for single-step evaluation')
    parser.add_argument('--num-sequences', type=int, default=10,
                        help='Number of sequences for autoregressive evaluation')
    parser.add_argument('--sequence-length', type=int, default=50,
                        help='Length of each autoregressive sequence')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    print("\nLoading models...")
    vqvae, world_model = load_models(
        args.vqvae_checkpoint,
        args.world_model_checkpoint,
        device
    )

    # 加载数据
    print("\nLoading data...")
    data = np.load(args.token_file)
    tokens = data['tokens']
    actions = data['actions']
    print(f"Tokens shape: {tokens.shape}")
    print(f"Actions shape: {actions.shape}")

    results = {}

    # 1. 单步预测评估
    print("\n" + "="*60)
    print("1. Single-Step Prediction Evaluation")
    print("="*60)
    single_step_results = evaluate_single_step(
        vqvae, world_model, tokens, actions, device,
        num_samples=args.num_samples
    )
    results['single_step'] = single_step_results

    print(f"\nToken Accuracy: {single_step_results['token_accuracy']:.4f}")
    print(f"PSNR: {single_step_results['psnr']:.2f} dB")
    print(f"SSIM: {single_step_results['ssim']:.4f}")
    if 'lpips' in single_step_results:
        print(f"LPIPS: {single_step_results['lpips']:.4f}")

    # 2. 自回归生成评估
    print("\n" + "="*60)
    print("2. Autoregressive Generation Evaluation")
    print("="*60)
    metrics_over_steps, ar_overall = evaluate_autoregressive(
        vqvae, world_model, tokens, actions, device,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length
    )
    results['autoregressive'] = {
        'over_steps': metrics_over_steps,
        'overall': ar_overall
    }

    print(f"\nOverall PSNR: {ar_overall['psnr']:.2f} dB")
    print(f"Overall SSIM: {ar_overall['ssim']:.4f}")

    # 显示稳定性指标
    print("\nLong-term Stability Metrics:")
    if ar_overall['collapse_frame'] > 0:
        print(f"  Collapse Frame: {ar_overall['collapse_frame']} (PSNR < 15 dB)")
    else:
        print(f"  Collapse Frame: Not collapsed within {args.sequence_length} frames")

    if ar_overall['psnr_half_life'] > 0:
        print(f"  PSNR Half-Life: {ar_overall['psnr_half_life']} frames")
    else:
        print(f"  PSNR Half-Life: > {args.sequence_length} frames")

    print(f"  PSNR Decay Rate: {ar_overall['psnr_decay_rate']:.4f} dB/frame")
    print(f"  Stability Score: {ar_overall['stability_score']:.1f}/100")

    print("\nPSNR degradation over time:")
    for i in range(0, len(metrics_over_steps['step']), 10):
        step = metrics_over_steps['step'][i]
        psnr = metrics_over_steps['psnr'][i]
        ssim = metrics_over_steps['ssim'][i]
        print(f"  Step {step:3d}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")

    # 3. 动作一致性评估
    print("\n" + "="*60)
    print("3. Action Consistency Evaluation")
    print("="*60)
    action_results = evaluate_action_consistency(
        vqvae, world_model, tokens, actions, device,
        num_samples=args.num_samples
    )
    results['action_consistency'] = action_results

    print(f"\nAction Sensitivity: {action_results['action_sensitivity_mean']:.4f} "
          f"(±{action_results['action_sensitivity_std']:.4f})")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to {output_path}")

    # 打印总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Single-step Token Accuracy: {single_step_results['token_accuracy']:.4f}")
    print(f"Single-step PSNR: {single_step_results['psnr']:.2f} dB")
    print(f"Autoregressive PSNR (avg): {ar_overall['psnr']:.2f} dB")
    print(f"PSNR at step 0: {metrics_over_steps['psnr'][0]:.2f} dB")
    print(f"PSNR at step {args.sequence_length-1}: {metrics_over_steps['psnr'][-1]:.2f} dB")
    print(f"PSNR degradation: {metrics_over_steps['psnr'][0] - metrics_over_steps['psnr'][-1]:.2f} dB")
    print(f"\nStability:")
    if ar_overall['collapse_frame'] > 0:
        print(f"  Collapsed at frame {ar_overall['collapse_frame']}")
    else:
        print(f"  No collapse (stable for {args.sequence_length} frames)")
    print(f"  Stability Score: {ar_overall['stability_score']:.1f}/100")


if __name__ == '__main__':
    main()
