"""
分析预测视频的质量衰减
"""
import torch
import numpy as np
import cv2
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from models.vqvae_v2 import load_vqvae_v2_checkpoint
from models.world_model import WorldModel
from train.config import WM_CONFIG
from evaluate.metrics import VideoMetrics


def load_models(vqvae_path, wm_path, device='cuda', num_embeddings=None):
    """加载模型"""
    vqvae, _ = load_vqvae_v2_checkpoint(vqvae_path, device)
    vqvae.eval()

    config = WM_CONFIG.copy()
    if num_embeddings is not None:
        config['num_embeddings'] = num_embeddings
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

    checkpoint = torch.load(wm_path, map_location=device, weights_only=False)
    world_model.load_state_dict(checkpoint['model_state_dict'])
    world_model.eval()

    return vqvae, world_model


def tokens_to_image(vqvae, tokens, device):
    """将tokens解码为图像"""
    with torch.no_grad():
        if tokens.ndim == 1:
            h = w = int(np.sqrt(len(tokens)))
            tokens = tokens.reshape(h, w)

        tokens_tensor = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
        frame = vqvae.decode_tokens(tokens_tensor)
        frame = frame.squeeze(0).cpu().numpy()
        frame = (frame + 1.0) / 2.0
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        frame = np.transpose(frame, (1, 2, 0))
        return frame


def analyze_prediction_quality(vqvae, world_model, tokens, actions, start_idx, num_frames, device):
    """分析预测质量随时间的变化"""

    context_frames = world_model.context_frames
    metrics_calc = VideoMetrics(device=device, use_lpips=False)

    # 初始化
    context_tokens = tokens[start_idx:start_idx+context_frames].copy()

    psnr_values = []
    ssim_values = []
    token_accuracy = []
    action_changes = []

    print(f"Analyzing {num_frames} frames...")
    print()

    with torch.no_grad():
        for t in range(num_frames):
            # 准备输入
            context_tensor = torch.from_numpy(context_tokens).long().unsqueeze(0).to(device)
            action_seq = actions[start_idx+t:start_idx+t+context_frames]
            action_tensor = torch.from_numpy(action_seq).float().unsqueeze(0).to(device)

            # 预测
            logits = world_model(context_tensor, action_tensor)
            pred_tokens = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

            # 获取真实tokens
            gt_idx = start_idx + context_frames + t
            if gt_idx >= len(tokens):
                break
            gt_tokens = tokens[gt_idx].flatten()

            # 计算指标
            accuracy = np.mean(pred_tokens == gt_tokens)
            token_accuracy.append(accuracy)

            # 解码为图像计算PSNR/SSIM
            pred_frame = tokens_to_image(vqvae, pred_tokens, device)
            gt_frame = tokens_to_image(vqvae, tokens[gt_idx], device)

            psnr = metrics_calc.compute_psnr(pred_frame, gt_frame)
            ssim = metrics_calc.compute_ssim(pred_frame, gt_frame)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            # 记录动作变化
            if t > 0:
                action_change = np.abs(action_seq[-1] - action_seq[-2]).sum()
                action_changes.append(action_change)

            # 更新上下文
            h = w = int(np.sqrt(len(pred_tokens)))
            pred_tokens_2d = pred_tokens.reshape(h, w)
            context_tokens = np.roll(context_tokens, -1, axis=0)
            context_tokens[-1] = pred_tokens_2d

            # 每10帧打印一次
            if (t + 1) % 10 == 0:
                print(f"Frame {t+1:3d}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, Token Acc={accuracy:.2%}")

    return {
        'psnr': np.array(psnr_values),
        'ssim': np.array(ssim_values),
        'token_accuracy': np.array(token_accuracy),
        'action_changes': np.array(action_changes),
    }


def plot_analysis(results, output_path):
    """绘制分析图表"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    frames = np.arange(len(results['psnr']))
    fps = 10
    time_seconds = frames / fps

    # PSNR随时间变化
    ax1 = axes[0, 0]
    ax1.plot(time_seconds, results['psnr'], 'b-', linewidth=2)
    ax1.axhline(y=30, color='orange', linestyle='--', label='Good quality (30 dB)')
    ax1.axhline(y=20, color='red', linestyle='--', label='Poor quality (20 dB)')
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('PSNR (dB)', fontsize=11)
    ax1.set_title('PSNR Degradation Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SSIM随时间变化
    ax2 = axes[0, 1]
    ax2.plot(time_seconds, results['ssim'], 'g-', linewidth=2)
    ax2.axhline(y=0.9, color='orange', linestyle='--', label='Good (0.9)')
    ax2.axhline(y=0.7, color='red', linestyle='--', label='Poor (0.7)')
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('SSIM', fontsize=11)
    ax2.set_title('SSIM Degradation Over Time', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Token准确率
    ax3 = axes[1, 0]
    ax3.plot(time_seconds, results['token_accuracy'] * 100, 'purple', linewidth=2)
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Token Accuracy (%)', fontsize=11)
    ax3.set_title('Token Prediction Accuracy', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 动作变化
    ax4 = axes[1, 1]
    if len(results['action_changes']) > 0:
        ax4.plot(time_seconds[1:], results['action_changes'], 'orange', linewidth=2)
        ax4.set_xlabel('Time (seconds)', fontsize=11)
        ax4.set_ylabel('Action Change', fontsize=11)
        ax4.set_title('Action Changes Over Time', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

    plt.suptitle('World Model Prediction Quality Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: {output_path}")


def main():
    print("="*70)
    print("  Video Quality Analysis")
    print("="*70)
    print()

    # 加载数据
    data = np.load('data/tokens_raw/tokens_actions.npz')
    tokens = data['tokens']
    actions = data['actions']
    num_embeddings = int(tokens.max()) + 1

    # 加载模型
    vqvae, world_model = load_models(
        'checkpoints/vqvae/vqvae_v2/best.pth',
        'checkpoints/wm_ss/world_model_v5_ss/best.pth',
        'cuda',
        num_embeddings=num_embeddings,
    )

    # 分析100帧（单个episode长度）
    start_idx = 1000
    num_frames = 100

    results = analyze_prediction_quality(
        vqvae, world_model, tokens, actions,
        start_idx, num_frames, 'cuda'
    )

    # 统计分析
    print()
    print("="*70)
    print("  Summary Statistics")
    print("="*70)
    print()

    # 找到质量下降的关键点
    psnr = results['psnr']

    # 找到PSNR < 30 dB的第一帧
    poor_quality_idx = np.where(psnr < 30)[0]
    if len(poor_quality_idx) > 0:
        poor_frame = poor_quality_idx[0]
        poor_time = poor_frame / 10
        print(f"Quality becomes poor (<30 dB) at: Frame {poor_frame} (~{poor_time:.1f}s)")

    # 找到PSNR < 20 dB的第一帧（崩溃）
    collapse_idx = np.where(psnr < 20)[0]
    if len(collapse_idx) > 0:
        collapse_frame = collapse_idx[0]
        collapse_time = collapse_frame / 10
        print(f"Prediction collapses (<20 dB) at: Frame {collapse_frame} (~{collapse_time:.1f}s)")
    else:
        print(f"No collapse detected in {num_frames} frames")

    print()
    print(f"Average PSNR: {psnr.mean():.2f} dB")
    print(f"Average SSIM: {results['ssim'].mean():.4f}")
    print(f"Average Token Accuracy: {results['token_accuracy'].mean():.2%}")
    print()

    # 分段统计
    print("Quality by time period:")
    periods = [(0, 33, '0-3.3s'), (33, 66, '3.3-6.6s'), (66, 100, '6.6-10s')]
    for start, end, label in periods:
        if end <= len(psnr):
            avg_psnr = psnr[start:end].mean()
            avg_ssim = results['ssim'][start:end].mean()
            print(f"  {label:8s}: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")

    # 绘制分析图
    plot_analysis(results, 'outputs/analysis/video_quality_analysis.png')

    print()
    print("="*70)


if __name__ == '__main__':
    main()
