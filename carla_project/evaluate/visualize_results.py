"""
评估结果可视化

生成评估指标的图表
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_metrics_over_time(results: dict, output_dir: Path):
    """绘制指标随时间变化的曲线"""

    if 'autoregressive' not in results:
        print("No autoregressive results found")
        return

    over_steps = results['autoregressive']['over_steps']
    ar_overall = results['autoregressive']['overall']
    steps = over_steps['step']
    psnr = over_steps['psnr']
    ssim = over_steps['ssim']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # PSNR
    axes[0].plot(steps, psnr, 'b-', linewidth=2)
    axes[0].set_xlabel('Generation Step', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('PSNR Degradation Over Time', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=psnr[0], color='g', linestyle='--', alpha=0.5, label=f'Initial: {psnr[0]:.1f}')
    axes[0].axhline(y=psnr[-1], color='r', linestyle='--', alpha=0.5, label=f'Final: {psnr[-1]:.1f}')

    # 标记崩溃点
    if ar_overall.get('collapse_frame', -1) > 0:
        collapse_frame = ar_overall['collapse_frame']
        axes[0].axvline(x=collapse_frame, color='red', linestyle=':', linewidth=2, label=f'Collapse: {collapse_frame}')

    # 标记半衰期
    if ar_overall.get('psnr_half_life', -1) > 0:
        half_life = ar_overall['psnr_half_life']
        axes[0].axvline(x=half_life, color='orange', linestyle=':', linewidth=2, label=f'Half-life: {half_life}')

    axes[0].legend()

    # SSIM
    axes[1].plot(steps, ssim, 'r-', linewidth=2)
    axes[1].set_xlabel('Generation Step', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM Degradation Over Time', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=ssim[0], color='g', linestyle='--', alpha=0.5, label=f'Initial: {ssim[0]:.3f}')
    axes[1].axhline(y=ssim[-1], color='r', linestyle='--', alpha=0.5, label=f'Final: {ssim[-1]:.3f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'metrics_over_time.png'}")


def plot_comparison_bar(results: dict, output_dir: Path):
    """绘制单步vs自回归的对比柱状图"""

    if 'single_step' not in results or 'autoregressive' not in results:
        print("Missing results for comparison")
        return

    single = results['single_step']
    ar = results['autoregressive']['overall']

    metrics = ['psnr', 'ssim']
    single_values = [single['psnr'], single['ssim']]
    ar_values = [ar['psnr'], ar['ssim']]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, single_values, width, label='Single-Step', color='steelblue')
    bars2 = ax.bar(x + width/2, ar_values, width, label='Autoregressive', color='coral')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Single-Step vs Autoregressive Generation', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['PSNR (dB)', 'SSIM'])
    ax.legend()

    # 添加数值标签
    for bar, val in zip(bars1, single_values):
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    for bar, val in zip(bars2, ar_values):
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'comparison_bar.png'}")


def plot_summary_table(results: dict, output_dir: Path):
    """生成汇总表格图"""

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # 准备数据
    data = []
    headers = ['Metric', 'Single-Step', 'Autoregressive', 'Degradation']

    if 'single_step' in results:
        single = results['single_step']
        ar = results['autoregressive']['overall'] if 'autoregressive' in results else {}

        data.append(['Token Accuracy', f"{single.get('token_accuracy', 0):.4f}", '-', '-'])
        data.append(['PSNR (dB)', f"{single.get('psnr', 0):.2f}",
                     f"{ar.get('psnr', 0):.2f}",
                     f"{single.get('psnr', 0) - ar.get('psnr', 0):.2f}"])
        data.append(['SSIM', f"{single.get('ssim', 0):.4f}",
                     f"{ar.get('ssim', 0):.4f}",
                     f"{single.get('ssim', 0) - ar.get('ssim', 0):.4f}"])

        if 'lpips' in single:
            data.append(['LPIPS', f"{single.get('lpips', 0):.4f}",
                         f"{ar.get('lpips', 0):.4f}",
                         f"{ar.get('lpips', 0) - single.get('lpips', 0):.4f}"])

        # 添加稳定性指标
        data.append(['', '', '', ''])  # 空行
        if ar.get('collapse_frame', -1) > 0:
            data.append(['Collapse Frame', '-', f"{ar['collapse_frame']}", '-'])
        else:
            data.append(['Collapse Frame', '-', 'No collapse', '-'])

        if ar.get('psnr_half_life', -1) > 0:
            data.append(['PSNR Half-Life', '-', f"{ar['psnr_half_life']} frames", '-'])
        else:
            data.append(['PSNR Half-Life', '-', '> sequence length', '-'])

        data.append(['PSNR Decay Rate', '-', f"{ar.get('psnr_decay_rate', 0):.4f} dB/frame", '-'])
        data.append(['Stability Score', '-', f"{ar.get('stability_score', 0):.1f}/100", '-'])

    if 'action_consistency' in results:
        ac = results['action_consistency']
        data.append(['', '', '', ''])  # 空行
        data.append(['Action Sensitivity', f"{ac['action_sensitivity_mean']:.4f}", '-', '-'])

    # 创建表格
    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['lightsteelblue'] * 4
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 设置标题
    ax.set_title('World Model Evaluation Summary', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'summary_table.png'}")


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to evaluation results JSON file')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Output directory for figures')

    args = parser.parse_args()

    # 加载结果
    with open(args.results, 'r') as f:
        results = json.load(f)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成图表
    print("Generating visualizations...")
    plot_metrics_over_time(results, output_dir)
    plot_comparison_bar(results, output_dir)
    plot_summary_table(results, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == '__main__':
    main()
