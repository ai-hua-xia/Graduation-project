"""
分析Scheduled Sampling训练过程中loss波动的原因
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

log_file = Path('/home/llb/HunyuanWorld-Voyager/bishe/carla_project/logs/train_ss.log')

epochs = []
losses = []
sampling_probs = []

# 解析日志
with open(log_file, 'r') as f:
    content = f.read()

# 使用正则表达式提取数据
pattern = r'Epoch (\d+):\s+Loss: ([\d\.]+)\s+CE: [\d\.]+\s+Sampling Prob: ([\d\.]+)'
matches = re.findall(pattern, content)

for match in matches:
    epochs.append(int(match[0]))
    losses.append(float(match[1]))
    sampling_probs.append(float(match[2]))

epochs = np.array(epochs)
losses = np.array(losses)
sampling_probs = np.array(sampling_probs)

print(f"提取了 {len(epochs)} 个epoch的数据")
print(f"Loss范围: {losses.min():.4f} - {losses.max():.4f}")
print(f"Sampling Prob范围: {sampling_probs.min():.2f} - {sampling_probs.max():.2f}")

# 计算移动平均
window = 5
if len(losses) >= window:
    loss_ma = np.convolve(losses, np.ones(window)/window, mode='valid')
    epochs_ma = epochs[window-1:]
else:
    loss_ma = losses
    epochs_ma = epochs

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Loss vs Epoch (原始 + 移动平均)
ax1 = axes[0, 0]
ax1.plot(epochs, losses, 'o-', alpha=0.5, color='steelblue', label='Raw Loss', markersize=4)
if len(loss_ma) > 0:
    ax1.plot(epochs_ma, loss_ma, '-', linewidth=2.5, color='darkblue', label=f'{window}-Epoch Moving Avg')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Loss Curve: Raw vs Smoothed', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Sampling Prob vs Epoch
ax2 = axes[0, 1]
ax2.plot(epochs, sampling_probs, 'o-', color='orange', linewidth=2, markersize=4)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Sampling Probability', fontsize=11)
ax2.set_title('Scheduled Sampling Probability', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Loss vs Sampling Prob (散点图)
ax3 = axes[1, 0]
scatter = ax3.scatter(sampling_probs, losses, c=epochs, cmap='viridis', s=50, alpha=0.7)
ax3.set_xlabel('Sampling Probability', fontsize=11)
ax3.set_ylabel('Loss', fontsize=11)
ax3.set_title('Loss vs Sampling Prob (colored by epoch)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Epoch', fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Loss变化率 vs Sampling Prob变化
ax4 = axes[1, 1]
if len(losses) > 1:
    loss_diff = np.diff(losses)
    sampling_diff = np.diff(sampling_probs)
    epochs_diff = epochs[1:]

    # 分段显示
    increase_mask = loss_diff > 0
    decrease_mask = loss_diff <= 0

    ax4.scatter(epochs_diff[increase_mask], loss_diff[increase_mask],
                c='red', alpha=0.6, s=40, label='Loss Increase')
    ax4.scatter(epochs_diff[decrease_mask], loss_diff[decrease_mask],
                c='green', alpha=0.6, s=40, label='Loss Decrease')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss Change (Δ)', fontsize=11)
    ax4.set_title('Loss Change per Epoch', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.suptitle('Scheduled Sampling Training Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_file = '/home/llb/HunyuanWorld-Voyager/bishe/carla_project/figures/ss_training_analysis.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n保存图表到: {output_file}")

# 统计分析
print("\n" + "="*60)
print("统计分析:")
print("="*60)

# 按sampling prob分段分析
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
for i in range(len(bins)-1):
    mask = (sampling_probs >= bins[i]) & (sampling_probs < bins[i+1])
    if mask.sum() > 0:
        avg_loss = losses[mask].mean()
        std_loss = losses[mask].std()
        print(f"Sampling Prob [{bins[i]:.1f}, {bins[i+1]:.1f}): "
              f"Avg Loss = {avg_loss:.4f} ± {std_loss:.4f}")

print("\n" + "="*60)
print("Loss波动分析:")
print("="*60)
if len(losses) > 1:
    loss_changes = np.diff(losses)
    increases = loss_changes[loss_changes > 0]
    decreases = loss_changes[loss_changes < 0]

    print(f"总epoch数: {len(epochs)}")
    print(f"Loss上升次数: {len(increases)} ({len(increases)/len(loss_changes)*100:.1f}%)")
    print(f"Loss下降次数: {len(decreases)} ({len(decreases)/len(loss_changes)*100:.1f}%)")
    print(f"平均上升幅度: {increases.mean():.4f}" if len(increases) > 0 else "平均上升幅度: N/A")
    print(f"平均下降幅度: {decreases.mean():.4f}" if len(decreases) > 0 else "平均下降幅度: N/A")
    print(f"Loss标准差: {losses.std():.4f}")
