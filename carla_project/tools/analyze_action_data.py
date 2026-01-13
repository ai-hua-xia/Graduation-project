"""
分析训练数据中的动作分布
"""
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load('data/tokens_v2/tokens_actions.npz')
actions = data['actions']

steering = actions[:, 0]
throttle = actions[:, 1]

print("="*70)
print("  Action Distribution Analysis")
print("="*70)
print()

print(f"Total samples: {len(actions)}")
print()

print("Steering statistics:")
print(f"  Mean: {steering.mean():.4f}")
print(f"  Std:  {steering.std():.4f}")
print(f"  Min:  {steering.min():.4f}")
print(f"  Max:  {steering.max():.4f}")
print()

# 分类统计
left_turn = np.sum(steering < -0.1)
straight = np.sum(np.abs(steering) <= 0.1)
right_turn = np.sum(steering > 0.1)

print("Steering distribution:")
print(f"  Left turn  (< -0.1): {left_turn:5d} ({left_turn/len(steering)*100:.1f}%)")
print(f"  Straight   (±0.1):   {straight:5d} ({straight/len(steering)*100:.1f}%)")
print(f"  Right turn (> 0.1):  {right_turn:5d} ({right_turn/len(steering)*100:.1f}%)")
print()

# 检查连续帧的动作变化
action_changes = np.abs(np.diff(steering))
print("Action changes between consecutive frames:")
print(f"  Mean change: {action_changes.mean():.4f}")
print(f"  Std change:  {action_changes.std():.4f}")
print(f"  Max change:  {action_changes.max():.4f}")
print()

# 统计显著变化的比例
significant_changes = np.sum(action_changes > 0.1)
print(f"Significant changes (>0.1): {significant_changes} ({significant_changes/len(action_changes)*100:.1f}%)")
print()

print("="*70)
print("Diagnosis:")
print("="*70)

if steering.std() < 0.1:
    print("⚠️  WARNING: Very low steering variance!")
    print("   Most actions are similar (mostly straight driving)")
    print("   Model may not learn action sensitivity")
elif straight / len(steering) > 0.8:
    print("⚠️  WARNING: Too much straight driving!")
    print(f"   {straight/len(steering)*100:.1f}% of samples are straight")
    print("   Model may ignore steering input")
elif action_changes.mean() < 0.01:
    print("⚠️  WARNING: Actions change very slowly!")
    print("   Consecutive frames have very similar actions")
    print("   Model may not see the effect of different actions")
else:
    print("✅ Action distribution looks reasonable")
    print("   The problem may be in training or model architecture")

print()
