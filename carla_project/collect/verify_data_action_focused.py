"""
验证动作控制优化版数据集

检查：
1. 动作-视觉变化的相关性
2. 不同驾驶模式的分布
3. 数据质量
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

DEFAULT_DATA_DIR = Path('../data/raw_action_correlated')
DEFAULT_MIN_SPEED = 1.0
DEFAULT_MIN_YAW_DIFF = 8.0
DEFAULT_MIN_IMAGE_DIFF = 0.02
DEFAULT_MIN_PASS_RATE = 0.8
DEFAULT_MAX_LAG = 3

BRANCH_NAMES = ['straight', 'left', 'right']


def compute_visual_change(img1, img2):
    """计算两帧之间的视觉变化（像素差异比例）"""
    diff = np.abs(img1.astype(float) - img2.astype(float))
    change = (diff > 30).mean()  # 超过30的像素差异
    return change


def load_episode_images(episode_dir, num_frames):
    images_dir = episode_dir / 'images'
    images = []
    for i in range(num_frames):
        img_path = images_dir / f'{i:04d}.png'
        if img_path.exists():
            images.append(cv2.imread(str(img_path)))
    return images


def compute_image_diff(images_a, images_b, start_idx):
    count = min(len(images_a), len(images_b)) - start_idx
    if count <= 0:
        return 0.0
    diffs = []
    for i in range(start_idx, start_idx + count):
        diff = np.mean(
            np.abs(images_a[i].astype(np.float32) - images_b[i].astype(np.float32))
        )
        diffs.append(diff / 255.0)
    return float(np.mean(diffs))


def compute_lagged_corr(action_changes, visual_changes, max_lag):
    min_len = min(len(action_changes), len(visual_changes))
    action_changes = action_changes[:min_len]
    visual_changes = visual_changes[:min_len]
    best_corr = -1.0
    best_lag = 0
    for lag in range(max_lag + 1):
        if lag == 0:
            a = action_changes
            v = visual_changes
        else:
            a = action_changes[:-lag]
            v = visual_changes[lag:]
        if len(a) < 2 or len(v) < 2:
            continue
        if np.std(a) < 1e-6 or np.std(v) < 1e-6:
            continue
        corr = float(np.corrcoef(a, v)[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_corr, best_lag


def compute_action_signal(actions):
    steering = np.abs(actions[:, 0])
    throttle = actions[:, 1]
    return steering * throttle


def analyze_episode(episode_dir):
    """分析单个episode"""
    # 加载动作
    actions = np.load(episode_dir / 'actions.npy')

    # 加载图像
    images_dir = episode_dir / 'images'
    images = []
    for i in range(len(actions)):
        img_path = images_dir / f'{i:04d}.png'
        if img_path.exists():
            img = cv2.imread(str(img_path))
            images.append(img)

    if len(images) < 2:
        return None

    # 计算动作变化
    action_changes = np.abs(np.diff(actions, axis=0)).sum(axis=1)
    action_signal = compute_action_signal(actions)[:-1]

    # 计算视觉变化
    visual_changes = []
    for i in range(len(images) - 1):
        change = compute_visual_change(images[i], images[i+1])
        visual_changes.append(change)

    visual_changes = np.array(visual_changes)

    # 对齐长度（图像缺失会导致不一致）
    min_len = min(len(action_changes), len(visual_changes), len(action_signal))
    action_changes = action_changes[:min_len]
    action_signal = action_signal[:min_len]
    visual_changes = visual_changes[:min_len]

    # 加载元数据
    metadata = {}
    meta_path = episode_dir / 'metadata.npy'
    if meta_path.exists():
        metadata = np.load(meta_path, allow_pickle=True).item()

    return {
        'episode_dir': episode_dir,
        'actions': actions,
        'action_changes': action_changes,
        'action_signal': action_signal,
        'visual_changes': visual_changes,
        'mode': metadata.get('mode'),
        'branch': metadata.get('branch'),
        'group_id': metadata.get('group_id'),
        'context_frames': metadata.get('context_frames'),
        'branch_frames': metadata.get('branch_frames'),
        'yaw_change_deg': metadata.get('yaw_change_deg'),
        'avg_speed_mps': metadata.get('avg_speed_mps'),
        'num_frames': len(images),
    }


def compute_branch_divergence(group_results, min_image_diff, min_yaw_diff, min_speed):
    diffs = []
    yaw_diffs = []
    speed_ok = []

    branch_data = {r['branch']: r for r in group_results if r['branch'] in BRANCH_NAMES}
    if not all(name in branch_data for name in BRANCH_NAMES):
        return None

    context_frames = branch_data[BRANCH_NAMES[0]].get('context_frames') or 0

    images = {}
    for name in BRANCH_NAMES:
        result = branch_data[name]
        images[name] = load_episode_images(result['episode_dir'], result['num_frames'])

    for i in range(len(BRANCH_NAMES)):
        for j in range(i + 1, len(BRANCH_NAMES)):
            a = BRANCH_NAMES[i]
            b = BRANCH_NAMES[j]
            diffs.append(compute_image_diff(images[a], images[b], context_frames))

    if all(branch_data[name].get('yaw_change_deg') is not None for name in BRANCH_NAMES):
        yaw_changes = {name: branch_data[name]['yaw_change_deg'] for name in BRANCH_NAMES}
        yaw_pairs = []
        for i in range(len(BRANCH_NAMES)):
            for j in range(i + 1, len(BRANCH_NAMES)):
                a = BRANCH_NAMES[i]
                b = BRANCH_NAMES[j]
                yaw_pairs.append(abs(yaw_changes[a] - yaw_changes[b]))
        if yaw_pairs:
            yaw_diffs.append(float(np.mean(yaw_pairs)))

    if all(branch_data[name].get('avg_speed_mps') is not None for name in BRANCH_NAMES):
        speeds = [branch_data[name]['avg_speed_mps'] for name in BRANCH_NAMES]
        speed_ok.append(all(speed >= min_speed for speed in speeds))

    mean_image_diff = float(np.mean(diffs)) if diffs else 0.0
    mean_yaw_diff = float(np.mean(yaw_diffs)) if yaw_diffs else 0.0

    return {
        'mean_image_diff': mean_image_diff,
        'mean_yaw_diff': mean_yaw_diff,
        'image_ok': mean_image_diff >= min_image_diff,
        'yaw_ok': mean_yaw_diff >= min_yaw_diff if yaw_diffs else False,
        'speed_ok': speed_ok[0] if speed_ok else False,
        'has_yaw': bool(yaw_diffs),
        'has_speed': bool(speed_ok),
    }


def main(data_dir, min_image_diff, min_yaw_diff, min_speed, min_pass_rate, max_lag):
    print("=" * 70)
    print("  验证动作控制优化版数据集")
    print("=" * 70)

    # 查找所有episodes
    episode_dirs = sorted(data_dir.glob('episode_*'))
    print(f"\n找到 {len(episode_dirs)} 个episodes")

    if len(episode_dirs) == 0:
        print("❌ 未找到数据！")
        return

    # 分析所有episodes
    print("\n分析数据...")
    all_results = []
    for episode_dir in tqdm(episode_dirs):
        result = analyze_episode(episode_dir)
        if result is not None:
            all_results.append(result)

    print(f"✓ 成功分析 {len(all_results)} 个episodes")

    # 统计分析
    print("\n" + "=" * 70)
    print("  数据统计")
    print("=" * 70)

    # 1. 动作分布
    all_actions = np.concatenate([r['actions'] for r in all_results])
    print(f"\n动作分布:")
    print(f"  Steering - Min: {all_actions[:, 0].min():.3f}, "
          f"Max: {all_actions[:, 0].max():.3f}, "
          f"Mean: {all_actions[:, 0].mean():.3f}, "
          f"Std: {all_actions[:, 0].std():.3f}")
    print(f"  Throttle - Min: {all_actions[:, 1].min():.3f}, "
          f"Max: {all_actions[:, 1].max():.3f}, "
          f"Mean: {all_actions[:, 1].mean():.3f}, "
          f"Std: {all_actions[:, 1].std():.3f}")
    abs_steer = np.abs(all_actions[:, 0])
    straight_ratio = np.mean(abs_steer < 0.1)
    turn_ratio = np.mean(abs_steer >= 0.3)
    mid_ratio = np.mean((abs_steer >= 0.3) & (abs_steer < 0.6))
    hard_ratio = np.mean(abs_steer >= 0.6)
    print(f"  Steering ratios: straight(<0.1)={straight_ratio:.1%}, "
          f"turn(>=0.3)={turn_ratio:.1%}, "
          f"mid(0.3-0.6)={mid_ratio:.1%}, hard(>=0.6)={hard_ratio:.1%}")

    # 2. 动作-视觉变化相关性
    all_action_changes = np.concatenate([r['action_changes'] for r in all_results])
    all_visual_changes = np.concatenate([r['visual_changes'] for r in all_results])

    print(f"\n动作变化统计:")
    print(f"  Mean: {all_action_changes.mean():.3f}")
    print(f"  Std: {all_action_changes.std():.3f}")
    print(f"  Max: {all_action_changes.max():.3f}")

    print(f"\n视觉变化统计:")
    print(f"  Mean: {all_visual_changes.mean():.3f}")
    print(f"  Std: {all_visual_changes.std():.3f}")
    print(f"  Max: {all_visual_changes.max():.3f}")

    has_branching = any(r['branch'] is not None for r in all_results)
    has_mode = any(r['mode'] is not None for r in all_results)

    all_action_signal = np.concatenate([r['action_signal'] for r in all_results])

    # 3. 相关性分析
    correlation = np.corrcoef(all_action_changes, all_visual_changes)[0, 1]
    lagged_corr, best_lag = compute_lagged_corr(all_action_changes, all_visual_changes, max_lag)
    signal_corr, signal_lag = compute_lagged_corr(all_action_signal, all_visual_changes, max_lag)
    if np.std(all_action_signal) < 1e-6 or np.std(all_visual_changes) < 1e-6:
        signal_corr0 = 0.0
    else:
        signal_corr0 = np.corrcoef(all_action_signal, all_visual_changes)[0, 1]
    print(f"\n动作变化-视觉相关性: {correlation:.4f}")
    print(f"动作变化-视觉滞后相关性: {lagged_corr:.4f} (lag={best_lag})")
    print(f"动作强度(|steer|*throttle)-视觉相关性: {signal_corr0:.4f}")
    print(f"动作强度-视觉滞后相关性: {signal_corr:.4f} (lag={signal_lag})")
    if has_branching:
        print("  注: Branching数据集不使用动作变化相关性作为质量判断")

    # 4. 按动作变化大小分组
    small_action = all_visual_changes[all_action_changes < 0.1]
    medium_action = all_visual_changes[(all_action_changes >= 0.1) & (all_action_changes < 0.5)]
    large_action = all_visual_changes[all_action_changes >= 0.5]

    print(f"\n动作变化 vs 视觉变化:")
    if len(small_action) > 0:
        print(f"  小动作变化 (<0.1): 视觉变化 {small_action.mean():.4f} ({len(small_action)} samples)")
    else:
        print("  小动作变化 (<0.1): N/A (0 samples)")
    if len(medium_action) > 0:
        print(f"  中动作变化 (0.1-0.5): 视觉变化 {medium_action.mean():.4f} ({len(medium_action)} samples)")
    else:
        print("  中动作变化 (0.1-0.5): N/A (0 samples)")
    if len(large_action) > 0:
        print(f"  大动作变化 (>=0.5): 视觉变化 {large_action.mean():.4f} ({len(large_action)} samples)")
    else:
        print("  大动作变化 (>=0.5): N/A (0 samples)")

    # 5. 驾驶模式或分支分布
    if has_branching:
        branch_counts = {}
        for r in all_results:
            branch = r['branch']
            if branch is None:
                continue
            branch_counts[branch] = branch_counts.get(branch, 0) + 1
        print(f"\n分支分布:")
        for branch, count in sorted(branch_counts.items()):
            print(f"  {branch}: {count} episodes")
    elif has_mode:
        mode_counts = {}
        for r in all_results:
            mode = r['mode']
            if mode is None:
                continue
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        print(f"\n驾驶模式分布:")
        mode_names = ['直行为主', '左转为主', '右转为主', '频繁变向', 'S型路线']
        for mode, count in sorted(mode_counts.items()):
            print(f"  模式{mode} ({mode_names[mode]}): {count} episodes")

    # 6. 分支一致性分析（仅对branching数据）
    branch_summary = None
    if has_branching:
        print("\n" + "=" * 70)
        print("  分支一致性分析")
        print("=" * 70)
        groups = {}
        for r in all_results:
            if r['group_id'] is None or r['branch'] is None:
                continue
            groups.setdefault(r['group_id'], []).append(r)

        group_metrics = []
        for results in groups.values():
            metrics = compute_branch_divergence(results, min_image_diff, min_yaw_diff, min_speed)
            if metrics is not None:
                group_metrics.append(metrics)

        if group_metrics:
            mean_image = np.mean([m['mean_image_diff'] for m in group_metrics])
            mean_yaw = np.mean([m['mean_yaw_diff'] for m in group_metrics if m['has_yaw']]) if any(m['has_yaw'] for m in group_metrics) else 0.0
            image_pass = sum(m['image_ok'] for m in group_metrics) / len(group_metrics)
            yaw_pass = sum(m['yaw_ok'] for m in group_metrics) / len(group_metrics)
            speed_pass = sum(m['speed_ok'] for m in group_metrics) / len(group_metrics) if any(m['has_speed'] for m in group_metrics) else 0.0

            print(f"Groups analyzed: {len(group_metrics)}")
            print(f"Mean pairwise image diff: {mean_image:.4f} (pass rate {image_pass:.1%})")
            if any(m['has_yaw'] for m in group_metrics):
                print(f"Mean pairwise yaw diff: {mean_yaw:.2f} deg (pass rate {yaw_pass:.1%})")
            if any(m['has_speed'] for m in group_metrics):
                print(f"Speed pass rate: {speed_pass:.1%}")
            branch_summary = {
                'image_pass': image_pass,
                'yaw_pass': yaw_pass,
                'speed_pass': speed_pass,
                'has_yaw': any(m['has_yaw'] for m in group_metrics),
                'has_speed': any(m['has_speed'] for m in group_metrics),
            }
        else:
            print("No complete branch groups found.")

    # 7. 生成可视化
    print("\n生成可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 动作分布
    ax = axes[0, 0]
    ax.hist2d(all_actions[:, 0], all_actions[:, 1], bins=50, cmap='viridis')
    ax.set_xlabel('Steering')
    ax.set_ylabel('Throttle')
    ax.set_title('Action Distribution')
    ax.grid(True, alpha=0.3)

    # 动作-视觉变化散点图
    ax = axes[0, 1]
    # 采样（太多点会很慢）
    sample_size = min(5000, len(all_action_changes))
    indices = np.random.choice(len(all_action_changes), sample_size, replace=False)
    ax.scatter(all_action_changes[indices], all_visual_changes[indices],
              alpha=0.3, s=1)
    ax.set_xlabel('Action Change')
    ax.set_ylabel('Visual Change')
    ax.set_title(f'Action-Visual Correlation (r={correlation:.3f})')
    ax.grid(True, alpha=0.3)

    # 动作变化分布
    ax = axes[1, 0]
    ax.hist(all_action_changes, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Action Change')
    ax.set_ylabel('Frequency')
    ax.set_title('Action Change Distribution')
    ax.grid(True, alpha=0.3)

    # 视觉变化分布
    ax = axes[1, 1]
    ax.hist(all_visual_changes, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.set_xlabel('Visual Change')
    ax.set_ylabel('Frequency')
    ax.set_title('Visual Change Distribution')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = data_dir / 'data_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化保存到: {output_path}")

    # 总结
    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)
    print(f"总帧数: {len(all_actions):,}")
    print(f"总episodes: {len(all_results)}")
    print(f"动作变化-视觉相关性: {correlation:.4f}")
    print(f"动作变化-视觉滞后相关性: {lagged_corr:.4f} (lag={best_lag})")
    print(f"动作强度(|steer|*throttle)-视觉相关性: {signal_corr0:.4f}")
    print(f"动作强度-视觉滞后相关性: {signal_corr:.4f} (lag={signal_lag})")

    if has_branching and branch_summary is not None:
        print("\n说明: Branching数据集不使用动作变化相关性作为质量判断")
        pass_checks = []
        pass_checks.append(branch_summary['image_pass'] >= min_pass_rate)
        if branch_summary['has_yaw']:
            pass_checks.append(branch_summary['yaw_pass'] >= min_pass_rate)
        if branch_summary['has_speed']:
            pass_checks.append(branch_summary['speed_pass'] >= min_pass_rate)

        if all(pass_checks):
            print("\n✅ 分支分离度达标，可用于训练")
        else:
            print("\n❌ 分支分离度不足，建议调整采集参数")
    else:
        # 优先使用动作强度相关性判断，适配分段恒定动作的采集策略
        if signal_corr0 >= 0.3 or signal_corr >= 0.3:
            print("\n✅ 数据质量良好！动作强度对视觉变化有明显影响")
            print("   可以开始训练World Model")
        elif correlation > 0.3:
            print("\n✅ 数据质量良好！动作变化对视觉变化有明显影响")
            print("   可以开始训练World Model")
        elif correlation > 0.1:
            print("\n⚠️  数据质量一般，建议:")
            print("   - 增加采样间隔（SAMPLE_INTERVAL）")
            print("   - 使用更大的转向角度")
        else:
            print("\n❌ 数据质量较差，建议重新采集")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify action-focused dataset")
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument('--min-speed', type=float, default=DEFAULT_MIN_SPEED)
    parser.add_argument('--min-yaw-diff', type=float, default=DEFAULT_MIN_YAW_DIFF)
    parser.add_argument('--min-image-diff', type=float, default=DEFAULT_MIN_IMAGE_DIFF)
    parser.add_argument('--min-pass-rate', type=float, default=DEFAULT_MIN_PASS_RATE)
    parser.add_argument('--max-lag', type=int, default=DEFAULT_MAX_LAG)
    args = parser.parse_args()
    main(
        Path(args.data_dir),
        args.min_image_diff,
        args.min_yaw_diff,
        args.min_speed,
        args.min_pass_rate,
        args.max_lag,
    )
