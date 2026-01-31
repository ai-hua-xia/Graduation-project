"""
生成World Model预测视频

从真实数据开始，使用模型自回归生成未来帧
"""
import torch
import numpy as np
import cv2
from pathlib import Path
import sys
from tqdm import tqdm
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from models.vqvae_v2 import load_vqvae_v2_checkpoint
from models.world_model import WorldModel
from train.config import WM_CONFIG


def wasd_to_action(key):
    """Map WASD-like keys to action vectors."""
    steering = 0.0
    throttle = 0.55

    key = key.upper()
    if key == 'W':
        steering = 0.0
        throttle = 0.65
    elif key == 'S':
        steering = 0.0
        throttle = 0.42
    elif key == 'A':
        steering = -0.4
        throttle = 0.55
    elif key == 'D':
        steering = 0.4
        throttle = 0.55
    elif key == 'Q':
        steering = -0.4
        throttle = 0.65
    elif key == 'E':
        steering = 0.4
        throttle = 0.65
    elif key == 'N':
        steering = 0.0
        throttle = 0.55

    return np.array([steering, throttle], dtype=np.float32)


def load_actions_from_txt(txt_path):
    """
    Load actions from a txt file.

    Supported formats:
    1) WASD-like keys, one per line (W/A/S/D/Q/E/N)
    2) Numeric lines: "<steer> <throttle>"
    """
    actions = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if len(line) == 1 and line.upper() in 'WASDQEN':
                actions.append(wasd_to_action(line))
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                steering = float(parts[0])
                throttle = float(parts[1])
            except ValueError:
                continue
            actions.append([steering, throttle])
    if len(actions) == 0:
        raise ValueError(f"No valid actions found in {txt_path}")
    return np.array(actions, dtype=np.float32)


def find_best_action_windows(actions, target_actions, steer_weight=1.0, throttle_weight=1.0,
                             stride=1, topk=1):
    """Find top-k windows that best match the target actions (lower is better)."""
    if topk < 1:
        raise ValueError("topk must be >= 1")
    total = len(actions)
    window = len(target_actions)
    if window > total:
        raise ValueError("Target actions longer than dataset actions.")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    scores = []
    for i in range(0, total - window + 1, stride):
        diff = actions[i:i + window] - target_actions
        diff[:, 0] *= steer_weight
        diff[:, 1] *= throttle_weight
        score = float(np.mean(diff * diff))
        scores.append((score, i))
    scores.sort(key=lambda x: x[0])
    return scores[:topk]


def load_models(vqvae_path, wm_path, device='cuda', num_embeddings=None):
    """加载模型"""
    print("Loading models...")

    # VQ-VAE
    vqvae, _ = load_vqvae_v2_checkpoint(vqvae_path, device)
    vqvae.eval()

    # World Model
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
        use_memory=config.get('use_memory', False),
        memory_dim=config.get('memory_dim', 256),
        dropout=config['dropout'],
    ).to(device)

    checkpoint = torch.load(wm_path, map_location=device, weights_only=False)
    world_model.load_state_dict(checkpoint['model_state_dict'])
    world_model.eval()

    return vqvae, world_model


def tokens_to_image(vqvae, tokens, device):
    """将tokens解码为图像"""
    with torch.no_grad():
        # tokens: (H, W) 或 (256,) -> (1, H, W)
        if tokens.ndim == 1:
            # 如果是展平的，reshape回(H, W)
            h = w = int(np.sqrt(len(tokens)))
            tokens = tokens.reshape(h, w)

        tokens_tensor = torch.from_numpy(tokens).long().unsqueeze(0).to(device)

        # 解码
        frame = vqvae.decode_tokens(tokens_tensor)

        # 转换为图像 (C, H, W) -> (H, W, C)
        frame = frame.squeeze(0).cpu().numpy()
        frame = (frame + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        frame = np.transpose(frame, (1, 2, 0))  # (C, H, W) -> (H, W, C)

        return frame


def generate_video(vqvae, world_model, tokens, actions, start_idx, num_frames, device,
                   use_gt_actions=True, temperature=1.0, action_type='straight',
                   action_override=None):
    """
    生成预测视频

    Args:
        vqvae: VQ-VAE模型
        world_model: World Model
        tokens: 完整的token序列
        actions: 完整的动作序列
        start_idx: 起始索引
        num_frames: 生成帧数
        device: 设备
        use_gt_actions: 是否使用真实动作（True）还是固定动作（False）
        temperature: 采样温度
        action_type: 固定动作类型 ('straight', 'left', 'right', 'smooth')
        action_override: Optional custom action sequence (num_frames + context)

    Returns:
        pred_frames: 预测的帧列表
        gt_frames: 真实的帧列表
    """
    context_frames = world_model.context_frames

    # 初始化上下文
    context_tokens = tokens[start_idx:start_idx+context_frames].copy()

    pred_frames = []
    gt_frames = []

    # 先解码初始上下文帧
    for i in range(context_frames):
        frame = tokens_to_image(vqvae, context_tokens[i], device)
        pred_frames.append(frame)
        gt_frames.append(frame)

    # 自回归生成
    with torch.no_grad():
        memory = None
        use_memory = getattr(world_model, 'use_memory', False)
        for t in tqdm(range(num_frames), desc="Generating frames"):
            # 准备输入
            context_tensor = torch.from_numpy(context_tokens).long().unsqueeze(0).to(device)

            # 获取动作
            if action_override is not None:
                action_seq = action_override[t:t + context_frames]
            elif use_gt_actions:
                action_seq = actions[start_idx+t:start_idx+t+context_frames]
            else:
                # 使用固定动作
                if action_type == 'straight':
                    # 直行
                    action_seq = np.tile(np.array([0.0, 0.5], dtype=np.float32), (context_frames, 1))
                elif action_type == 'left':
                    # 左转
                    action_seq = np.tile(np.array([-0.3, 0.5], dtype=np.float32), (context_frames, 1))
                elif action_type == 'right':
                    # 右转
                    action_seq = np.tile(np.array([0.3, 0.5], dtype=np.float32), (context_frames, 1))
                elif action_type == 'smooth':
                    # 平滑变化：先直行，然后缓慢左转，再缓慢右转
                    progress = t / num_frames
                    if progress < 0.33:
                        steering = 0.0
                    elif progress < 0.66:
                        steering = -0.3 * ((progress - 0.33) / 0.33)
                    else:
                        steering = -0.3 + 0.6 * ((progress - 0.66) / 0.34)
                    action_seq = np.tile(np.array([steering, 0.5], dtype=np.float32), (context_frames, 1))
                else:
                    action_seq = np.tile(np.array([0.0, 0.5], dtype=np.float32), (context_frames, 1))

            action_tensor = torch.from_numpy(action_seq).float().unsqueeze(0).to(device)

            # 预测下一帧
            if use_memory:
                logits, memory = world_model(
                    context_tensor, action_tensor, memory=memory, return_memory=True
                )
            else:
                logits = world_model(context_tensor, action_tensor)

            # 采样或贪婪选择
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                pred_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.shape[:-1])
            else:
                pred_tokens = torch.argmax(logits, dim=-1)

            pred_tokens = pred_tokens.squeeze(0).cpu().numpy()  # (256,) 展平的tokens

            # 解码预测帧
            pred_frame = tokens_to_image(vqvae, pred_tokens, device)
            pred_frames.append(pred_frame)

            # 获取真实帧
            gt_idx = start_idx + context_frames + t
            if gt_idx < len(tokens):
                gt_frame = tokens_to_image(vqvae, tokens[gt_idx], device)
                gt_frames.append(gt_frame)
            else:
                gt_frames.append(np.zeros_like(pred_frame))

            # 更新上下文（滑动窗口）
            # 需要将pred_tokens reshape回(H, W)
            h = w = int(np.sqrt(len(pred_tokens)))
            pred_tokens_2d = pred_tokens.reshape(h, w)

            context_tokens = np.roll(context_tokens, -1, axis=0)
            context_tokens[-1] = pred_tokens_2d

    return pred_frames, gt_frames


def generate_video_hybrid(vqvae, world_model, tokens, actions, start_idx, num_frames, device,
                          reset_every=12, use_gt_actions=True, temperature=1.0,
                          action_type='straight', action_override=None):
    """
    生成混合视频：周期性重置上下文为真实token，减少漂移。
    """
    context_frames = world_model.context_frames

    # 初始化上下文
    context_tokens = tokens[start_idx:start_idx+context_frames].copy()

    pred_frames = []
    gt_frames = []

    # 先解码初始上下文帧
    for i in range(context_frames):
        frame = tokens_to_image(vqvae, context_tokens[i], device)
        pred_frames.append(frame)
        gt_frames.append(frame)

    # 自回归生成
    with torch.no_grad():
        memory = None
        use_memory = getattr(world_model, 'use_memory', False)
        for t in tqdm(range(num_frames), desc="Generating frames"):
            if reset_every > 0 and t > 0 and (t % reset_every) == 0:
                reset_start = start_idx + t - context_frames
                reset_end = reset_start + context_frames
                if reset_start < 0 or reset_end > len(tokens):
                    break
                context_tokens = tokens[reset_start:reset_end].copy()
                if use_memory:
                    memory = None

            # 准备输入
            context_tensor = torch.from_numpy(context_tokens).long().unsqueeze(0).to(device)

            # 获取动作
            if action_override is not None:
                action_seq = action_override[t:t + context_frames]
            elif use_gt_actions:
                action_seq = actions[start_idx+t:start_idx+t+context_frames]
            else:
                # 使用固定动作
                if action_type == 'straight':
                    action_seq = np.tile(np.array([0.0, 0.5], dtype=np.float32), (context_frames, 1))
                elif action_type == 'left':
                    action_seq = np.tile(np.array([-0.3, 0.5], dtype=np.float32), (context_frames, 1))
                elif action_type == 'right':
                    action_seq = np.tile(np.array([0.3, 0.5], dtype=np.float32), (context_frames, 1))
                elif action_type == 'smooth':
                    progress = t / num_frames
                    if progress < 0.33:
                        steering = 0.0
                    elif progress < 0.66:
                        steering = -0.3 * ((progress - 0.33) / 0.33)
                    else:
                        steering = -0.3 + 0.6 * ((progress - 0.66) / 0.34)
                    action_seq = np.tile(np.array([steering, 0.5], dtype=np.float32), (context_frames, 1))
                else:
                    action_seq = np.tile(np.array([0.0, 0.5], dtype=np.float32), (context_frames, 1))

            action_tensor = torch.from_numpy(action_seq).float().unsqueeze(0).to(device)

            # 预测下一帧
            if use_memory:
                logits, memory = world_model(
                    context_tensor, action_tensor, memory=memory, return_memory=True
                )
            else:
                logits = world_model(context_tensor, action_tensor)

            # 采样或贪婪选择
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                pred_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.shape[:-1])
            else:
                pred_tokens = torch.argmax(logits, dim=-1)

            pred_tokens = pred_tokens.squeeze(0).cpu().numpy()

            # 解码预测帧
            pred_frame = tokens_to_image(vqvae, pred_tokens, device)
            pred_frames.append(pred_frame)

            # 获取真实帧
            gt_idx = start_idx + context_frames + t
            if gt_idx < len(tokens):
                gt_frame = tokens_to_image(vqvae, tokens[gt_idx], device)
                gt_frames.append(gt_frame)
            else:
                gt_frames.append(np.zeros_like(pred_frame))

            # 更新上下文（滑动窗口）
            h = w = int(np.sqrt(len(pred_tokens)))
            pred_tokens_2d = pred_tokens.reshape(h, w)
            context_tokens = np.roll(context_tokens, -1, axis=0)
            context_tokens[-1] = pred_tokens_2d

    return pred_frames, gt_frames


def create_comparison_video(pred_frames, gt_frames, output_path, fps=10):
    """创建对比视频（预测 vs 真实）"""
    import subprocess
    import tempfile

    if len(pred_frames) == 0:
        print("No frames to save!")
        return

    h, w = pred_frames[0].shape[:2]

    # 先用mp4v编码生成临时文件
    temp_path = output_path.parent / f"temp_{output_path.name}"

    # 创建视频写入器（使用mp4v）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (w*2, h))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {temp_path}")
        return

    for pred, gt in zip(pred_frames, gt_frames):
        # 添加标签
        pred_labeled = pred.copy()
        gt_labeled = gt.copy()

        cv2.putText(pred_labeled, 'Predicted', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(gt_labeled, 'Ground Truth', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 左右拼接
        combined = np.hstack([pred_labeled, gt_labeled])

        # 转换为BGR（OpenCV格式）
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        out.write(combined_bgr)

    out.release()

    # 使用FFmpeg转换为H.264
    try:
        cmd = [
            'ffmpeg', '-y',  # 覆盖输出文件
            '-i', str(temp_path),  # 输入文件
            '-c:v', 'libx264',  # H.264编码
            '-preset', 'medium',  # 编码速度
            '-crf', '23',  # 质量（18-28，越小越好）
            '-pix_fmt', 'yuv420p',  # 像素格式（兼容性）
            '-loglevel', 'error',  # 只显示错误
            str(output_path)
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        # 删除临时文件
        temp_path.unlink()

        print(f"Video saved to: {output_path} (H.264)")

    except subprocess.CalledProcessError as e:
        print(f"Warning: FFmpeg conversion failed, keeping mp4v version")
        print(f"Error: {e.stderr.decode()}")
        # 如果转换失败，重命名临时文件为最终文件
        temp_path.rename(output_path)
        print(f"Video saved to: {output_path} (mp4v)")

    except FileNotFoundError:
        print("Warning: FFmpeg not found, keeping mp4v version")
        temp_path.rename(output_path)
        print(f"Video saved to: {output_path} (mp4v)")


def create_prediction_only_video(pred_frames, output_path, fps=10):
    """创建纯预测视频（只显示预测帧）"""
    import subprocess
    import tempfile

    if len(pred_frames) == 0:
        print("No frames to save!")
        return

    h, w = pred_frames[0].shape[:2]

    # 先用mp4v编码生成临时文件
    temp_path = output_path.parent / f"temp_{output_path.name}"

    # 创建视频写入器（使用mp4v）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {temp_path}")
        return

    for pred in pred_frames:
        # 添加标签
        pred_labeled = pred.copy()
        cv2.putText(pred_labeled, 'World Model Prediction', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 转换为BGR（OpenCV格式）
        pred_bgr = cv2.cvtColor(pred_labeled, cv2.COLOR_RGB2BGR)

        out.write(pred_bgr)

    out.release()

    # 使用FFmpeg转换为H.264
    try:
        cmd = [
            'ffmpeg', '-y',  # 覆盖输出文件
            '-i', str(temp_path),  # 输入文件
            '-c:v', 'libx264',  # H.264编码
            '-preset', 'medium',  # 编码速度
            '-crf', '23',  # 质量（18-28，越小越好）
            '-pix_fmt', 'yuv420p',  # 像素格式（兼容性）
            '-loglevel', 'error',  # 只显示错误
            str(output_path)
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        # 删除临时文件
        temp_path.unlink()

        print(f"Video saved to: {output_path} (H.264)")

    except subprocess.CalledProcessError as e:
        print(f"Warning: FFmpeg conversion failed, keeping mp4v version")
        print(f"Error: {e.stderr.decode()}")
        # 如果转换失败，重命名临时文件为最终文件
        temp_path.rename(output_path)
        print(f"Video saved to: {output_path} (mp4v)")

    except FileNotFoundError:
        print("Warning: FFmpeg not found, keeping mp4v version")
        temp_path.rename(output_path)
        print(f"Video saved to: {output_path} (mp4v)")


def main():
    parser = argparse.ArgumentParser(description='Generate World Model prediction videos')
    parser.add_argument('--mode', type=str, default='predict',
                       choices=['predict', 'retrieve', 'hybrid'],
                       help='predict: WM autoregressive; retrieve: best-match replay; hybrid: periodic reset')
    parser.add_argument('--vqvae-checkpoint', type=str,
                       default='checkpoints/vqvae/vqvae_v2/best.pth')
    parser.add_argument('--world-model-checkpoint', type=str,
                       default='checkpoints/wm_ss/world_model_v5_ss/best.pth')
    parser.add_argument('--token-file', type=str,
                       default='data/tokens_raw/tokens_actions.npz')
    parser.add_argument('--output-dir', type=str, default='results/videos')
    parser.add_argument('--num-videos', type=int, default=5,
                       help='Number of videos to generate')
    parser.add_argument('--num-frames', type=int, default=32,
                       help='Number of frames to predict')
    parser.add_argument('--fps', type=int, default=10,
                       help='Video FPS')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (0 for greedy)')
    parser.add_argument('--start-idx', type=int, default=None,
                       help='Fixed start index (default: random)')
    parser.add_argument('--prediction-only', action='store_true',
                       help='Only show prediction (no ground truth comparison)')
    parser.add_argument('--fixed-action', action='store_true',
                       help='Use fixed smooth action sequence instead of GT actions')
    parser.add_argument('--action-type', type=str, default='straight',
                       choices=['straight', 'left', 'right', 'smooth'],
                       help='Type of fixed action: straight/left/right/smooth')
    parser.add_argument('--action-txt', type=str, default=None,
                       help='Action text file (WASD or numeric format)')
    parser.add_argument('--hybrid-reset-every', type=int, default=12,
                       help='Reset context every N frames (hybrid mode)')
    parser.add_argument('--match-steer-weight', type=float, default=1.0,
                       help='Steering weight for action matching (retrieve mode)')
    parser.add_argument('--match-throttle-weight', type=float, default=1.0,
                       help='Throttle weight for action matching (retrieve mode)')
    parser.add_argument('--match-stride', type=int, default=1,
                       help='Stride for action matching window scan')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("  World Model Video Generation")
    print("="*70)
    print()

    # 加载数据
    print("Loading data...")
    data = np.load(args.token_file)
    tokens = data['tokens']
    actions = data['actions']
    num_embeddings = int(tokens.max()) + 1
    print(f"Loaded {len(tokens)} frames")
    print(f"Num embeddings: {num_embeddings}")
    print()

    # 加载模型
    if args.mode in ('predict', 'hybrid'):
        vqvae, world_model = load_models(
            args.vqvae_checkpoint,
            args.world_model_checkpoint,
            args.device,
            num_embeddings=num_embeddings,
        )
        context_frames = world_model.context_frames
    else:
        vqvae, _ = load_vqvae_v2_checkpoint(args.vqvae_checkpoint, args.device)
        vqvae.eval()
        world_model = None
        context_frames = WM_CONFIG['context_frames']

    max_start_idx = len(tokens) - context_frames - args.num_frames

    action_override = None
    if args.action_txt:
        action_override = load_actions_from_txt(args.action_txt)
        if action_override.ndim != 2 or action_override.shape[1] != 2:
            raise ValueError(f"Invalid action shape from {args.action_txt}: {action_override.shape}")
        required = args.num_frames + context_frames
        if len(action_override) < required:
            pad = np.repeat(action_override[-1][None], required - len(action_override), axis=0)
            action_override = np.concatenate([action_override, pad], axis=0)
        if args.fixed_action:
            print("Warning: --action-txt provided, ignoring --fixed-action.")
        if not args.prediction_only and args.mode == 'predict':
            print("Warning: custom actions provided; GT comparison may be misleading.")

    # 生成多个视频
    print(f"Generating {args.num_videos} videos...")
    print()

    # 确定起始位置
    match_scores = None
    if args.mode in ('retrieve', 'hybrid'):
        if args.start_idx is None and action_override is None:
            raise ValueError("retrieve/hybrid mode requires --action-txt or --start-idx")
        if args.start_idx is not None:
            start_indices = [args.start_idx] * args.num_videos
            print(f"Using fixed start index: {args.start_idx}")
        else:
            matches = find_best_action_windows(
                actions,
                action_override[:args.num_frames + context_frames],
                steer_weight=args.match_steer_weight,
                throttle_weight=args.match_throttle_weight,
                stride=args.match_stride,
                topk=args.num_videos,
            )
            start_indices = [idx for _, idx in matches]
            match_scores = [score for score, _ in matches]
            print("Using best-matched start indices:", start_indices)
    else:
        if args.start_idx is not None:
            # 使用固定起始位置
            if args.start_idx < 0 or args.start_idx > max_start_idx:
                print(f"Warning: start_idx {args.start_idx} out of range [0, {max_start_idx}]")
                print(f"Using random start index instead")
                start_indices = [np.random.randint(0, max_start_idx) for _ in range(args.num_videos)]
            else:
                # 所有视频使用相同的起始位置
                start_indices = [args.start_idx] * args.num_videos
                print(f"Using fixed start index: {args.start_idx}")
        else:
            # 随机选择起始位置
            start_indices = [np.random.randint(0, max_start_idx) for _ in range(args.num_videos)]
            print(f"Using random start indices")
    print()

    for i in range(args.num_videos):
        start_idx = start_indices[i]

        print(f"Video {i+1}/{args.num_videos} (starting from frame {start_idx})...")
        if match_scores is not None:
            print(f"  Match score: {match_scores[i]:.6f}")

        if args.mode == 'retrieve':
            total_frames = context_frames + args.num_frames
            pred_frames = []
            for t in range(total_frames):
                frame_idx = start_idx + t
                if frame_idx >= len(tokens):
                    break
                pred_frames.append(tokens_to_image(vqvae, tokens[frame_idx], args.device))
            output_path = output_dir / f'retrieval_{i+1:02d}_idx{start_idx}.mp4'
            create_prediction_only_video(pred_frames, output_path, args.fps)
            print()
            continue

        if args.mode == 'hybrid':
            pred_frames, gt_frames = generate_video_hybrid(
                vqvae, world_model, tokens, actions,
                start_idx, args.num_frames, args.device,
                reset_every=args.hybrid_reset_every,
                use_gt_actions=not args.fixed_action,
                temperature=args.temperature,
                action_type=args.action_type,
                action_override=action_override
            )
        else:
            pred_frames, gt_frames = generate_video(
                vqvae, world_model, tokens, actions,
                start_idx, args.num_frames, args.device,
                use_gt_actions=not args.fixed_action,
                temperature=args.temperature,
                action_type=args.action_type,
                action_override=action_override
            )

        # 保存视频
        output_path = output_dir / f'prediction_{i+1:02d}.mp4'

        if args.prediction_only or args.mode == 'hybrid':
            # 纯预测模式：只显示预测帧
            create_prediction_only_video(pred_frames, output_path, args.fps)
        else:
            # 对比模式：左预测，右真实
            create_comparison_video(pred_frames, gt_frames, output_path, args.fps)
        print()

    print("="*70)
    print(f"✅ All videos saved to: {output_dir}/")
    print("="*70)
    print()
    print("Generated files:")
    for video_file in sorted(output_dir.glob('*.mp4')):
        size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"  {video_file.name} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
