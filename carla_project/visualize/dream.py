"""
视频生成（Dream）脚本
"""

import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.vqvae_v2 import VQVAE_V2
from models.world_model import WorldModel
from train.config import WM_CONFIG


def wasd_to_action(key):
    """
    将WASD按键转换为动作向量（基于训练数据分布优化）

    训练数据统计：
    - Steering: [-0.6, 0.6], 均值0.006, 53%在[-0.2, 0.2]
    - Throttle: [0.4, 0.7], 均值0.549, 无刹车数据

    Args:
        key: 按键字符 ('W', 'A', 'S', 'D', 'N')

    Returns:
        action: [steering, throttle]
    """
    # 默认：直行 + 中等油门
    steering = 0.0
    throttle = 0.55

    key = key.upper()

    if key == 'W':  # 加速
        steering = 0.0
        throttle = 0.65  # 训练数据上限附近
    elif key == 'S':  # 减速（不是刹车！训练数据无刹车）
        steering = 0.0
        throttle = 0.42  # 训练数据下限附近
    elif key == 'A':  # 左转
        steering = -0.4  # 训练数据常见范围
        throttle = 0.55
    elif key == 'D':  # 右转
        steering = 0.4   # 训练数据常见范围
        throttle = 0.55
    elif key == 'Q':  # 左转+加速
        steering = -0.4
        throttle = 0.65
    elif key == 'E':  # 右转+加速
        steering = 0.4
        throttle = 0.65
    elif key == 'N':  # 空档（保持直行）
        steering = 0.0
        throttle = 0.55
    else:
        # 未知按键，返回默认
        pass

    return np.array([steering, throttle], dtype=np.float32)


def load_actions_from_txt(txt_path):
    """
    从文本文件加载动作序列

    支持两种格式：
    1. WASD格式（每行一个字母）：
       W
       W
       A
       D
       S

    2. 数值格式（每行两个数字：steering throttle）：
       0.0 0.65
       0.0 0.65
       -0.4 0.55
       0.4 0.55
       0.0 0.42

    Args:
        txt_path: 文本文件路径

    Returns:
        actions: numpy数组 (N, 2)
        keys: 按键字符列表 (N,) 或 None
    """
    actions = []
    keys = []
    is_wasd_format = False

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # 跳过空行和注释
                continue

            # 尝试解析为WASD
            if len(line) == 1 and line.upper() in 'WASDQEN':
                action = wasd_to_action(line)
                actions.append(action)
                keys.append(line.upper())
                is_wasd_format = True
            else:
                # 尝试解析为数值
                try:
                    parts = line.split()
                    if len(parts) == 2:
                        steering = float(parts[0])
                        throttle = float(parts[1])
                        actions.append([steering, throttle])
                        keys.append(None)
                    else:
                        print(f"Warning: 跳过无效行: {line}")
                except ValueError:
                    print(f"Warning: 跳过无效行: {line}")

    if len(actions) == 0:
        raise ValueError(f"未能从 {txt_path} 中解析出任何动作")

    return np.array(actions, dtype=np.float32), (keys if is_wasd_format else None)


def draw_wasd_overlay(frame, action, key_char=None):
    """
    在帧上绘制WASD按键指示器

    Args:
        frame: 图像帧 (H, W, 3) BGR格式
        action: 动作向量 [steering, throttle]
        key_char: 按键字符（如果有的话）

    Returns:
        frame: 添加了按键指示的帧
    """
    frame = frame.copy()
    h, w = frame.shape[:2]

    # 按键布局（右下角）
    # 位置：右下角，留出边距
    base_x = w - 120
    base_y = h - 120
    key_size = 30
    gap = 5

    # 定义按键位置
    keys = {
        'W': (base_x + key_size + gap, base_y),
        'A': (base_x, base_y + key_size + gap),
        'S': (base_x + key_size + gap, base_y + key_size + gap),
        'D': (base_x + 2 * (key_size + gap), base_y + key_size + gap),
        'Q': (base_x, base_y),
        'E': (base_x + 2 * (key_size + gap), base_y),
    }

    # 根据动作判断按下的键
    steering, throttle = action
    active_key = None

    if key_char:
        active_key = key_char.upper()
    else:
        # 根据动作推断按键
        if abs(steering) < 0.1:  # 直行
            if throttle > 0.6:
                active_key = 'W'
            elif throttle < 0.45:
                active_key = 'S'
        elif steering < -0.2:  # 左转
            if throttle > 0.6:
                active_key = 'Q'
            else:
                active_key = 'A'
        elif steering > 0.2:  # 右转
            if throttle > 0.6:
                active_key = 'E'
            else:
                active_key = 'D'

    # 绘制所有按键
    for key, (x, y) in keys.items():
        if key == active_key:
            # 激活状态：亮绿色
            color = (0, 255, 0)
            thickness = -1  # 填充
            text_color = (0, 0, 0)
        else:
            # 未激活状态：深灰色边框
            color = (100, 100, 100)
            thickness = 2
            text_color = (150, 150, 150)

        # 绘制按键背景
        cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), color, thickness)

        # 绘制按键字母
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size = cv2.getTextSize(key, font, font_scale, font_thickness)[0]
        text_x = x + (key_size - text_size[0]) // 2
        text_y = y + (key_size + text_size[1]) // 2
        cv2.putText(frame, key, (text_x, text_y), font, font_scale, text_color, font_thickness)

    # 添加标签
    label = "Controls"
    cv2.putText(frame, label, (base_x, base_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 显示当前动作值（左上角）
    info_text = f"Steer: {steering:+.2f}  Throttle: {throttle:.2f}"
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def generate_video(vqvae, world_model, initial_frames, actions, device, output_path, fps=10,
                   show_controls=False, action_keys=None):
    """
    生成视频

    Args:
        vqvae: VQ-VAE模型
        world_model: World Model
        initial_frames: 初始帧tokens (context_frames, H, W)
        actions: 动作序列 (num_frames, action_dim)
        device: 设备
        output_path: 输出视频路径
        fps: 视频帧率 (default: 10)
        show_controls: 是否显示按键指示器 (default: False)
        action_keys: 按键字符列表，与actions对应 (optional)
    """
    vqvae.eval()
    world_model.eval()

    num_frames = len(actions)
    context_frames = world_model.context_frames

    # 初始化token buffer
    token_buffer = torch.from_numpy(initial_frames).unsqueeze(0).to(device)  # (1, T, H, W)

    frames = []

    print(f"\nGenerating {num_frames} frames...")

    with torch.no_grad():
        for t in range(num_frames):
            # 当前动作窗口
            if t < context_frames:
                action_window = np.zeros((1, context_frames, actions.shape[-1]))
                action_window[0, -t-1:] = actions[:t+1]
            else:
                action_window = actions[t-context_frames+1:t+1].reshape(1, context_frames, -1)

            action_window = torch.from_numpy(action_window).float().to(device)

            # 预测下一帧
            # 使用较低温度和top-k采样减少随机性，提高稳定性
            next_tokens = world_model.predict_next_frame(
                token_buffer,
                action_window,
                temperature=0.7,  # 降低温度，减少随机性
                top_k=50,  # top-k采样，过滤低概率token
            )  # (1, H, W)

            # 解码为图像
            frame = vqvae.decode_tokens(next_tokens)  # (1, 3, 256, 256)

            # 转换为numpy
            frame = frame.squeeze(0).cpu().numpy()  # (3, 256, 256)
            frame = (frame + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)
            frame = np.transpose(frame, (1, 2, 0))  # (H, W, 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 添加按键指示器
            if show_controls:
                key_char = action_keys[t] if action_keys and t < len(action_keys) else None
                frame = draw_wasd_overlay(frame, actions[t], key_char)

            frames.append(frame)

            # 更新buffer
            token_buffer = torch.cat([
                token_buffer[:, 1:],
                next_tokens.unsqueeze(1)
            ], dim=1)

            if (t + 1) % 50 == 0:
                print(f"Generated {t+1}/{num_frames} frames")

    # 保存视频（先用临时文件，再转H.264）
    import subprocess
    import tempfile

    temp_path = str(output_path) + '.temp.mp4'
    print(f"\nSaving video to {output_path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, float(fps), (256, 256))

    for frame in frames:
        out.write(frame)

    out.release()

    # 转换为H.264格式
    print("Converting to H.264...")
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        str(output_path)
    ], capture_output=True)

    # 删除临时文件
    Path(temp_path).unlink(missing_ok=True)
    print(f"Video saved!")

    return frames


def load_models(vqvae_path, world_model_path, device):
    """加载模型"""
    # VQ-VAE v2
    vqvae = VQVAE_V2().to(device)
    checkpoint = torch.load(vqvae_path, map_location=device)
    vqvae.load_state_dict(checkpoint['model_state_dict'])

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
        use_memory=config.get('use_memory', False),
        memory_dim=config.get('memory_dim', 256),
        dropout=config['dropout'],
    ).to(device)

    checkpoint = torch.load(world_model_path, map_location=device)
    world_model.load_state_dict(checkpoint['model_state_dict'])

    return vqvae, world_model


def main():
    parser = argparse.ArgumentParser(description='Generate video with World Model')
    parser.add_argument('--vqvae-checkpoint', type=str, required=True,
                        help='Path to VQ-VAE checkpoint')
    parser.add_argument('--world-model-checkpoint', type=str, required=True,
                        help='Path to World Model checkpoint')
    parser.add_argument('--token-file', type=str, required=True,
                        help='Path to token file (for initial frames)')
    parser.add_argument('--output', type=str, default='../outputs/dream_result.mp4',
                        help='Output video path')
    parser.add_argument('--num-frames', type=int, default=300,
                        help='Number of frames to generate')
    parser.add_argument('--action-file', type=str, default=None,
                        help='Action file (.npy or .txt, if not provided, use from token file)')
    parser.add_argument('--action-txt', type=str, default=None,
                        help='Action text file (WASD or numeric format)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Video FPS (default: 10)')
    parser.add_argument('--show-controls', action='store_true',
                        help='Show WASD control overlay on video')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

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

    # 加载初始帧和动作
    print("\nLoading data...")
    data = np.load(args.token_file)
    tokens = data['tokens']  # (N, H, W)
    actions = data['actions']  # (N, action_dim)

    context_frames = world_model.context_frames

    # 使用前context_frames作为初始帧
    initial_frames = tokens[:context_frames]

    # 动作序列（优先级：action-txt > action-file > token file）
    action_keys = None
    if args.action_txt:
        # 从文本文件加载动作（支持WASD和数值格式）
        print(f"Loading actions from text file: {args.action_txt}")
        actions, action_keys = load_actions_from_txt(args.action_txt)
        print(f"Loaded {len(actions)} actions from text file")
    elif args.action_file:
        # 从.npy文件加载动作
        print(f"Loading actions from .npy file: {args.action_file}")
        actions = np.load(args.action_file)
    else:
        # 使用数据集中的动作
        actions = actions[context_frames:context_frames + args.num_frames]

    # 如果动作数量超过num_frames，截断
    if len(actions) > args.num_frames:
        actions = actions[:args.num_frames]
        if action_keys:
            action_keys = action_keys[:args.num_frames]
        print(f"Truncated actions to {args.num_frames} frames")

    print(f"Initial frames shape: {initial_frames.shape}")
    print(f"Actions shape: {actions.shape}")

    # 生成视频
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = generate_video(
        vqvae,
        world_model,
        initial_frames,
        actions,
        device,
        output_path,
        fps=args.fps,
        show_controls=args.show_controls,
        action_keys=action_keys
    )

    print(f"\nGenerated {len(frames)} frames")


if __name__ == '__main__':
    main()
