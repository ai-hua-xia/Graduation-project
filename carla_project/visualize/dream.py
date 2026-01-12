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

from models.vqvae import VQVAE
from models.world_model import WorldModel
from train.config import WM_CONFIG


def generate_video(vqvae, world_model, initial_frames, actions, device, output_path):
    """
    生成视频

    Args:
        vqvae: VQ-VAE模型
        world_model: World Model
        initial_frames: 初始帧tokens (context_frames, H, W)
        actions: 动作序列 (num_frames, action_dim)
        device: 设备
        output_path: 输出视频路径
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
            next_tokens = world_model.predict_next_frame(
                token_buffer,
                action_window,
                temperature=1.0,
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
    out = cv2.VideoWriter(temp_path, fourcc, 20.0, (256, 256))

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
    # VQ-VAE
    vqvae = VQVAE().to(device)
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
                        help='Action file (if not provided, use from token file)')
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

    # 动作序列
    if args.action_file:
        # 从文件加载动作
        actions = np.load(args.action_file)
    else:
        # 使用数据集中的动作
        actions = actions[context_frames:context_frames + args.num_frames]

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
        output_path
    )

    print(f"\nGenerated {len(frames)} frames")


if __name__ == '__main__':
    main()
