"""
生成对比视频：真实帧 vs 预测帧
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


def generate_comparison_video(vqvae, world_model, tokens, actions, device, output_path, num_frames=200):
    """生成对比视频"""
    vqvae.eval()
    world_model.eval()

    context_frames = world_model.context_frames

    # 初始化token buffer
    token_buffer = torch.from_numpy(tokens[:context_frames]).unsqueeze(0).to(device)

    frames = []

    print(f"\nGenerating {num_frames} comparison frames...")

    with torch.no_grad():
        for t in range(num_frames):
            # 获取真实帧tokens
            real_token = tokens[context_frames + t]

            # 构建动作窗口
            action_idx = context_frames + t
            if action_idx >= context_frames:
                action_window = actions[action_idx - context_frames + 1:action_idx + 1]
            else:
                action_window = np.zeros((context_frames, actions.shape[-1]))
                action_window[-action_idx-1:] = actions[:action_idx+1]

            action_window = torch.from_numpy(action_window).float().unsqueeze(0).to(device)

            # 预测下一帧
            pred_tokens = world_model.predict_next_frame(
                token_buffer,
                action_window,
                temperature=0.8,  # 稍低温度，更确定性
            )

            # 解码真实帧
            real_token_tensor = torch.from_numpy(real_token).unsqueeze(0).to(device)
            real_frame = vqvae.decode_tokens(real_token_tensor)
            real_frame = real_frame.squeeze(0).cpu().numpy()
            real_frame = (real_frame + 1.0) / 2.0
            real_frame = np.clip(real_frame, 0, 1)
            real_frame = (real_frame * 255).astype(np.uint8)
            real_frame = np.transpose(real_frame, (1, 2, 0))

            # 解码预测帧
            pred_frame = vqvae.decode_tokens(pred_tokens)
            pred_frame = pred_frame.squeeze(0).cpu().numpy()
            pred_frame = (pred_frame + 1.0) / 2.0
            pred_frame = np.clip(pred_frame, 0, 1)
            pred_frame = (pred_frame * 255).astype(np.uint8)
            pred_frame = np.transpose(pred_frame, (1, 2, 0))

            # 并排显示
            combined = np.hstack([real_frame, pred_frame])

            # 添加标签
            combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            cv2.putText(combined, 'Real', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, 'Predicted', (266, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示动作信息
            steer, throttle = actions[action_idx]
            cv2.putText(combined, f'Steer: {steer:.2f}', (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(combined, f'Throttle: {throttle:.2f}', (130, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            frames.append(combined)

            # 更新buffer（使用真实token，而不是预测的）
            real_token_tensor = torch.from_numpy(real_token).unsqueeze(0).unsqueeze(0).to(device)
            token_buffer = torch.cat([token_buffer[:, 1:], real_token_tensor], dim=1)

            if (t + 1) % 50 == 0:
                print(f"Generated {t+1}/{num_frames} frames")

    # 保存视频（先用临时文件，再转H.264）
    import subprocess

    temp_path = str(output_path) + '.temp.mp4'
    print(f"\nSaving video to {output_path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, 20.0, (512, 256))

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


def main():
    parser = argparse.ArgumentParser(description='Generate comparison video')
    parser.add_argument('--vqvae-checkpoint', type=str, required=True)
    parser.add_argument('--world-model-checkpoint', type=str, required=True)
    parser.add_argument('--token-file', type=str, required=True)
    parser.add_argument('--output', type=str, default='../outputs/comparison.mp4')
    parser.add_argument('--num-frames', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    print("\nLoading models...")
    vqvae = VQVAE_V2().to(device)
    checkpoint = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)
    vqvae.load_state_dict(checkpoint['model_state_dict'])

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

    checkpoint = torch.load(args.world_model_checkpoint, map_location=device, weights_only=False)
    world_model.load_state_dict(checkpoint['model_state_dict'])

    # 加载数据
    print("\nLoading data...")
    data = np.load(args.token_file)
    tokens = data['tokens']
    actions = data['actions']

    print(f"Tokens shape: {tokens.shape}")
    print(f"Actions shape: {actions.shape}")

    # 生成视频
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_comparison_video(
        vqvae, world_model, tokens, actions, device, output_path, args.num_frames
    )


if __name__ == '__main__':
    main()
