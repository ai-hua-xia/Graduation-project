"""
导出VQ-VAE tokens
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.vqvae import VQVAE
from collect.utils import load_actions
import cv2


def load_episode_data(episode_dir):
    """加载一个episode的数据"""
    episode_dir = Path(episode_dir)

    # 加载图像
    images_dir = episode_dir / "images"
    image_paths = sorted(images_dir.glob("*.png"))

    images = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        images.append(img)

    # 加载动作
    actions = load_actions(episode_dir)

    return np.array(images), actions


def main():
    parser = argparse.ArgumentParser(description='Export VQ-VAE tokens')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to CARLA raw data')
    parser.add_argument('--vqvae-checkpoint', type=str, required=True,
                        help='Path to VQ-VAE checkpoint')
    parser.add_argument('--output', type=str, default='../data/tokens/tokens_actions.npz',
                        help='Output file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载VQ-VAE
    print("\nLoading VQ-VAE...")
    model = VQVAE().to(device)
    checkpoint = torch.load(args.vqvae_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 收集所有episode
    data_path = Path(args.data_path)
    episode_dirs = sorted(data_path.glob("episode_*"))
    print(f"Found {len(episode_dirs)} episodes")

    # 编码所有数据
    all_tokens = []
    all_actions = []

    for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
        try:
            images, actions = load_episode_data(episode_dir)

            # 确保长度一致
            min_len = min(len(images), len(actions))
            images = images[:min_len]
            actions = actions[:min_len]

            # 编码图像
            episode_tokens = []
            with torch.no_grad():
                for i in range(0, len(images), 32):  # 批处理
                    batch = images[i:i+32]
                    batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
                    batch = batch * 2.0 - 1.0  # 归一化到[-1, 1]

                    tokens = model.encode(batch)  # (B, H, W)
                    episode_tokens.append(tokens.cpu().numpy())

            episode_tokens = np.concatenate(episode_tokens, axis=0)

            all_tokens.append(episode_tokens)
            all_actions.append(actions)

        except Exception as e:
            print(f"Error processing {episode_dir}: {e}")
            continue

    # 合并所有episode
    all_tokens = np.concatenate(all_tokens, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    print(f"\nTotal frames: {len(all_tokens)}")
    print(f"Token shape: {all_tokens.shape}")
    print(f"Action shape: {all_actions.shape}")

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        tokens=all_tokens,
        actions=all_actions,
    )

    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
