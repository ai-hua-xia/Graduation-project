"""
数据集类
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
import json


class CARLAImageDataset(Dataset):
    """CARLA图像数据集（用于VQ-VAE训练）"""
    def __init__(self, data_root, transform=None):
        """
        Args:
            data_root: 数据根目录（包含多个episode_XXXX文件夹）
            transform: 图像变换
        """
        self.data_root = Path(data_root)
        self.transform = transform

        # 收集所有图像路径
        self.image_paths = []
        for episode_dir in sorted(self.data_root.glob("episode_*")):
            images_dir = episode_dir / "images"
            if images_dir.exists():
                self.image_paths.extend(sorted(images_dir.glob("*.png")))

        print(f"Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 归一化到[0, 1]
        img = img.astype(np.float32) / 255.0

        # 转为tensor
        img = torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # 应用transform
        if self.transform:
            img = self.transform(img)

        return img


class CARLASequenceDataset(Dataset):
    """CARLA序列数据集（用于World Model训练）"""
    def __init__(self, token_file, context_frames=4):
        """
        Args:
            token_file: .npz文件路径（包含tokens和actions）
            context_frames: 上下文帧数
        """
        self.context_frames = context_frames

        # 加载数据
        data = np.load(token_file)
        self.tokens = data['tokens']  # (N, H, W)
        self.actions = data['actions']  # (N, action_dim)

        print(f"Loaded {len(self.tokens)} frames")
        print(f"Token shape: {self.tokens.shape}")
        print(f"Action shape: {self.actions.shape}")

        # 计算有效样本数
        self.valid_indices = len(self.tokens) - context_frames

    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        """
        返回:
            context_tokens: (context_frames, H, W)
            context_actions: (context_frames, action_dim)
            target_token: (H, W)
        """
        # 上下文帧
        context_tokens = self.tokens[idx:idx + self.context_frames]
        context_actions = self.actions[idx:idx + self.context_frames]

        # 目标帧
        target_token = self.tokens[idx + self.context_frames]

        return {
            'context_tokens': torch.from_numpy(context_tokens).long(),
            'context_actions': torch.from_numpy(context_actions).float(),
            'target_token': torch.from_numpy(target_token).long(),
        }


def get_vqvae_dataloader(data_root, batch_size, num_workers=8, transform=None):
    """获取VQ-VAE数据加载器"""
    dataset = CARLAImageDataset(data_root, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


def get_world_model_dataloader(token_file, batch_size, context_frames=4, num_workers=8):
    """获取World Model数据加载器"""
    dataset = CARLASequenceDataset(token_file, context_frames)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


class CARLALongSequenceDataset(Dataset):
    """CARLA长序列数据集（用于Scheduled Sampling训练）"""
    def __init__(self, token_file, seq_len=16):
        """
        Args:
            token_file: .npz文件路径（包含tokens和actions）
            seq_len: 序列长度
        """
        self.seq_len = seq_len

        # 加载数据
        data = np.load(token_file)
        self.tokens = data['tokens']  # (N, H, W)
        self.actions = data['actions']  # (N, action_dim)

        print(f"Loaded {len(self.tokens)} frames for sequence training")
        print(f"Token shape: {self.tokens.shape}")
        print(f"Action shape: {self.actions.shape}")

        # 计算有效样本数（确保能取到完整序列）
        self.valid_indices = len(self.tokens) - seq_len

    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        """
        返回:
            tokens: (seq_len, H, W) - 连续的token序列
            actions: (seq_len, action_dim) - 对应的动作序列
        """
        tokens = self.tokens[idx:idx + self.seq_len]
        actions = self.actions[idx:idx + self.seq_len]

        return {
            'tokens': torch.from_numpy(tokens).long(),
            'actions': torch.from_numpy(actions).float(),
        }


def get_world_model_sequence_dataloader(token_file, batch_size, seq_len=16, num_workers=8):
    """获取World Model序列数据加载器（用于Scheduled Sampling）"""
    dataset = CARLALongSequenceDataset(token_file, seq_len)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader
