"""
数据集类
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
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
    def __init__(self, token_file, context_frames=4, rollout_steps=0):
        """
        Args:
            token_file: .npz文件路径（包含tokens和actions）
            context_frames: 上下文帧数
            rollout_steps: 额外返回的未来步数（用于短rollout训练）
        """
        self.context_frames = context_frames
        self.rollout_steps = int(max(0, rollout_steps))

        # 加载数据
        data = np.load(token_file)
        self.tokens = data['tokens']  # (N, H, W)
        self.actions = data['actions']  # (N, action_dim)
        self.episode_ids = data['episode_ids'] if 'episode_ids' in data.files else None

        print(f"Loaded {len(self.tokens)} frames")
        print(f"Token shape: {self.tokens.shape}")
        print(f"Action shape: {self.actions.shape}")

        # 计算有效样本索引（避免跨episode）
        max_idx = len(self.tokens) - context_frames - self.rollout_steps
        if self.episode_ids is not None:
            self.valid_indices = [
                i for i in range(max_idx)
                if self.episode_ids[i] == self.episode_ids[i + context_frames + self.rollout_steps]
            ]
        else:
            self.valid_indices = list(range(max_idx))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        返回:
            context_tokens: (context_frames, H, W)
            context_actions: (context_frames, action_dim)
            target_token: (H, W)
        """
        idx = self.valid_indices[idx]

        # 上下文帧
        context_tokens = self.tokens[idx:idx + self.context_frames]
        context_actions = self.actions[idx:idx + self.context_frames]

        # 目标帧
        target_token = self.tokens[idx + self.context_frames]
        target_action = self.actions[idx + self.context_frames]

        sample = {
            'context_tokens': torch.from_numpy(context_tokens).long(),
            'context_actions': torch.from_numpy(context_actions).float(),
            'target_token': torch.from_numpy(target_token).long(),
            'target_action': torch.from_numpy(target_action).float(),
        }

        if self.rollout_steps > 0:
            # 未来rollout监督目标（从t+2开始，因为t+1是target_token）
            start_token = idx + self.context_frames + 1
            end_token = start_token + self.rollout_steps
            future_tokens = self.tokens[start_token:end_token]

            # 对应每个rollout step追加到动作窗口的动作（从t+1开始）
            start_action = idx + self.context_frames
            end_action = start_action + self.rollout_steps
            future_actions = self.actions[start_action:end_action]

            sample['future_tokens'] = torch.from_numpy(future_tokens).long()
            sample['future_actions'] = torch.from_numpy(future_actions).float()

        return sample


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


def get_world_model_dataloader(
    token_file,
    batch_size,
    context_frames=4,
    rollout_steps=0,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
    stratified_ab=False,
    ab_split=750,
    distributed=False,
    rank=0,
    world_size=1,
    return_sampler=False,
):
    """获取World Model数据加载器"""
    dataset = CARLASequenceDataset(token_file, context_frames, rollout_steps=rollout_steps)
    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        shuffle = False

    if stratified_ab and not distributed and dataset.episode_ids is not None:
        if batch_size % 2 != 0:
            raise ValueError("stratified_ab requires an even batch_size")

        # Build A/B index lists based on episode_id threshold
        a_indices = []
        b_indices = []
        for pos, idx in enumerate(dataset.valid_indices):
            ep = int(dataset.episode_ids[idx])
            if ep <= ab_split:
                a_indices.append(pos)
            else:
                b_indices.append(pos)

        if not a_indices or not b_indices:
            raise ValueError(
                f"stratified_ab requires non-empty A/B splits (A={len(a_indices)}, B={len(b_indices)})."
            )

        class BalancedABBatchSampler(torch.utils.data.Sampler):
            def __init__(self, a_idx, b_idx, batch_size):
                self.a_idx = a_idx
                self.b_idx = b_idx
                self.batch_size = batch_size
                self.half = batch_size // 2

            def __iter__(self):
                a = np.random.permutation(self.a_idx)
                b = np.random.permutation(self.b_idx)
                min_len = min(len(a), len(b))
                num_batches = (min_len * 2) // self.batch_size
                for i in range(num_batches):
                    a_start = i * self.half
                    b_start = i * self.half
                    batch = list(a[a_start:a_start + self.half]) + list(b[b_start:b_start + self.half])
                    np.random.shuffle(batch)
                    yield batch

            def __len__(self):
                min_len = min(len(self.a_idx), len(self.b_idx))
                return (min_len * 2) // self.batch_size

        sampler = BalancedABBatchSampler(a_indices, b_indices, batch_size)
        shuffle = False

    if sampler is not None and isinstance(sampler, torch.utils.data.Sampler) and not shuffle and not distributed and stratified_ab:
        dataloader_kwargs = dict(
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        dataloader_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
        dataloader_kwargs['persistent_workers'] = persistent_workers
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    if return_sampler:
        return dataloader, sampler
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
        self.episode_ids = data['episode_ids'] if 'episode_ids' in data.files else None

        print(f"Loaded {len(self.tokens)} frames for sequence training")
        print(f"Token shape: {self.tokens.shape}")
        print(f"Action shape: {self.actions.shape}")

        # 计算有效样本索引（确保能取到完整序列，不跨episode）
        max_idx = len(self.tokens) - seq_len
        if self.episode_ids is not None:
            self.valid_indices = [
                i for i in range(max_idx)
                if self.episode_ids[i] == self.episode_ids[i + seq_len - 1]
            ]
        else:
            self.valid_indices = list(range(max_idx))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        返回:
            tokens: (seq_len, H, W) - 连续的token序列
            actions: (seq_len, action_dim) - 对应的动作序列
        """
        idx = self.valid_indices[idx]
        tokens = self.tokens[idx:idx + self.seq_len]
        actions = self.actions[idx:idx + self.seq_len]

        return {
            'tokens': torch.from_numpy(tokens).long(),
            'actions': torch.from_numpy(actions).float(),
        }


def get_world_model_sequence_dataloader(
    token_file,
    batch_size,
    seq_len=16,
    num_workers=8,
    distributed=False,
    rank=0,
    world_size=1,
    return_sampler=False,
):
    """获取World Model序列数据加载器（用于Scheduled Sampling）"""
    dataset = CARLALongSequenceDataset(token_file, seq_len)
    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        shuffle = False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if return_sampler:
        return dataloader, sampler
    return dataloader
