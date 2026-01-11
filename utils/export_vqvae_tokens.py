import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 允许从子目录直接运行
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train.train_vqvae_256 import VQVAE
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VQ-VAE tokens for a dataset")
    parser.add_argument(
        "--dataset",
        default="dataset_v2_complex/images/*.png",
        help="Glob pattern for images",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints_vqvae_256/vqvae_256_ep99.pth",
        help="Path to VQ-VAE checkpoint",
    )
    parser.add_argument(
        "--output",
        default="dataset_v2_complex/tokens_actions_vqvae_16x16.npz",
        help="Output .npz path for tokens + actions",
    )
    parser.add_argument(
        "--meta",
        default="dataset_v2_complex/tokens_actions_vqvae_16x16.json",
        help="Output .json metadata path",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class DriveDataset(Dataset):
    def __init__(self, files: list[str], image_size: int):
        self.files = files
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.files[idx]
        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        img_tensor = torch.from_numpy(img).float() / 255.0
        return img_tensor.permute(2, 0, 1)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: str) -> None:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

def parse_image_index(path: str) -> int | None:
    name = os.path.basename(path)
    match = re.search(r"img_(\d+)\.png$", name)
    if not match:
        return None
    return int(match.group(1))


def load_actions_for_indices(dataset_glob: str, indices: list[int]) -> np.ndarray:
    images_dir = os.path.dirname(os.path.abspath(dataset_glob.rstrip("/*")))
    dataset_dir = os.path.dirname(images_dir)
    actions_path = os.path.join(dataset_dir, "actions.npy")
    if not os.path.exists(actions_path):
        raise FileNotFoundError(f"Actions file not found: {actions_path}")

    actions = np.load(actions_path)
    print(f"Actions shape: {actions.shape}")
    if actions.ndim != 2 or actions.shape[1] < 2:
        raise ValueError("Unexpected actions shape.")
    max_idx = max(indices)
    if actions.shape[0] <= max_idx:
        raise ValueError(
            f"Actions length {actions.shape[0]} <= max image index {max_idx}"
        )
    aligned = actions[np.array(indices, dtype=np.int64)]
    return aligned


def report_action_stats(actions: np.ndarray) -> None:
    mean_abs = np.mean(np.abs(actions), axis=0)
    std = np.std(actions, axis=0)
    near_zero = np.mean(np.all(np.abs(actions) < 1e-3, axis=1))
    print(f"Actions mean_abs: {mean_abs}, std: {std}, near_zero_ratio: {near_zero:.3f}")


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.dataset))
    if not files:
        print("No images found. Check --dataset.")
        return

    indexed = []
    for path in files:
        idx = parse_image_index(path)
        if idx is None:
            raise ValueError(f"Unrecognized filename: {path}")
        indexed.append((idx, path))
    indexed.sort(key=lambda x: x[0])
    image_indices = [idx for idx, _ in indexed]
    files = [path for _, path in indexed]
    num_images = len(files)
    if len(image_indices) != len(set(image_indices)):
        raise ValueError("Duplicate image indices found.")

    token_h = args.image_size // args.downsample
    token_w = args.image_size // args.downsample
    if args.image_size % args.downsample != 0:
        print("Error: image_size must be divisible by downsample.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.meta), exist_ok=True)

    dataset = DriveDataset(files, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = VQVAE().to(args.device)
    load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    tokens = np.zeros((num_images, token_h, token_w), dtype=np.uint16)
    offset = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.to(args.device, non_blocking=True)
            token_indices = model.encode_indices(data)
            batch_tokens = token_indices.cpu().numpy().astype(np.uint16)
            batch_size = batch_tokens.shape[0]
            tokens[offset:offset + batch_size] = batch_tokens
            offset += batch_size
            if batch_idx % 20 == 0:
                print(f"Encoded {offset}/{num_images}")

    actions = load_actions_for_indices(args.dataset, image_indices)
    np.savez_compressed(
        args.output,
        tokens=tokens,
        actions=actions.astype(np.float32),
        indices=np.array(image_indices, dtype=np.int64),
    )
    meta = {
        "dataset_glob": args.dataset,
        "num_images": num_images,
        "image_size": args.image_size,
        "downsample": args.downsample,
        "token_shape": [token_h, token_w],
        "token_dtype": str(tokens.dtype),
        "checkpoint": args.checkpoint,
        "output_format": "npz",
        "contains": ["tokens", "actions", "indices"],
        "min_index": int(min(image_indices)),
        "max_index": int(max(image_indices)),
        "missing_count": int((max(image_indices) + 1) - len(image_indices)),
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report_action_stats(actions)
    print(f"Saved tokens to {args.output}")
    print(f"Saved meta to {args.meta}")


if __name__ == "__main__":
    main()
