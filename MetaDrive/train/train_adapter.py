import argparse
import glob
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train.train_vqvae_256 import VQVAE, IMAGE_SIZE, EMBED_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train decoder adapter for style transfer")
    parser.add_argument("--base-glob", default="dataset_v2_complex/images/*.png")
    parser.add_argument("--style-dir", default="dataset_style/night/images")
    parser.add_argument("--vqvae-ckpt", default="checkpoints_vqvae_256/vqvae_256_ep99.pth")
    parser.add_argument("--out-dir", default="checkpoints_adapter")
    parser.add_argument("--style-name", default="")
    parser.add_argument("--mode", choices=["decoder", "adapter"], default="decoder")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter-bottleneck", type=int, default=64)
    parser.add_argument("--laplacian-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    return parser.parse_args()


class PairDataset(Dataset):
    def __init__(self, base_glob: str, style_dir: str, image_size: int):
        self.image_size = image_size
        base_files = sorted(glob.glob(base_glob))
        pairs: List[Tuple[str, str]] = []
        for base_path in base_files:
            name = os.path.basename(base_path)
            style_path = os.path.join(style_dir, name)
            if os.path.exists(style_path):
                pairs.append((base_path, style_path))
        self.pairs = pairs
        print(f"Found {len(self.pairs)} paired images")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_img(self, path: str) -> torch.Tensor:
        img = cv2.imread(path)
        if img is None:
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        img_tensor = torch.from_numpy(img).float() / 255.0
        return img_tensor.permute(2, 0, 1)

    def __getitem__(self, idx: int) -> dict:
        base_path, style_path = self.pairs[idx]
        base_img = self._load_img(base_path)
        style_img = self._load_img(style_path)
        return {"base": base_img, "style": style_img}


class LatentAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1),
            nn.SiLU(),
            nn.Conv2d(bottleneck, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def load_vqvae(ckpt_path: str, device: str) -> VQVAE:
    model = VQVAE().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def laplacian_filter(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    )
    kernel = kernel.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))


def main() -> None:
    args = parse_args()
    def resolve_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(PROJECT_ROOT, path)

    args.base_glob = resolve_path(args.base_glob)
    args.style_dir = resolve_path(args.style_dir)
    args.vqvae_ckpt = resolve_path(args.vqvae_ckpt)
    args.out_dir = resolve_path(args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = PairDataset(args.base_glob, args.style_dir, IMAGE_SIZE)
    if len(dataset) == 0:
        print("No paired images found. Check base/style paths.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    vqvae = load_vqvae(args.vqvae_ckpt, args.device)
    adapter = None
    train_params = []
    mode_tag = args.mode
    if args.mode == "adapter":
        adapter = LatentAdapter(EMBED_DIM, bottleneck=args.adapter_bottleneck).to(args.device)
        train_params = list(adapter.parameters())
    else:
        for p in vqvae.decoder.parameters():
            p.requires_grad = True
        train_params = list(vqvae.decoder.parameters())

    optimizer = torch.optim.Adam(train_params, lr=args.lr)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = args.use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    print(f"Start {mode_tag} training | style={args.style_name} | device={args.device}")
    for epoch in range(args.epochs):
        if adapter is not None:
            adapter.train()
        else:
            vqvae.decoder.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            base = batch["base"].to(args.device, non_blocking=True)
            target = batch["style"].to(args.device, non_blocking=True)

            with torch.no_grad():
                z = vqvae.encoder(base)
                quantized, _, _, _ = vqvae.quantizer(z)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=amp_dtype):
                z_adapt = adapter(quantized) if adapter is not None else quantized
                recon = vqvae.decoder(z_adapt)
                l1_loss = F.l1_loss(recon, target)
                if args.laplacian_weight > 0:
                    hf_loss = F.l1_loss(laplacian_filter(recon), laplacian_filter(target))
                    loss = l1_loss + args.laplacian_weight * hf_loss
                else:
                    hf_loss = torch.tensor(0.0, device=l1_loss.device)
                    loss = l1_loss

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            if step % 50 == 0:
                print(
                    f"Epoch {epoch} Step {step}/{len(dataloader)} | L1 {l1_loss.item():.4f} "
                    f"HF {hf_loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Avg L1: {avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_prefix = "adapter" if adapter is not None else "decoder"
            ckpt_path = os.path.join(args.out_dir, f"{ckpt_prefix}_{args.style_name}_ep{epoch}.pth")
            payload = {
                "style": args.style_name,
                "vqvae_ckpt": args.vqvae_ckpt,
                "epoch": epoch,
                "mode": args.mode,
            }
            if adapter is not None:
                payload["adapter"] = adapter.state_dict()
                payload["bottleneck"] = args.adapter_bottleneck
            else:
                payload["decoder"] = vqvae.decoder.state_dict()
            torch.save(payload, ckpt_path)
            print(f"Saved: {ckpt_path}")

        with torch.no_grad():
            base = batch["base"][:8].to(args.device)
            target = batch["style"][:8].to(args.device)
            z = vqvae.encoder(base)
            quantized, _, _, _ = vqvae.quantizer(z)
            z_adapt = adapter(quantized) if adapter is not None else quantized
            recon = vqvae.decoder(z_adapt)
            preview = torch.cat([base, target, recon])
            save_path = os.path.join(args.out_dir, f"preview_{args.style_name}_{mode_tag}_ep{epoch}.png")
            save_image(preview.cpu(), save_path, nrow=8)


if __name__ == "__main__":
    main()
