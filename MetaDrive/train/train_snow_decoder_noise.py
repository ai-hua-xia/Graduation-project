import argparse
import glob
import os
import sys
import time
from typing import Tuple

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
    parser = argparse.ArgumentParser(description="Train snow-conditioned decoder with noise map.")
    parser.add_argument("--base-glob", default="dataset_v2_complex/images/*.png")
    parser.add_argument("--vqvae-ckpt", default="checkpoints_vqvae_256/vqvae_256_ep99.pth")
    parser.add_argument("--out-dir", default="checkpoints_adapter")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--cond-hidden", type=int, default=64)
    parser.add_argument("--cond-scale", type=float, default=100.0)
    parser.add_argument("--snow-weight", type=float, default=100.0)
    parser.add_argument("--laplacian-weight", type=float, default=50)
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    return parser.parse_args()


def apply_snow_with_mask(img: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    img_f = img.astype(np.float32) / 255.0
    h, w = img_f.shape[:2]
    snow_mask = np.zeros((h, w), dtype=np.float32)
    num_flakes = int(h * w * rng.uniform(0.0006, 0.0012))
    for _ in range(num_flakes):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(1, 3))
        cv2.circle(snow_mask, (x, y), r, 1.0, -1)
    if rng.random() < 0.6:
        for _ in range(num_flakes // 4):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            length = int(rng.integers(5, 15))
            cv2.line(snow_mask, (x, y), (x + length, y + length), 1.0, 1)
    snow_mask = np.clip(snow_mask, 0, 1)
    snow_rgb = np.repeat(snow_mask[:, :, None], 3, axis=2)
    alpha = rng.uniform(0.2, 0.35)
    out = img_f * (1 - alpha) + snow_rgb * alpha
    out[..., 2] = np.clip(out[..., 2] * rng.uniform(1.02, 1.08), 0, 1)
    out[..., 0] = np.clip(out[..., 0] * rng.uniform(0.95, 1.02), 0, 1)
    mask = (snow_mask * alpha)[:, :, None]
    return out.astype(np.float32), mask.astype(np.float32)


class SnowNoiseDataset(Dataset):
    def __init__(self, base_glob: str, seed: int, max_images: int):
        self.files = sorted(glob.glob(base_glob))
        if max_images > 0:
            self.files = self.files[:max_images]
        self.seed = seed
        print(f"Found {len(self.files)} images | target: {IMAGE_SIZE}x{IMAGE_SIZE}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        img = cv2.imread(path)
        if img is None:
            base = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
            target = base.copy()
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            rng = np.random.default_rng(self.seed + idx)
            target, mask = apply_snow_with_mask(img, rng)
            base = img.astype(np.float32) / 255.0
        base_t = torch.from_numpy(base).permute(2, 0, 1)
        target_t = torch.from_numpy(target).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).permute(2, 0, 1)
        return {"base": base_t, "target": target_t, "mask": mask_t}


class SnowConditioner(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, embed_dim, 3, padding=1),
        )

    def forward(self, snow_map: torch.Tensor) -> torch.Tensor:
        return self.net(snow_map)


def laplacian_filter(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    )
    kernel = kernel.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))


def load_vqvae(ckpt_path: str, device: str) -> VQVAE:
    model = VQVAE().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.quantizer.parameters():
        p.requires_grad = False
    return model


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    dataset = SnowNoiseDataset(args.base_glob, args.seed, args.max_images)
    if len(dataset) == 0:
        print("No images found. Check --base-glob.")
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
    conditioner = SnowConditioner(EMBED_DIM, hidden=args.cond_hidden).to(args.device)

    if args.freeze_decoder:
        for p in vqvae.decoder.parameters():
            p.requires_grad = False
        vqvae.decoder.eval()
        train_params = list(conditioner.parameters())
    else:
        for p in vqvae.decoder.parameters():
            p.requires_grad = True
        vqvae.decoder.train()
        train_params = list(conditioner.parameters()) + list(vqvae.decoder.parameters())

    optimizer = torch.optim.Adam(train_params, lr=args.lr)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = args.use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    print(f"Start snow decoder training | device={args.device} | freeze_decoder={args.freeze_decoder}")
    for epoch in range(args.epochs):
        conditioner.train()
        if not args.freeze_decoder:
            vqvae.decoder.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            base = batch["base"].to(args.device, non_blocking=True)
            target = batch["target"].to(args.device, non_blocking=True)
            snow_mask = batch["mask"].to(args.device, non_blocking=True)

            with torch.no_grad():
                z = vqvae.encoder(base)
                quantized, _, _, _ = vqvae.quantizer(z)

            snow_latent = F.interpolate(
                snow_mask, size=quantized.shape[-2:], mode="area"
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.use_amp, dtype=amp_dtype):
                cond = conditioner(snow_latent)
                z_cond = quantized + args.cond_scale * cond
                recon = vqvae.decoder(z_cond)

                diff = torch.abs(recon - target)
                if args.snow_weight > 0:
                    weight = 1.0 + args.snow_weight * snow_mask
                    l1_loss = (diff * weight).mean()
                else:
                    l1_loss = diff.mean()
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
        print(
            f"Epoch {epoch} Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s"
        )

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"decoder_snow_noise_ep{epoch}.pth")
            torch.save(
                {
                    "decoder": vqvae.decoder.state_dict(),
                    "conditioner": conditioner.state_dict(),
                    "vqvae_ckpt": args.vqvae_ckpt,
                    "epoch": epoch,
                    "mode": "snow_noise",
                    "cond_hidden": args.cond_hidden,
                    "cond_scale": args.cond_scale,
                },
                ckpt_path,
            )
            print(f"Saved: {ckpt_path}")

        with torch.no_grad():
            base = batch["base"][:8].to(args.device)
            target = batch["target"][:8].to(args.device)
            snow_mask = batch["mask"][:8].to(args.device)
            z = vqvae.encoder(base)
            quantized, _, _, _ = vqvae.quantizer(z)
            snow_latent = F.interpolate(
                snow_mask, size=quantized.shape[-2:], mode="area"
            )
            recon = vqvae.decoder(quantized + args.cond_scale * conditioner(snow_latent))
            snow_vis = snow_mask.repeat(1, 3, 1, 1)
            preview = torch.cat([base, target, recon, snow_vis])
            save_path = os.path.join(args.out_dir, f"preview_snow_noise_ep{epoch}.png")
            save_image(preview.cpu(), save_path, nrow=8)


if __name__ == "__main__":
    main()
