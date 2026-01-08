import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

# ================= VQ-VAE 256 CONFIG =================
DATASET_PATH = "dataset_v2_complex/images/*.png"
SAVE_DIR = "checkpoints_vqvae_256"
IMAGE_SIZE = 256
EPOCHS = 100
BATCH_SIZE = 64
LR = 2e-4
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_CHANNELS = 128
EMBED_DIM = 256
NUM_EMBEDDINGS = 1024
COMMITMENT_COST = 0.25
EMA_DECAY = 0.99
USE_AMP = True
AMP_DTYPE = "bf16"  # "bf16" or "fp16"
SAVE_EVERY = 5
SEED = 42
# =====================================================


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class DriveDataset(Dataset):
    def __init__(self, glob_pattern: str):
        self.files = sorted(glob.glob(glob_pattern))
        print(f"Found {len(self.files)} images | target: {IMAGE_SIZE}x{IMAGE_SIZE}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.files[idx]
        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_tensor = torch.from_numpy(img).float() / 255.0
        return img_tensor.permute(2, 0, 1)


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 128,
                 ch_mults=(1, 1, 2, 2, 4), num_res_blocks: int = 2,
                 z_channels: int = 256):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.stage_convs = nn.ModuleList()
        self.stage_blocks = nn.ModuleList()
        self.stage_downs = nn.ModuleList()

        cur_ch = base_channels
        for i, mult in enumerate(ch_mults):
            out_ch = base_channels * mult
            self.stage_convs.append(
                nn.Conv2d(cur_ch, out_ch, 3, padding=1) if out_ch != cur_ch else nn.Identity()
            )
            self.stage_blocks.append(
                nn.ModuleList([ResBlock(out_ch) for _ in range(num_res_blocks)])
            )
            if i != len(ch_mults) - 1:
                self.stage_downs.append(Downsample(out_ch))
            cur_ch = out_ch

        self.norm_out = group_norm(cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, z_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for i in range(len(self.stage_blocks)):
            x = self.stage_convs[i](x)
            for block in self.stage_blocks[i]:
                x = block(x)
            if i < len(self.stage_downs):
                x = self.stage_downs[i](x)
        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, base_channels: int = 128,
                 ch_mults=(4, 2, 2, 1, 1), num_res_blocks: int = 2,
                 z_channels: int = 256):
        super().__init__()
        self.conv_in = nn.Conv2d(z_channels, base_channels * ch_mults[0], 3, padding=1)
        self.stage_convs = nn.ModuleList()
        self.stage_blocks = nn.ModuleList()
        self.stage_ups = nn.ModuleList()

        cur_ch = base_channels * ch_mults[0]
        for i, mult in enumerate(ch_mults):
            out_ch = base_channels * mult
            self.stage_convs.append(
                nn.Conv2d(cur_ch, out_ch, 3, padding=1) if out_ch != cur_ch else nn.Identity()
            )
            self.stage_blocks.append(
                nn.ModuleList([ResBlock(out_ch) for _ in range(num_res_blocks)])
            )
            if i != len(ch_mults) - 1:
                self.stage_ups.append(Upsample(out_ch))
            cur_ch = out_ch

        self.norm_out = group_norm(cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for i in range(len(self.stage_blocks)):
            x = self.stage_convs[i](x)
            for block in self.stage_blocks[i]:
                x = block(x)
            if i < len(self.stage_ups):
                x = self.stage_ups[i](x)
        x = self.norm_out(x)
        x = F.silu(x)
        return torch.sigmoid(self.conv_out(x))


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad = False

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)

        # Quantization math in fp32 for stability under AMP
        flat_input_fp32 = flat_input.float()
        embed_fp32 = self.embedding.weight.float()

        distances = (
            flat_input_fp32.pow(2).sum(1, keepdim=True)
            + embed_fp32.pow(2).sum(1)
            - 2 * torch.matmul(flat_input_fp32, embed_fp32.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input_fp32.dtype)

        quantized = torch.matmul(encodings, embed_fp32)
        quantized = quantized.view(inputs.shape).type_as(flat_input_fp32)

        if self.training:
            with torch.no_grad():
                self.ema_cluster_size.mul_(self.decay).add_(encodings.sum(0) * (1 - self.decay))
                dw = torch.matmul(encodings.t(), flat_input_fp32)
                self.ema_w.mul_(self.decay).add_(dw * (1 - self.decay))

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
                )
                cluster_size = torch.clamp(cluster_size, min=self.eps)
                self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), inputs.float())
        loss = self.commitment_cost * e_latent_loss

        quantized = inputs.float() + (quantized - inputs.float()).detach()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous().type_as(inputs)
        encoding_indices = encoding_indices.view(inputs.shape[0], inputs.shape[1], inputs.shape[2])
        return quantized, loss, perplexity, encoding_indices


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(base_channels=BASE_CHANNELS, z_channels=EMBED_DIM)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=NUM_EMBEDDINGS,
            embedding_dim=EMBED_DIM,
            commitment_cost=COMMITMENT_COST,
            decay=EMA_DECAY,
        )
        self.decoder = Decoder(base_channels=BASE_CHANNELS, z_channels=EMBED_DIM)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quantized, vq_loss, perplexity, indices = self.quantizer(z)
        recon = self.decoder(quantized)
        return recon, vq_loss, perplexity, indices

    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(x)
            _, _, _, indices = self.quantizer(z)
        return indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            quantized = self.quantizer.embedding(indices)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            return self.decoder(quantized)


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    set_seed(SEED)

    dataset = DriveDataset(DATASET_PATH)
    if len(dataset) == 0:
        print("No images found. Check DATASET_PATH.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    model = VQVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16
    use_scaler = USE_AMP and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    print(f"Start VQ-VAE training | device: {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0

        for batch_idx, data in enumerate(dataloader):
            data = data.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP, dtype=amp_dtype):
                recon, vq_loss, perplexity, _ = model(data)
                recon_loss = F.l1_loss(recon, data)
                loss = recon_loss + vq_loss

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"loss={loss.item():.4f} recon={recon_loss.item():.4f} "
                    f"vq={vq_loss.item():.4f} perplexity={perplexity.item():.2f}"
                )

        with torch.no_grad():
            test_data = next(iter(dataloader)).to(DEVICE)[:8]
            recon, _, _, _ = model(test_data)
            comparison = torch.cat([test_data, recon])
            save_image(comparison.cpu(), f"{SAVE_DIR}/recon_ep{epoch}.png", nrow=8)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} avg_loss={avg_loss:.4f}")

        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = os.path.join(SAVE_DIR, f"vqvae_256_ep{epoch}.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    train()
