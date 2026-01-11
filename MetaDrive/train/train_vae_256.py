import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
import glob
from torchvision.utils import save_image

# ================= ⚡️ 256px 高清配置 ⚡️ =================
DATASET_PATH = "dataset_v2_complex/images/*.png"  # 你的新数据集路径
SAVE_DIR = "checkpoints_vae_256"                  # 新的保存路径
BATCH_SIZE = 64                                   # A800 显存大，如果还嫌慢可以改成 128
LR = 1e-4
EPOCHS = 100                                      # 256 分辨率稍微多练一会儿
LATENT_DIM = 256                                  # 图像大了，潜在向量也给大一点
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256                                  # 目标分辨率
# =======================================================

class DriveDataset(Dataset):
    def __init__(self, glob_pattern):
        self.files = sorted(glob.glob(glob_pattern))
        print(f"📊 找到 {len(self.files)} 张图片 | 目标分辨率: {IMAGE_SIZE}x{IMAGE_SIZE}")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1) 
        return img_tensor

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()
        
        # Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # -> 32 x 128 x 128
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # -> 64 x 64 x 64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # -> 128 x 32 x 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# -> 256 x 16 x 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),# -> 512 x 8 x 8
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(512*8*8, latent_dim)
        self.fc_logvar = nn.Linear(512*8*8, latent_dim)
        
        # ✅ 这个层负责把 z (256) 变回 feature map (32768)
        self.decoder_input = nn.Linear(latent_dim, 512*8*8)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # -> 256 x 16 x 16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # -> 128 x 32 x 32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> 64 x 64 x 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # -> 32 x 128 x 128
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # -> 3 x 256 x 256
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # ✅ [修复的核心位置] 先通过 Linear 层放大，再进 Decoder
        z_projected = self.decoder_input(z)
        recon_x = self.decoder(z_projected)
        
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + 0.0001 * KLD

def train():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    dataset = DriveDataset(DATASET_PATH)
    if len(dataset) == 0:
        print("❌ 错误: 没找到图片，请检查路径！")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    model = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"🚀 开始训练 HD-VAE (256x256)... 设备: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item()/len(data):.1f}")

        # 保存对比图
        with torch.no_grad():
            test_data = next(iter(dataloader)).to(DEVICE)[:8]
            recon, _, _ = model(test_data)
            comparison = torch.cat([test_data, recon])
            save_image(comparison.cpu(), f"{SAVE_DIR}/recon_ep{epoch}.png", nrow=8)
        
        print(f"====> Epoch {epoch} Avg Loss: {train_loss / len(dataloader.dataset):.1f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/vae_256_ep{epoch}.pth")
            print(f"💾 模型已保存")

if __name__ == "__main__":
    train()