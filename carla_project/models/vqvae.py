"""
VQ-VAE (Vector Quantized Variational AutoEncoder)
用于视觉压缩和离散表示学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """向量量化层"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 嵌入码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        """
        Args:
            inputs: (B, C, H, W)

        Returns:
            quantized: 量化后的向量
            loss: VQ损失
            indices: 索引
        """
        # 转换shape: (B, C, H, W) -> (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # 展平: (B, H, W, C) -> (BHW, C)
        flat_input = inputs.view(-1, self.embedding_dim)

        # 计算距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # 找到最近的embedding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # 损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # 转换回原始shape: (B, H, W, C) -> (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # 返回索引: (BHW,) -> (B, H, W)
        indices = encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])

        return quantized, loss, indices


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        return F.relu(x + residual)


class Encoder(nn.Module):
    """编码器: 256x256 -> 16x16"""
    def __init__(self, in_channels=3, base_channels=64, embed_dim=256):
        super().__init__()

        self.layers = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(base_channels, base_channels),

            # 128 -> 64
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(base_channels * 2, base_channels * 2),

            # 64 -> 32
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(base_channels * 4, base_channels * 4),

            # 32 -> 16
            nn.Conv2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(base_channels * 4, base_channels * 4),

            # 投影到embedding维度
            nn.Conv2d(base_channels * 4, embed_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    """解码器: 16x16 -> 256x256"""
    def __init__(self, embed_dim=256, base_channels=64, out_channels=3):
        super().__init__()

        self.layers = nn.Sequential(
            # 投影
            nn.Conv2d(embed_dim, base_channels * 4, 3, padding=1),

            # 16 -> 32
            ResidualBlock(base_channels * 4, base_channels * 4),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1),
            nn.ReLU(),

            # 32 -> 64
            ResidualBlock(base_channels * 4, base_channels * 4),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(),

            # 64 -> 128
            ResidualBlock(base_channels * 2, base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.ReLU(),

            # 128 -> 256
            ResidualBlock(base_channels, base_channels),
            nn.ConvTranspose2d(base_channels, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),  # 输出范围[-1, 1]
        )

    def forward(self, x):
        return self.layers(x)


class VQVAE(nn.Module):
    """完整的VQ-VAE模型"""
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        embed_dim=256,
        num_embeddings=1024,
        commitment_cost=0.25
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, base_channels, embed_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embed_dim, commitment_cost)
        self.decoder = Decoder(embed_dim, base_channels, in_channels)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 256, 256)

        Returns:
            recon: 重建图像
            vq_loss: VQ损失
            indices: token索引 (B, 16, 16)
        """
        z = self.encoder(x)  # (B, embed_dim, 16, 16)
        quantized, vq_loss, indices = self.quantizer(z)
        recon = self.decoder(quantized)

        return recon, vq_loss, indices

    def encode(self, x):
        """仅编码，返回token索引"""
        with torch.no_grad():
            z = self.encoder(x)
            _, _, indices = self.quantizer(z)
        return indices  # (B, 16, 16)

    def decode_tokens(self, indices):
        """从token索引解码图像"""
        with torch.no_grad():
            # 从索引获取embedding
            quantized = self.quantizer.embedding(indices)  # (B, H, W, C)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            recon = self.decoder(quantized)
        return recon
