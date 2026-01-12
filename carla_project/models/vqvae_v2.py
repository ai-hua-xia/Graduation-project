"""
VQ-VAE V2 - 改进版本，解决codebook collapse问题

改进点:
1. EMA (Exponential Moving Average) 更新codebook - 更稳定
2. 死码重置机制 - 防止codebook collapse
3. 可选的感知损失 - 提升视觉质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """
    使用EMA更新的向量量化层
    参考: https://arxiv.org/abs/1711.00937
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        reset_threshold: int = 1,  # 使用次数低于此值的码将被重置
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.reset_threshold = reset_threshold

        # 嵌入码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()

        # EMA统计量 (不参与梯度计算)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

        # 追踪每个码的使用次数
        self.register_buffer('usage_count', torch.zeros(num_embeddings))

    def forward(self, inputs):
        """
        Args:
            inputs: (B, C, H, W)

        Returns:
            quantized: 量化后的向量
            loss: VQ损失 (commitment loss)
            indices: 索引 (B, H, W)
            perplexity: 困惑度 (用于监控codebook使用情况)
        """
        # 转换shape: (B, C, H, W) -> (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # 展平: (B, H, W, C) -> (BHW, C)
        flat_input = inputs.view(-1, self.embedding_dim)

        # 计算距离 (L2)
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # 找到最近的embedding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight)

        # EMA更新 (仅在训练时)
        if self.training:
            # 更新使用次数统计
            self.usage_count.add_(encodings.sum(0))

            # 更新EMA cluster size
            self.ema_cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            # 更新EMA embedding
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # 归一化得到新的embedding
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        # Commitment loss (只有这个需要梯度)
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        loss = self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = flat_input + (quantized - flat_input).detach()

        # 计算perplexity (用于监控)
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # 转换回原始shape
        quantized = quantized.view(input_shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        indices = encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])

        return quantized, loss, indices, perplexity

    def reset_dead_codes(self, inputs):
        """
        重置使用次数很低的码 (死码)
        用随机的encoder输出来初始化
        """
        if not self.training:
            return 0

        # 找到死码
        dead_codes = self.usage_count < self.reset_threshold
        num_dead = dead_codes.sum().item()

        if num_dead == 0:
            return 0

        # 用随机的输入来重置死码
        flat_input = inputs.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)

        # 随机选择一些输入
        if flat_input.shape[0] >= num_dead:
            indices = torch.randperm(flat_input.shape[0])[:num_dead]
            new_codes = flat_input[indices]
        else:
            # 如果输入不够，就重复使用
            repeat_times = (num_dead // flat_input.shape[0]) + 1
            expanded = flat_input.repeat(repeat_times, 1)
            indices = torch.randperm(expanded.shape[0])[:num_dead]
            new_codes = expanded[indices]

        # 添加一些噪声
        new_codes = new_codes + torch.randn_like(new_codes) * 0.01

        # 重置embedding
        self.embedding.weight.data[dead_codes] = new_codes
        self.ema_w.data[dead_codes] = new_codes
        self.ema_cluster_size.data[dead_codes] = 1.0
        self.usage_count.data[dead_codes] = 0

        return num_dead

    def get_codebook_usage(self):
        """返回codebook使用统计"""
        used = (self.usage_count > 0).sum().item()
        return {
            'used_codes': used,
            'total_codes': self.num_embeddings,
            'usage_ratio': used / self.num_embeddings,
        }


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
        x = F.silu(x)  # 使用SiLU代替ReLU

        x = self.conv2(x)
        x = self.norm2(x)

        return F.silu(x + residual)


class Encoder(nn.Module):
    """编码器: 256x256 -> 16x16"""
    def __init__(self, in_channels=3, base_channels=128, embed_dim=256):
        super().__init__()

        self.layers = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),

            # 128 -> 64
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),

            # 64 -> 32
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),

            # 32 -> 16
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),

            # 投影到embedding维度
            nn.Conv2d(base_channels * 2, embed_dim, 1),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    """解码器: 16x16 -> 256x256"""
    def __init__(self, embed_dim=256, base_channels=128, out_channels=3):
        super().__init__()

        self.layers = nn.Sequential(
            # 投影
            nn.Conv2d(embed_dim, base_channels * 2, 1),

            # 16 -> 32
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),

            # 32 -> 64
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),

            # 64 -> 128
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),

            # 128 -> 256
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
            nn.ConvTranspose2d(base_channels, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),  # 输出范围[-1, 1]
        )

    def forward(self, x):
        return self.layers(x)


class VQVAE_V2(nn.Module):
    """
    改进版VQ-VAE模型

    改进点:
    1. 更大的encoder/decoder (base_channels: 64->128)
    2. 更多的残差块
    3. 使用EMA更新的VectorQuantizer
    4. SiLU激活函数
    """
    def __init__(
        self,
        in_channels=3,
        base_channels=128,  # 增加到128
        embed_dim=256,
        num_embeddings=1024,
        commitment_cost=0.25,
        ema_decay=0.99,
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, base_channels, embed_dim)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings, embed_dim, commitment_cost, decay=ema_decay
        )
        self.decoder = Decoder(embed_dim, base_channels, in_channels)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 256, 256)

        Returns:
            recon: 重建图像
            vq_loss: VQ损失
            indices: token索引 (B, 16, 16)
            perplexity: 困惑度
        """
        z = self.encoder(x)
        quantized, vq_loss, indices, perplexity = self.quantizer(z)
        recon = self.decoder(quantized)

        return recon, vq_loss, indices, perplexity

    def encode(self, x):
        """仅编码，返回token索引"""
        with torch.no_grad():
            z = self.encoder(x)
            _, _, indices, _ = self.quantizer(z)
        return indices

    def decode_tokens(self, indices):
        """从token索引解码图像"""
        with torch.no_grad():
            quantized = self.quantizer.embedding(indices)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            recon = self.decoder(quantized)
        return recon

    def reset_dead_codes(self, x):
        """重置死码"""
        z = self.encoder(x)
        return self.quantizer.reset_dead_codes(z)

    def get_codebook_usage(self):
        """获取codebook使用统计"""
        return self.quantizer.get_codebook_usage()
