"""
World Model - Transformer with FiLM
基于历史帧token和动作预测未来帧
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .film import FiLMedTransformerLayer


class WorldModel(nn.Module):
    """
    世界模型

    输入：
        - token序列：4帧历史 × 256个token = 1024个token
        - 动作序列：4个动作向量 (每个2维)

    输出：
        - 下一帧的token分布：256个位置，每个位置1024个类别
    """
    def __init__(
        self,
        num_embeddings=1024,  # VQ-VAE词表大小
        embed_dim=256,  # Token embedding维度
        hidden_dim=512,  # Transformer隐藏层维度
        num_heads=8,
        num_layers=8,
        context_frames=4,  # 上下文帧数
        action_dim=2,  # 动作维度
        tokens_per_frame=256,  # 每帧token数 (16×16)
        use_memory=False,
        memory_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.context_frames = context_frames
        self.tokens_per_frame = tokens_per_frame
        self.hidden_dim = hidden_dim
        self.use_memory = use_memory
        self.memory_dim = memory_dim

        # Token embedding
        self.token_embedding = nn.Embedding(num_embeddings, embed_dim)

        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, context_frames * tokens_per_frame, embed_dim) * 0.02
        )

        # Token投影到hidden_dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # 动作编码
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * context_frames, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
        )

        # 记忆模块：汇总历史上下文 + 动作
        if self.use_memory:
            self.memory_input_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, memory_dim),
                nn.Tanh(),
            )
            self.memory_gru = nn.GRUCell(memory_dim, memory_dim)
            self.memory_to_hidden = nn.Linear(memory_dim, hidden_dim)
            self.memory_pos_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # FiLMed Transformer层
        self.transformer_layers = nn.ModuleList([
            FiLMedTransformerLayer(hidden_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_embeddings),
        )

    def forward(self, token_seq, action_seq, memory=None, return_features=False, return_memory=False):
        """
        Args:
            token_seq: (B, context_frames, H, W) - token索引
            action_seq: (B, context_frames, action_dim) - 动作序列
            memory: (B, memory_dim) - 记忆向量

        Returns:
            logits: (B, tokens_per_frame, num_embeddings) - 下一帧预测
        """
        B = token_seq.shape[0]

        # 展平token: (B, context_frames, H, W) -> (B, context_frames * H * W)
        token_seq = token_seq.view(B, -1)  # (B, context_frames * tokens_per_frame)

        # Token embedding
        x = self.token_embedding(token_seq)  # (B, T, embed_dim)

        # 添加位置编码
        x = x + self.pos_embedding

        # 投影到hidden_dim
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # 编码动作
        action_seq_flat = action_seq.view(B, -1)  # (B, context_frames * action_dim)
        action_embedding = self.action_encoder(action_seq_flat)  # (B, hidden_dim)

        memory_next = None
        if self.use_memory:
            if memory is None:
                memory = torch.zeros(B, self.memory_dim, device=x.device, dtype=x.dtype)
            token_summary = x.mean(dim=1)
            memory_input = torch.cat([token_summary, action_embedding], dim=-1)
            memory_input = self.memory_input_proj(memory_input)
            memory_next = self.memory_gru(memory_input, memory)
            memory_token = self.memory_to_hidden(memory_next).unsqueeze(1)
            memory_token = memory_token + self.memory_pos_embedding
            x = torch.cat([memory_token, x], dim=1)

        # Transformer with FiLM
        for layer in self.transformer_layers:
            x = layer(x, action_embedding)

        # 只取最后一帧对应的token（预测下一帧）
        # 实际上我们预测整个序列的下一个token，但这里简化为预测下一帧的所有token
        # 更精确的做法是预测最后256个位置
        x_next = x[:, -self.tokens_per_frame:, :]  # (B, tokens_per_frame, hidden_dim)

        # 输出logits
        logits = self.output_proj(x_next)  # (B, tokens_per_frame, num_embeddings)

        if return_features and return_memory:
            return logits, x_next, memory_next
        if return_features:
            return logits, x_next
        if return_memory:
            return logits, memory_next
        return logits

    def predict_next_frame(self, token_seq, action_seq, memory=None, temperature=1.0, top_k=None, return_memory=False):
        """
        预测下一帧token

        Args:
            token_seq: (B, context_frames, H, W)
            action_seq: (B, context_frames, action_dim)
            memory: (B, memory_dim) - 记忆向量
            temperature: 采样温度
            top_k: top-k采样

        Returns:
            next_tokens: (B, H, W) - 下一帧token索引
        """
        self.eval()
        with torch.no_grad():
            if return_memory:
                logits, memory_next = self.forward(
                    token_seq, action_seq, memory=memory, return_memory=True
                )
            else:
                logits = self.forward(token_seq, action_seq, memory=memory)  # (B, tokens_per_frame, num_embeddings)

            # 温度采样
            logits = logits / temperature

            # Top-k过滤
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, :, [-1]]] = -float('Inf')

            # 采样
            probs = F.softmax(logits, dim=-1)  # (B, tokens_per_frame, num_embeddings)
            next_tokens = torch.multinomial(probs.view(-1, self.num_embeddings), 1)
            next_tokens = next_tokens.view(logits.shape[0], self.tokens_per_frame)

            # Reshape到2D
            H = W = int(self.tokens_per_frame ** 0.5)
            next_tokens = next_tokens.view(-1, H, W)

        if return_memory:
            return next_tokens, memory_next
        return next_tokens

    def dream(self, initial_tokens, actions, vqvae, device='cuda'):
        """
        生成视频序列

        Args:
            initial_tokens: (B, context_frames, H, W) - 初始帧
            actions: (B, num_steps, action_dim) - 动作序列
            vqvae: VQ-VAE模型，用于解码
            device: 设备

        Returns:
            frames: list of (B, 3, 256, 256) - 生成的RGB帧
        """
        self.eval()
        vqvae.eval()

        B, num_steps, _ = actions.shape
        _, _, H, W = initial_tokens.shape

        # 初始化token buffer
        token_buffer = initial_tokens.clone().to(device)  # (B, context_frames, H, W)
        memory = None

        frames = []

        with torch.no_grad():
            for t in range(num_steps):
                # 当前动作窗口
                if t < self.context_frames:
                    # 前几步，用零填充
                    action_window = torch.zeros(B, self.context_frames, actions.shape[-1], device=device)
                    action_window[:, -t-1:] = actions[:, :t+1]
                else:
                    action_window = actions[:, t-self.context_frames+1:t+1]

                # 预测下一帧
                next_tokens, memory = self.predict_next_frame(
                    token_buffer, action_window, memory=memory, return_memory=True
                )  # (B, H, W)

                # 解码为图像
                frame = vqvae.decode_tokens(next_tokens)  # (B, 3, 256, 256)
                frames.append(frame)

                # 更新buffer: 移除最旧的帧，添加新帧
                token_buffer = torch.cat([
                    token_buffer[:, 1:],
                    next_tokens.unsqueeze(1)
                ], dim=1)

        return frames


def compute_temporal_smoothness_loss(logits_seq, action_magnitudes, beta=2.0):
    """
    计算时间平滑损失（动作自适应）

    Args:
        logits_seq: (B, T, tokens_per_frame, num_embeddings) - 连续帧的logits
        action_magnitudes: (B, T-1) - 动作幅度
        beta: 自适应系数

    Returns:
        smooth_loss: 标量
    """
    if logits_seq.shape[1] < 2:
        return torch.tensor(0.0, device=logits_seq.device)

    # 计算相邻帧logits的KL散度
    kl_losses = []
    for t in range(logits_seq.shape[1] - 1):
        p = F.softmax(logits_seq[:, t], dim=-1)  # (B, tokens, vocab)
        q = F.softmax(logits_seq[:, t + 1], dim=-1)

        kl = F.kl_div(q.log(), p, reduction='none').sum(dim=-1).mean(dim=-1)  # (B,)

        # 动作自适应权重
        action_mag = action_magnitudes[:, t]  # (B,)
        weight = torch.exp(-beta * action_mag)  # 动作越大，权重越小

        kl_losses.append((kl * weight).mean())

    return torch.stack(kl_losses).mean()
