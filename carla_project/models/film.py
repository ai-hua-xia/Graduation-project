"""
FiLM (Feature-wise Linear Modulation) 层
用于动作条件调制
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    FiLM层：gamma * x + beta

    通过动作embedding生成gamma和beta，对特征进行仿射变换
    """
    def __init__(self, condition_dim, feature_dim):
        """
        Args:
            condition_dim: 条件向量维度（动作embedding维度）
            feature_dim: 特征维度（要调制的hidden state维度）
        """
        super().__init__()

        self.gamma_layer = nn.Linear(condition_dim, feature_dim)
        self.beta_layer = nn.Linear(condition_dim, feature_dim)

        # 初始化：gamma接近1，beta接近0（恒等变换起点）
        nn.init.ones_(self.gamma_layer.weight)
        nn.init.zeros_(self.gamma_layer.bias)
        nn.init.zeros_(self.beta_layer.weight)
        nn.init.zeros_(self.beta_layer.bias)

    def forward(self, x, condition):
        """
        Args:
            x: 特征 (B, ..., feature_dim)
            condition: 条件向量 (B, condition_dim)

        Returns:
            modulated: 调制后的特征
        """
        gamma = self.gamma_layer(condition)  # (B, feature_dim)
        beta = self.beta_layer(condition)    # (B, feature_dim)

        # 广播到x的shape
        # 如果x是(B, T, D)，需要unsqueeze
        if len(x.shape) == 3:  # (B, T, D)
            gamma = gamma.unsqueeze(1)  # (B, 1, D)
            beta = beta.unsqueeze(1)    # (B, 1, D)

        return gamma * x + beta


class FiLMedTransformerLayer(nn.Module):
    """
    带FiLM调制的Transformer层

    结构：
        x -> Self-Attention -> Add & Norm -> FiLM -> FFN -> Add & Norm -> out
    """
    def __init__(self, hidden_dim, num_heads, condition_dim, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.film = FiLM(condition_dim, hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, condition, mask=None):
        """
        Args:
            x: (B, T, hidden_dim)
            condition: (B, condition_dim)
            mask: attention mask

        Returns:
            out: (B, T, hidden_dim)
        """
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # FiLM调制
        x = self.film(x, condition)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
