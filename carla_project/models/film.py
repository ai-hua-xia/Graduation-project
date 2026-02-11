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

        # 初始化：严格恒等起点
        # gamma = 1 + 0*cond, beta = 0*cond
        nn.init.zeros_(self.gamma_layer.weight)
        nn.init.ones_(self.gamma_layer.bias)
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


def modulate(x, shift, scale):
    """
    AdaLN 风格调制：
        y = x * (1 + scale) + shift
    """
    if x.dim() == 3:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1.0 + scale) + shift


class AdaLNZeroTransformerLayer(nn.Module):
    """
    AdaLN-Zero Transformer layer

    结构（DiT风格）：
      x = x + gate_msa * Attn( AdaLN(x) )
      x = x + gate_mlp * MLP( AdaLN(x) )

    调制参数由条件向量生成，且 zero-init，使条件路径从“零影响”开始学习。
    """

    def __init__(self, hidden_dim, num_heads, condition_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # 保留命名，尽量兼容旧权重加载
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_dim * 6),
        )

        # Zero init: 条件分支从零开始，避免早期注入扰动主任务
        proj = self.adaLN_modulation[1]
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)

    def forward(self, x, condition, mask=None):
        mod = self.adaLN_modulation(condition)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        x_msa = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attention(x_msa, x_msa, x_msa, attn_mask=mask)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.ffn(x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x
