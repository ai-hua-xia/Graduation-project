"""
训练配置文件
"""

# ============= VQ-VAE配置 =============
VQVAE_CONFIG = {
    # 模型参数
    'in_channels': 3,
    'base_channels': 64,
    'embed_dim': 256,
    'num_embeddings': 1024,
    'commitment_cost': 0.25,

    # 训练参数
    'lr': 2e-4,
    'epochs': 100,
    'batch_size': 64,
    'num_workers': 8,

    # 混合精度
    'use_amp': True,
    'amp_dtype': 'bf16',  # 'bf16' or 'fp16'

    # 保存
    'save_every': 5,
    'log_every': 100,
}

# ============= World Model配置 =============
WM_CONFIG = {
    # 模型参数 (~200M params，参考业界Transformer配置)
    'num_embeddings': 1024,
    'embed_dim': 512,        # 256 -> 512 (更大的embedding)
    'hidden_dim': 1024,      # 512 -> 1024
    'num_heads': 16,         # 8 -> 16
    'num_layers': 16,        # 8 -> 16
    'context_frames': 4,
    'action_dim': 2,
    'tokens_per_frame': 256,  # 16×16
    'dropout': 0.1,

    # 训练参数
    'lr': 5e-5,              # 大模型用更小学习率
    'epochs': 300,           # 更多epoch
    'batch_size': 16,        # 优化为32以更好利用显存
    'num_workers': 12,

    # 损失权重
    'ce_weight': 1.0,
    'smooth_weight_start': 0.0,  # 初始平滑权重
    'smooth_weight_end': 0.005,  # 最终平滑权重
    'smooth_warmup_epochs': 60,  # 平滑权重预热轮数
    'beta': 2.0,  # 动作自适应系数

    # 混合精度
    'use_amp': True,
    'amp_dtype': 'bf16',

    # 动作依赖增强
    'action_contrast_weight': 1.0,
    'action_contrast_margin': 1.0,
    'action_contrast_prob': 1.0,
    'action_contrast_mode': 'inverse',  # 'inverse' or 'hinge'
    'action_contrast_type': 'swap',  # 'swap' or 'noise'
    'action_noise_std_steer': 0.1,
    'action_noise_std_throttle': 0.05,

    # 记忆模块
    'use_memory': True,
    'memory_dim': 256,

    # 保存
    'save_every': 5,
    'log_every': 100,
}

# ============= 数据配置 =============
DATA_CONFIG = {
    'image_size': 256,
    'normalize_mean': [0.5, 0.5, 0.5],
    'normalize_std': [0.5, 0.5, 0.5],
}
