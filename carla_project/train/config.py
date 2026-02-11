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
    # 模型参数 (f=8, L 配置 ~600–800M)
    'num_embeddings': 1024,
    'embed_dim': 896,
    'hidden_dim': 1792,
    'num_heads': 8,
    'num_layers': 20,
    'context_frames': 4,
    'action_dim': 2,
    'tokens_per_frame': 1024,  # 32×32 (f=8)
    'dropout': 0.1,
    'conditioning_type': 'adaln_zero',  # 'adaln_zero' or 'film'
    'use_action_aux': True,

    # 训练参数
    'lr': 3e-5,
    'epochs': 300,
    'batch_size': 8,
    'num_workers': 12,
    'pin_memory': True,
    'prefetch_factor': 4,
    'persistent_workers': True,
    'tqdm_mininterval': 15.0,  # 进度条刷新间隔（秒）
    'tqdm_miniters': 20,  # 进度条最小刷新步数
    'max_steps_per_epoch': 5000,  # 限制每个epoch的最大step数
    'stratified_ab': True,  # A/B分层采样（仅单卡）
    'ab_split': 750,  # A/B分界的episode id
    'lr_warmup_steps': 1500,  # 学习率预热步数（0=不启用）
    'max_grad_norm': 1.0,  # 梯度裁剪，稳定训练

    # 损失权重
    'ce_weight': 1.0,
    'smooth_weight_start': 0.0,  # 初始平滑权重
    'smooth_weight_end': 0.003,  # 最终平滑权重（降低跨样本过强耦合）
    'smooth_warmup_epochs': 80,  # 更慢预热，减少早期震荡
    'beta': 2.0,  # 动作自适应系数

    # 混合精度
    'use_amp': True,
    'amp_dtype': 'bf16',

    # 动作依赖增强
    'action_contrast_weight': 0.02,
    'action_contrast_margin': 0.03,
    'action_contrast_prob': 0.15,
    'action_contrast_mode': 'hinge',  # 'inverse' or 'hinge'
    'action_contrast_type': 'swap',  # 'swap' or 'noise'
    'action_noise_std_steer': 0.1,
    'action_noise_std_throttle': 0.05,
    # 动作注入/对比损失渐进开启（降低早期震荡）
    'action_inject_scale_start': 1.0,
    'action_inject_scale_end': 1.0,
    'action_inject_start_epoch': 0,
    'action_inject_warmup_epochs': 1,
    'action_contrast_weight_start': 0.0,
    'action_contrast_weight_end': 0.02,
    'action_contrast_start_epoch': 20,
    'action_contrast_warmup_epochs': 60,
    # 动作辅助损失（强制学习动作语义）
    'action_aux_weight_start': 0.03,
    'action_aux_weight_end': 0.12,
    'action_aux_warmup_epochs': 30,

    # 短rollout监督（缓解free-run漂移）
    'rollout_steps': 2,  # 额外预测2步（t+2, t+3）
    'rollout_weight_start': 0.02,
    'rollout_weight_end': 0.15,
    'rollout_start_epoch': 0,
    'rollout_warmup_epochs': 15,

    # 记忆模块
    'use_memory': True,
    'memory_dim': 512,

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
