"""
诊断脚本：检查模型是否真的在预测，还是只是复制输入
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from models.vqvae_v2 import load_vqvae_v2_checkpoint
from models.world_model import WorldModel
from train.config import WM_CONFIG


def pick_default_world_model():
    candidates = [
        Path("checkpoints/world_model_v4_ss/best.pth"),
        Path("checkpoints/world_model_v4/best.pth"),
        Path("checkpoints/world_model_v3_ss/best.pth"),
        Path("checkpoints/world_model_v3/best.pth"),
        Path("checkpoints/world_model_ss/best.pth"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[-1])


def pick_default_token_file():
    candidates = [
        Path("data/tokens_action_corr_v2/tokens_actions.npz"),
        Path("data/tokens_action_corr/tokens_actions.npz"),
        Path("data/tokens_raw/tokens_actions.npz"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[-1])


def diagnose_model(vqvae_path, wm_path, token_file, device='cuda'):
    """诊断模型行为"""

    print("="*70)
    print("  World Model Diagnostic Tool")
    print("="*70)
    print()

    # 加载数据
    print("Loading data...")
    data = np.load(token_file)
    tokens = data['tokens']
    actions = data['actions']
    num_embeddings = int(tokens.max()) + 1

    # 加载模型
    print("Loading models...")
    vqvae, _ = load_vqvae_v2_checkpoint(vqvae_path, device)
    vqvae.eval()

    config = WM_CONFIG.copy()
    config['num_embeddings'] = num_embeddings
    world_model = WorldModel(
        num_embeddings=config['num_embeddings'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        context_frames=config['context_frames'],
        action_dim=config['action_dim'],
        tokens_per_frame=config['tokens_per_frame'],
        use_memory=config.get('use_memory', False),
        memory_dim=config.get('memory_dim', 256),
        dropout=config['dropout'],
    ).to(device)

    checkpoint = torch.load(wm_path, map_location=device, weights_only=False)
    world_model.load_state_dict(checkpoint['model_state_dict'])
    world_model.eval()

    context_frames = world_model.context_frames
    print(f"Context frames: {context_frames}")
    print()

    # 测试1: 检查预测是否与输入相同
    print("="*70)
    print("Test 1: Checking if model is copying input")
    print("="*70)

    num_tests = 10
    copy_count = 0
    different_count = 0

    with torch.no_grad():
        for i in range(num_tests):
            idx = np.random.randint(0, len(tokens) - context_frames - 1)

            # 准备输入
            context_tokens = tokens[idx:idx+context_frames]
            # 需要context_frames个动作，而不是单个动作
            action_seq = actions[idx:idx+context_frames]
            target_tokens = tokens[idx+context_frames]

            # 转换为tensor（确保数据类型正确）
            context_tensor = torch.from_numpy(context_tokens).long().unsqueeze(0).to(device)
            action_tensor = torch.from_numpy(action_seq).float().unsqueeze(0).to(device)

            # 预测
            logits = world_model(context_tensor, action_tensor)
            pred_tokens = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

            # 展平token以便比较（pred_tokens是(256,)，target_tokens是(16,16)）
            target_tokens_flat = target_tokens.flatten()

            # 检查是否与最后一帧相同
            last_frame_tokens = context_tokens[-1].flatten()
            is_copy = np.array_equal(pred_tokens, last_frame_tokens)

            # 检查是否与目标相同
            is_correct = np.array_equal(pred_tokens, target_tokens_flat)

            if is_copy:
                copy_count += 1
                print(f"  Sample {i+1}: ⚠️  COPYING last frame")
            elif is_correct:
                different_count += 1
                print(f"  Sample {i+1}: ✅ Correct prediction (different from input)")
            else:
                different_count += 1
                accuracy = np.mean(pred_tokens == target_tokens_flat)
                print(f"  Sample {i+1}: 🔄 Predicting (accuracy: {accuracy:.2%})")

    print()
    print(f"Summary:")
    print(f"  Copying input: {copy_count}/{num_tests} ({copy_count/num_tests*100:.1f}%)")
    print(f"  Making predictions: {different_count}/{num_tests} ({different_count/num_tests*100:.1f}%)")
    print()

    if copy_count > num_tests * 0.5:
        print("⚠️  WARNING: Model is mostly copying input!")
        print("   This explains the inf PSNR values.")
        print("   The model may not be learning to predict properly.")
    else:
        print("✅ Model is making genuine predictions.")
        print("   High PSNR is due to excellent prediction quality.")

    print()

    # 测试2: 检查不同动作的影响
    print("="*70)
    print("Test 2: Checking action sensitivity")
    print("="*70)

    idx = 100
    context_tokens = tokens[idx:idx+context_frames]
    context_tensor = torch.from_numpy(context_tokens).long().unsqueeze(0).to(device)

    # 获取基础动作序列
    base_actions = actions[idx:idx+context_frames]

    # 测试不同的最后一个动作（保持前面的动作不变）
    actions_to_test = [
        np.array([0.0, 0.5], dtype=np.float32),   # 直行
        np.array([0.5, 0.5], dtype=np.float32),   # 右转
        np.array([-0.5, 0.5], dtype=np.float32),  # 左转
    ]

    predictions = []
    with torch.no_grad():
        for test_action in actions_to_test:
            # 创建动作序列：前面保持不变，最后一个替换为测试动作
            action_seq = base_actions.copy()
            action_seq[-1] = test_action
            action_tensor = torch.from_numpy(action_seq).float().unsqueeze(0).to(device)
            logits = world_model(context_tensor, action_tensor)
            pred_tokens = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            predictions.append(pred_tokens)

    # 比较预测的差异
    diff_01 = np.mean(predictions[0] != predictions[1])
    diff_02 = np.mean(predictions[0] != predictions[2])
    diff_12 = np.mean(predictions[1] != predictions[2])

    print(f"  Straight vs Right turn: {diff_01:.2%} tokens different")
    print(f"  Straight vs Left turn:  {diff_02:.2%} tokens different")
    print(f"  Right vs Left turn:     {diff_12:.2%} tokens different")
    print()

    avg_diff = (diff_01 + diff_02 + diff_12) / 3
    if avg_diff < 0.05:
        print("⚠️  WARNING: Model is not sensitive to actions!")
        print("   Predictions are almost identical regardless of action.")
    else:
        print(f"✅ Model responds to actions (avg {avg_diff:.1%} difference)")

    print()
    print("="*70)
    print("Diagnostic complete!")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="World Model Diagnostic Tool")
    parser.add_argument("--vqvae-checkpoint", type=str, default="checkpoints/vqvae_v2/best.pth")
    parser.add_argument("--world-model-checkpoint", type=str, default=pick_default_world_model())
    parser.add_argument("--token-file", type=str, default=pick_default_token_file())
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    diagnose_model(
        vqvae_path=args.vqvae_checkpoint,
        wm_path=args.world_model_checkpoint,
        token_file=args.token_file,
        device=args.device,
    )
