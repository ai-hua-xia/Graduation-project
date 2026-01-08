import torch
import cv2
import numpy as np
import time
# 确保导入名和你的一致
from train_vqvae_256 import VQVAE, DEVICE 
from train_world_model import WorldModelGPT, BLOCK_SIZE

# ================= 配置 =================
VQVAE_PATH = "checkpoints_vqvae_256/vqvae_256_ep99.pth"
# 🔴 试着换一个稍微早一点的 GPT 权重，不要用 Loss < 1 的那个
WORLD_MODEL_PATH = "checkpoints_world_model/world_model_ep10.pth" # 举例，请修改

# 🔴 关键调整：极度保守的参数
TEMPERATURE = 0.1  # 极低温度
TOP_K = 5          # 只看前5名

def load_models():
    print(f"⏳ Loading GPT from {WORLD_MODEL_PATH}...")
    gpt = WorldModelGPT().to(DEVICE)
    # 使用 weights_only=False 忽略警告
    gpt.load_state_dict(torch.load(WORLD_MODEL_PATH, map_location=DEVICE, weights_only=False)["model"])
    gpt.eval()
    
    print(f"⏳ Loading VQ-VAE from {VQVAE_PATH}...")
    vqvae = VQVAE().to(DEVICE)
    vqvae.load_state_dict(torch.load(VQVAE_PATH, map_location=DEVICE, weights_only=False)["model"])
    vqvae.eval()
    return vqvae, gpt

def decode_indices(vqvae, indices):
    with torch.no_grad():
        indices_tensor = torch.LongTensor(indices).unsqueeze(0).to(DEVICE)
        z_q = vqvae.quantizer.embedding(indices_tensor).permute(0, 3, 1, 2)
        decoded_img = vqvae.decoder(z_q)
        img = decoded_img[0].cpu().permute(1, 2, 0).numpy()
        # 归一化修复：确保范围在 0-255
        img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255.0
        return img.astype(np.uint8)

def main():
    vqvae, gpt = load_models()
    
    # 手动造一个简单的启动 Token (模拟全黑或简单的起步)
    # 这里我们随机生成一个起始帧，看看能不能变成有意义的东西
    # 或者用全 0 (假设 0 是天空/背景)
    current_tokens = torch.randint(0, 100, (1, 1, 256)).to(DEVICE) # 随机噪声启动
    
    # 模拟“一直踩油门”的动作
    # 动作 (1, 1, 2) -> [转向0, 油门1]
    current_actions = torch.tensor([[[0.0, 1.0]]]).to(DEVICE)

    print("🚀 开始调试生成 (生成 1 帧)...")
    
    with torch.no_grad():
        # 构造输入
        pred_tokens_so_far = torch.zeros((1, 1, 256), dtype=torch.long).to(DEVICE)
        full_input_tokens = torch.cat([current_tokens, pred_tokens_so_far], dim=1)
        full_input_actions = torch.cat([current_actions, current_actions], dim=1) # 动作重复

        # 逐像素生成
        generated_indices = []
        for i in range(16): # 只生成前 16 个像素看看
            logits, _ = gpt(full_input_tokens, full_input_actions)
            
            # 获取当前位置的预测
            # 上下文长度 1帧(257) + 当前第i个
            target_idx = 1 * 257 + i - 1
            next_token_logits = logits[:, target_idx, :] / TEMPERATURE
            
            # Top-K 采样
            v, _ = torch.topk(next_token_logits, TOP_K)
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            
            val = idx.item()
            generated_indices.append(val)
            print(f"像素 {i}: 预测 Token ID = {val}")

    print(f"\n📊 预测的前16个 Token: {generated_indices}")
    print("👉 如果这些数字全是同一个数 (如 0,0,0) -> 模式崩塌 (Mode Collapse)")
    print("👉 如果这些数字非常随机且杂乱 -> 模型没训练好或 Temperature 太高")
    print("👉 正常情况下，应该是一组有规律变化的整数。")

if __name__ == "__main__":
    main()